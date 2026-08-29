"""A form submission -> an argv list and an environment.

This is the only place that turns UI state into something executable, and it is
deliberately strict: every value is checked against the flag spec that argspec
extracted, and anything unrecognised is dropped rather than passed through.
The server binds to loopback only, but a local web page is still an untrusted
input surface, and `subprocess` is invoked with a list and never a shell string,
so there is nothing for a stray character to escape into.

The two rules that make the generated command trustworthy:

* **A field left at its default contributes nothing.** An empty form produces
  exactly `python -u torch_transformer_benchmark.py`, the same command the
  harness would get by hand. That keeps the dashboard from silently pinning a
  value the harness would otherwise choose for itself -- which matters most for
  the optimization flags, whose real defaults live in optimized/config.py, not
  in argparse.

* **The command is shown before it runs.** `describe()` renders exactly what
  will be spawned, environment included, so a number in the table can always be
  traced back to a command you could paste into a terminal yourself.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import argspec, knobs

REPO = argspec.REPO
HARNESS = os.path.join(REPO, "torch_transformer_benchmark.py")


@dataclass
class RunSpec:
    """One child process: what to run, where, and with what environment."""

    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = REPO
    label: str = ""

    def command_line(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)

    def describe(self) -> str:
        """The command as you would type it, environment prefix included."""
        prefix = "".join(f"{name}={value} " for name, value
                         in sorted(self.env.items()))
        return prefix + self.command_line()


def _flag_index() -> Dict[str, Dict[str, Any]]:
    return {spec["dest"]: spec for spec in argspec.load()["fields"]}


def _coerce(spec: Dict[str, Any], value: Any) -> Optional[Any]:
    """A submitted value, checked against its flag spec. None means 'skip'."""
    kind = spec["kind"]
    if value is None or value == "":
        return None

    if kind == "flag":
        return True if value in (True, "true", "on", 1, "1") else None

    if kind == "tristate":
        if value in (True, "true", "on", 1, "1"):
            return True
        if value in (False, "false", "off", 0, "0"):
            return False
        return None

    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if kind == "choice":
        text = str(value)
        return text if text in (spec["choices"] or []) else None

    return str(value)


def build_argv(form: Dict[str, Any], script: str = HARNESS) -> List[str]:
    """Form -> argv, dropping every field already set to what it would be.

    "Already set to what it would be" means the *effective* value, not
    argparse's `default`. For the five optimization flags argparse says
    default=None while the real value sits in optimized/config.py, so comparing
    against the default would emit `--attn-impl auto` for a form nobody touched
    -- a flag that changes nothing, on every command, forever.

    Since the dropdowns preselect the effective value, this is what keeps an
    untouched form producing a bare `python -u torch_transformer_benchmark.py`.

    `-u` is what makes live streaming work: piped stdout is block-buffered
    otherwise, and the log pane would sit empty until the run finished. It
    changes only buffering, never the numbers.
    """
    argv = [sys.executable, "-u", script]
    index = _flag_index()

    for dest, spec in index.items():
        value = _coerce(spec, form.get(dest))
        if value is None:
            continue

        # The value the run uses if this flag is absent. Falls back to
        # argparse's own default for flags with no config.py constant behind
        # them.
        effective = spec.get("effective", spec["default"])
        kind = spec["kind"]

        if kind == "flag":
            # store_true: present means on. Only emit when it is not already
            # what the harness would do (always False for these).
            if value and not effective:
                argv.append(spec["flag"])
            continue

        if kind == "tristate":
            # --allow-tf32 / --no-allow-tf32. Emitting the one already in force
            # would be harmless but noisy in the previewed command.
            if value == effective:
                continue
            argv.append(spec["flag"] if value
                        else "--no-" + spec["flag"][2:])
            continue

        if effective is not None and str(value) == str(effective):
            continue
        argv.extend([spec["flag"], str(value)])

    return argv


def build_env(form: Dict[str, Any]) -> Dict[str, str]:
    """The environment-variable knobs, minus any left at their default.

    Only the knobs that were actually moved appear, so the previewed command
    stays readable and a default run adds nothing to the child's environment.
    PYTHONUNBUFFERED is added unconditionally as the belt to `-u`'s braces.
    """
    env: Dict[str, str] = {}
    submitted = form.get("env") or {}
    if not isinstance(submitted, dict):
        return env

    for name, value in submitted.items():
        resolved = knobs.env_value(name, value)
        if resolved is not None:
            env[name] = resolved
    return env


def for_harness(form: Dict[str, Any], label: str = "") -> RunSpec:
    return RunSpec(argv=build_argv(form), env=build_env(form), label=label)


DEVENV = os.path.join(REPO, "scripts", "devenv.bat")


def through_devenv(argv: List[str]) -> List[str]:
    """Run argv with MSVC on PATH, via the repository's own helper.

    Invoked as `cmd.exe /c scripts\\devenv.bat <argv>`, which is the form
    devenv.bat documents and the only one it supports -- it deliberately omits
    setlocal so the environment it sets stays live for the command, which only
    holds when it is the sole command of a one-shot cmd.
    """
    if not os.path.isfile(DEVENV):
        return argv
    return ["cmd.exe", "/c", DEVENV] + argv


def script_path(name: str) -> Optional[str]:
    """Resolve a scripts/ entry by bare filename, or None if it is not one.

    The name is matched against an actual directory listing rather than merely
    sanitised, so nothing outside scripts/ is reachable however the name is
    spelled.
    """
    if not name or os.path.sep in name or "/" in name or name.startswith("."):
        return None
    directory = os.path.join(REPO, "scripts")
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    if name not in entries or not name.endswith(".py"):
        return None
    path = os.path.join(directory, name)
    return path if os.path.isfile(path) else None


def for_script(name: str, form: Dict[str, Any], extra: str = "",
               label: str = "") -> Optional[RunSpec]:
    """One scripts/ entry, with its own argparse-derived flags.

    Half of scripts/ takes no arguments at all, so `extra` is a free-text box
    parsed with shlex -- the escape hatch for those, and for flags this has no
    spec for. It is split into a list, never handed to a shell.
    """
    path = script_path(name)
    if path is None:
        return None

    argv = [sys.executable, "-u", path]
    for spec in argspec.script_arguments(path):
        value = _coerce(spec, form.get(spec["dest"]))
        if value is None:
            continue
        if spec["kind"] == "flag":
            if value and not spec["default"]:
                argv.append(spec["flag"])
            continue
        if spec["kind"] == "tristate":
            if value == spec["default"]:
                continue
            argv.append(spec["flag"] if value else "--no-" + spec["flag"][2:])
            continue
        if spec["default"] is not None and value == spec["default"]:
            continue
        argv.extend([spec["flag"], str(value)])

    if extra.strip():
        try:
            argv.extend(shlex.split(extra))
        except ValueError:
            # Unbalanced quotes; better to run the script without the extras
            # than to refuse, and the previewed command shows what happened.
            pass

    return RunSpec(argv=argv, env=build_env(form), label=label or name)
