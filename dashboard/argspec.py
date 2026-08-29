"""The harness's command-line flags, read out of its source with `ast`.

The obvious way to enumerate the flags would be to import the modules and walk
the ArgumentParser. That is not available here: importing `optimized.cli` pulls
in torch, which creates a CUDA context, which is exactly what this dashboard
promises not to do. So the two files that call `add_argument` are parsed as
source instead and never executed.

The payoff is that the form stays in sync by itself. Add a flag to
optimized/cli.py and it appears in the dashboard on the next page load, with its
own help text as the tooltip -- there is no second copy of the flag list to
forget to update.

Both source files use literal keywords throughout (`choices=("a", "b")`,
`default=8`, `type=int`, `action="store_true"`), which is what makes this
tractable. Anything non-literal is reported as `unsupported` rather than
guessed at, and FALLBACK below covers the case where the extraction finds
nothing at all -- a renamed function, a moved file, an argparse call built in a
loop. A stale fallback is better than an empty form.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (path relative to the repo root, name of the function whose body to read,
#  group label for the UI). The harness builds its parser in parse_args() and
# then hands it to optimized.cli.add_arguments(), so the two together are the
# complete flag set -- the same set `python torch_transformer_benchmark.py -h`
# would print.
SOURCES = [
    ("torch_transformer_benchmark.py", "parse_args", "harness"),
    (os.path.join("optimized", "cli.py"), "add_arguments", "optimization"),
]

# Which UI section each flag belongs in. Flags not named here land in "other",
# which is rendered as a collapsed advanced section, so a newly added flag shows
# up somewhere sensible without this table needing an edit.
GROUPS = {
    "shape": ["--batch-size", "--seq-len", "--d-model", "--heads", "--ffn-dim",
              "--layers", "--causal"],
    "data": ["--dtype", "--padding-ratio", "--input-scale", "--seed", "--device"],
    "accuracy": ["--accuracy-trials", "--rtol", "--atol", "--benchmark-on-failure"],
    "timing": ["--warmup", "--repeats", "--benchmark-rounds"],
    "optimization": ["--attn-backend", "--attn-impl", "--attn-fp16",
                     "--linear-gelu", "--cuda-graph"],
    "torch": ["--matmul-precision", "--allow-tf32", "--compile-baseline",
              "--compile-user", "--compile-mode", "--non-strict-weight-copy"],
}


def _group_for(flag: str, default: str) -> str:
    for name, flags in GROUPS.items():
        if flag in flags:
            return name
    return default if default != "harness" else "other"


def _literal(node: ast.AST) -> Any:
    """A literal value, or the sentinel `_UNSUPPORTED` for anything else.

    ast.literal_eval raises on names and attributes, which is precisely the
    case that matters: `action=argparse.BooleanOptionalAction` and
    `type=int` are both non-literal and both need handling, so they are
    reported rather than swallowed.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _UNSUPPORTED


class _Unsupported:
    def __repr__(self) -> str:
        return "<unsupported>"


_UNSUPPORTED = _Unsupported()


def _dotted_name(node: ast.AST) -> Optional[str]:
    """`argparse.BooleanOptionalAction` -> "argparse.BooleanOptionalAction"."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _parse_add_argument(call: ast.Call, group_default: str) -> Optional[Dict[str, Any]]:
    """One `parser.add_argument(...)` call -> a form-field description."""
    flags = [a.value for a in call.args
             if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    long_flags = [f for f in flags if f.startswith("--")]
    if not long_flags:
        # Positional arguments; the harness has none and the dashboard has no
        # way to render one usefully.
        return None
    flag = long_flags[0]

    spec: Dict[str, Any] = {
        "flag": flag,
        "dest": flag[2:].replace("-", "_"),
        "kind": "str",
        "choices": None,
        "default": None,
        "help": "",
        "group": _group_for(flag, group_default),
    }

    for kw in call.keywords:
        if kw.arg == "choices":
            value = _literal(kw.value)
            if value is not _UNSUPPORTED:
                spec["choices"] = list(value)
        elif kw.arg == "default":
            value = _literal(kw.value)
            spec["default"] = None if value is _UNSUPPORTED else value
        elif kw.arg == "help":
            value = _literal(kw.value)
            if isinstance(value, str):
                # argparse help is wrapped across source lines; collapse it so
                # it fits a tooltip.
                spec["help"] = " ".join(value.split())
        elif kw.arg == "type":
            name = _dotted_name(kw.value)
            if name in ("int", "float"):
                spec["kind"] = name
        elif kw.arg == "action":
            value = _literal(kw.value)
            if value == "store_true":
                spec["kind"] = "flag"
            elif _dotted_name(kw.value) == "argparse.BooleanOptionalAction":
                # --allow-tf32 / --no-allow-tf32. Rendered as a tri-state so
                # "leave the harness default alone" stays expressible.
                spec["kind"] = "tristate"

    if spec["choices"] is not None and spec["kind"] not in ("int", "float"):
        spec["kind"] = "choice"
    if spec["kind"] == "flag" and spec["default"] is None:
        # store_true defaults to False; argparse supplies that implicitly, so
        # the source has no `default=` for the UI to have read.
        spec["default"] = False
    return spec


def extract(repo: str = REPO) -> List[Dict[str, Any]]:
    """Every flag the harness accepts, in source order.

    Never raises: an unreadable or restructured source file yields no fields
    from that file, and `load()` substitutes the fallback when the result comes
    back implausibly short.
    """
    specs: List[Dict[str, Any]] = []
    seen = set()
    for relative, function, group in SOURCES:
        path = os.path.join(repo, relative)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
        except (OSError, SyntaxError):
            continue
        target = _find_function(tree, function)
        if target is None:
            continue
        for node in ast.walk(target):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            spec = _parse_add_argument(node, group)
            if spec is not None and spec["flag"] not in seen:
                seen.add(spec["flag"])
                specs.append(spec)
    return specs


# Used only when extraction comes back short, which means the harness moved out
# from under this file. Deliberately minimal: enough to keep the dashboard
# usable, not a second maintained copy of the whole flag list.
FALLBACK: List[Dict[str, Any]] = [
    {"flag": "--batch-size", "dest": "batch_size", "kind": "int", "choices": None,
     "default": 8, "help": "", "group": "shape"},
    {"flag": "--seq-len", "dest": "seq_len", "kind": "int", "choices": None,
     "default": 128, "help": "", "group": "shape"},
    {"flag": "--d-model", "dest": "d_model", "kind": "int", "choices": None,
     "default": 512, "help": "", "group": "shape"},
    {"flag": "--heads", "dest": "heads", "kind": "int", "choices": None,
     "default": 8, "help": "", "group": "shape"},
    {"flag": "--ffn-dim", "dest": "ffn_dim", "kind": "int", "choices": None,
     "default": 2048, "help": "", "group": "shape"},
    {"flag": "--layers", "dest": "layers", "kind": "int", "choices": None,
     "default": 6, "help": "", "group": "shape"},
    {"flag": "--causal", "dest": "causal", "kind": "flag", "choices": None,
     "default": False, "help": "", "group": "shape"},
    {"flag": "--dtype", "dest": "dtype", "kind": "choice",
     "choices": ["float32", "float16", "bfloat16"], "default": "float32",
     "help": "", "group": "data"},
    {"flag": "--attn-backend", "dest": "attn_backend", "kind": "choice",
     "choices": ["auto", "sdpa", "custom"], "default": None, "help": "",
     "group": "optimization"},
    {"flag": "--attn-impl", "dest": "attn_impl", "kind": "choice",
     "choices": ["auto", "scalar", "wmma", "tile", "tile-bf16", "tile-tf32",
                 "tile-fp16"],
     "default": None, "help": "", "group": "optimization"},
    {"flag": "--attn-fp16", "dest": "attn_fp16", "kind": "choice",
     "choices": ["auto", "tf32"], "default": None, "help": "",
     "group": "optimization"},
    {"flag": "--linear-gelu", "dest": "linear_gelu", "kind": "choice",
     "choices": ["auto", "tf32", "off"], "default": None, "help": "",
     "group": "optimization"},
    {"flag": "--cuda-graph", "dest": "cuda_graph", "kind": "choice",
     "choices": ["off", "auto", "always"], "default": None, "help": "",
     "group": "optimization"},
]

# Below this the extraction is assumed to have failed rather than to have found
# a genuinely tiny parser. The real harness has 20-odd flags plus five from
# optimized/cli.py.
_MIN_PLAUSIBLE = 10


def load(repo: str = REPO) -> Dict[str, Any]:
    """Flags plus a note about where they came from, for the UI to display."""
    specs = extract(repo)
    source = "ast"
    if len(specs) < _MIN_PLAUSIBLE:
        specs, source = list(FALLBACK), "fallback"

    constants = config_values(repo)
    binding = config_binding(repo)
    for spec in specs:
        constant = binding.get(spec["dest"])
        entry = constants.get(constant) if constant else None
        if entry is not None:
            # The value the run will actually use, and where it is written down,
            # so the UI never has to say the unhelpful word "default".
            spec["effective"] = entry["value"]
            spec["effective_source"] = f"optimized/config.py: {constant}"
            spec["choice_help"] = entry["legend"]
        else:
            spec["effective"] = spec["default"]
            spec["effective_source"] = ("torch_transformer_benchmark.py"
                                        if spec["default"] is not None else "")
            spec["choice_help"] = {}
    return {"fields": specs, "source": source, "count": len(specs)}


# --- what a flag actually resolves to when you do not pass it ---------------
#
# argparse reports `default=None` for the five optimization flags, because the
# real value lives in optimized/config.py and cli.py only overrides it when the
# flag is given. "None" is useless in a dropdown -- it says nothing about what
# the run will do. The three functions below dig out the value that is really in
# force, and the per-value explanations config.py already carries beside it.

# A legend entry in config.py: three spaces, a quoted value, then prose.
#
#     #   "auto"     custom CUDA kernel when it builds and loads, else SDPA
#     #   "sdpa"     always F.scaled_dot_product_attention. No build required.
#
# Continuation lines are indented past where the prose started, which is what
# separates them from the next entry.
# Two or more spaces after the "#" is required, and it is load-bearing: the
# prose below a legend quotes the same values inline ("tf32" here is
# BIT-IDENTICAL to F.linear + F.gelu ...), and with a looser pattern those
# sentences overwrite the real entry. Legend lines are indented; prose is not.
_LEGEND_ENTRY = re.compile(r'^#\s{2,}"([\w.-]+)"(\s+)(\S.*)$')
_COMMENT_TEXT = re.compile(r"^#\s?(.*)$")


def _legend_above(lines: List[str], lineno: int) -> Dict[str, str]:
    """Parse the `"value"  explanation` block in the comment above a constant.

    `lineno` is 1-based, as ast reports it. Walks up while the lines are
    comments, then reads the block downwards.
    """
    start = lineno - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    block = lines[start:lineno - 1]

    legend: Dict[str, str] = {}
    current: Optional[str] = None
    prose_column = 0
    started = False
    for raw in block:
        raw = raw.rstrip()
        match = _LEGEND_ENTRY.match(raw)
        if match:
            current = match.group(1)
            prose_column = raw.index(match.group(3))
            legend.setdefault(current, match.group(3).strip())
            started = True
            continue

        text = _COMMENT_TEXT.match(raw)
        blank = not text or not text.group(1).strip()
        if blank:
            # A blank comment line closes the legend. Everything after it is
            # prose about the setting as a whole -- often several paragraphs of
            # it, quoting the same values inline -- so stop rather than let any
            # of that be read as an entry.
            if started:
                break
            current = None
            continue

        if current is None:
            continue
        if len(raw) > prose_column and not raw[:prose_column].strip("# "):
            legend[current] += " " + text.group(1).strip()
        else:
            current = None
    return legend


def config_values(repo: str = REPO) -> Dict[str, Dict[str, Any]]:
    """Module-level string constants in optimized/config.py, with their legends.

    Returns {CONSTANT: {"value": ..., "legend": {choice: explanation}}}.
    """
    path = os.path.join(repo, "optimized", "config.py")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
    except (OSError, SyntaxError):
        return {}
    lines = source.splitlines()

    found: Dict[str, Dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        found[target.id] = {
            "value": node.value.value,
            "legend": _legend_above(lines, node.lineno),
        }
    return found


def numeric_constants(relative: str, names: List[str],
                      repo: str = REPO) -> Dict[str, Any]:
    """Module-level int/float constants, read out of a source file.

    The harness decides for itself how to split a batch and whether the
    baseline can run, and it does that from two module constants. The dashboard
    has to predict the same answers to say up front whether a shape is runnable
    -- so it reads the constants rather than keeping its own copies, which would
    silently disagree the first time one is retuned.
    """
    path = os.path.join(repo, relative)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return {}
    wanted = set(names)
    found: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = _literal(node.value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            found[target.id] = value
    return found


def config_binding(repo: str = REPO) -> Dict[str, str]:
    """Which config constant each flag overrides, read from apply_overrides().

    optimized/cli.py says it in the plainest possible way:

        if args.attn_backend is not None:
            config.ATTENTION_BACKEND = args.attn_backend

    so the mapping is `args.<dest>` -> `config.<CONSTANT>`. Reading it rather
    than writing it down keeps this correct when a flag is renamed.
    """
    path = os.path.join(repo, "optimized", "cli.py")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return {}
    target_fn = _find_function(tree, "apply_overrides")
    if target_fn is None:
        return {}

    binding: Dict[str, str] = {}
    for node in ast.walk(target_fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        left, right = node.targets[0], node.value
        if not (isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name)
                and left.value.id == "config"):
            continue
        if not (isinstance(right, ast.Attribute) and isinstance(right.value, ast.Name)
                and right.value.id == "args"):
            continue
        binding[right.attr] = left.attr
    return binding


def module_summary(path: str) -> str:
    """A script's own first docstring line, for the listing.

    Every script in scripts/ opens with a docstring saying what it measures,
    which is a better description than anything this could invent.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def script_arguments(path: str) -> List[Dict[str, Any]]:
    """Flags for one script in scripts/, or [] when it has no argparse.

    Half the scripts in scripts/ take no arguments at all and several build
    their parser at module level rather than inside a function, so this walks
    the whole module instead of looking for a named function. Scripts with no
    flags get a free-text box in the UI instead.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return []
    specs = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        spec = _parse_add_argument(node, "script")
        if spec is not None and spec["flag"] not in seen:
            seen.add(spec["flag"])
            spec["group"] = "script"
            specs.append(spec)
    return specs
