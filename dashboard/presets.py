"""Saved shapes, in an editable JSON file.

presets.json is re-read on every request rather than cached at import, so you
can add a shape in an editor and it appears on the next page load without
restarting the server. A malformed file falls back to the built-ins and reports
the parse error to the UI instead of taking the dashboard down.

A preset carries only shape fields (batch_size, seq_len, d_model, heads,
ffn_dim, layers, causal). It deliberately does not carry optimization settings:
selecting a shape should not silently change which kernel is being measured.

The Presets tab writes this file. Two things follow from that, and both are
handled in save():

  * Anything the UI sends is validated first, and a rejected save changes
    nothing. The alternative -- writing whatever arrives and finding out on the
    next page load -- turns a typo into a dashboard that will not start.

  * The write is atomic, via a temporary file and os.replace, with the previous
    contents kept as presets.json.bak. A half-written JSON file is the one
    failure mode that would lose every preset at once.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_PATH = os.path.join(HERE, "presets.json")

SHAPE_KEYS = ("batch_size", "seq_len", "d_model", "heads", "ffn_dim", "layers",
              "causal")

# Used when presets.json is missing or unreadable. Kept short on purpose --
# the file is the source of truth and is meant to be edited.
BUILTIN: List[Dict[str, Any]] = [
    {"name": "harness default", "batch_size": 8, "seq_len": 128, "d_model": 512,
     "heads": 8, "ffn_dim": 2048, "layers": 6, "causal": False,
     "note": "what torch_transformer_benchmark.py runs with no flags"},
    {"name": "smoke (fast)", "batch_size": 2, "seq_len": 64, "d_model": 128,
     "heads": 4, "ffn_dim": 128, "layers": 2, "causal": True,
     "note": "seconds, not minutes -- for checking the dashboard itself"},
]


def _clean(entry: Any) -> Dict[str, Any]:
    """One JSON entry -> a preset, or {} if it is not usable.

    Unknown keys are dropped rather than passed through, so a typo in the file
    cannot become a stray command-line flag.
    """
    if not isinstance(entry, dict):
        return {}
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        return {}
    preset: Dict[str, Any] = {"name": name.strip()}
    for key in SHAPE_KEYS:
        if key not in entry:
            continue
        value = entry[key]
        if key == "causal":
            preset[key] = bool(value)
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            preset[key] = value
    note = entry.get("note")
    if isinstance(note, str):
        preset["note"] = note
    return preset


def load() -> Dict[str, Any]:
    """{"presets": [...], "path": ..., "error": None | str}."""
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return {"presets": list(BUILTIN), "path": PRESETS_PATH,
                "error": "presets.json not found; using built-ins"}
    except (OSError, ValueError) as exc:
        return {"presets": list(BUILTIN), "path": PRESETS_PATH,
                "error": f"presets.json could not be read: {exc}"}

    entries = raw.get("presets") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return {"presets": list(BUILTIN), "path": PRESETS_PATH,
                "error": "presets.json must hold a list, or an object with a "
                         "'presets' list"}

    presets = [preset for preset in (_clean(entry) for entry in entries) if preset]
    if not presets:
        return {"presets": list(BUILTIN), "path": PRESETS_PATH,
                "error": "presets.json held no usable entries"}
    return {"presets": presets, "path": PRESETS_PATH, "error": None}


# Sane upper bounds. Not the GPU's limits -- preflight handles those, from the
# harness's own rules -- just enough that a slipped keystroke cannot queue a
# shape that will sit there for a week.
_MAX = {
    "batch_size": 1_000_000,
    "seq_len": 1_000_000,
    "d_model": 65_536,
    "heads": 4_096,
    "ffn_dim": 262_144,
    "layers": 512,
}
_REQUIRED = ("batch_size", "seq_len", "d_model", "heads", "ffn_dim", "layers")


def validate(entries: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Check a list of presets from the UI. Returns (clean, problems).

    Problems carry the row index so the editor can mark the offending field
    rather than showing one message for the whole table. `clean` is only
    meaningful when `problems` is empty -- a partial save is worse than none,
    because it silently drops the rows that failed.
    """
    problems: List[Dict[str, Any]] = []
    clean: List[Dict[str, Any]] = []

    def fail(index: int, field: str, message: str) -> None:
        problems.append({"row": index, "field": field, "message": message})

    if not isinstance(entries, list):
        return [], [{"row": -1, "field": "", "message": "expected a list of presets"}]
    if not entries:
        return [], [{"row": -1, "field": "",
                     "message": "a preset file with no presets in it would leave "
                                "the shape list empty"}]

    seen: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(index, "", "not an object")
            continue

        name = entry.get("name")
        name = name.strip() if isinstance(name, str) else ""
        if not name:
            fail(index, "name", "needs a name")
        elif name in seen:
            fail(index, "name", f"duplicate of row {seen[name] + 1}; names pick "
                                f"the preset out of a list, so they must differ")
        else:
            seen[name] = index

        preset: Dict[str, Any] = {"name": name}
        for key in _REQUIRED:
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(index, key, "must be a whole number")
                continue
            if value != int(value):
                fail(index, key, "must be a whole number")
                continue
            value = int(value)
            if value < 1:
                fail(index, key, "must be at least 1")
            elif value > _MAX[key]:
                fail(index, key, f"is above the {_MAX[key]:,} cap")
            else:
                preset[key] = value

        # The harness's own rule, checked here so a preset cannot be saved that
        # TransformerConfig.validate() would reject the moment it is run.
        if "d_model" in preset and "heads" in preset:
            if preset["d_model"] % preset["heads"]:
                fail(index, "heads", f"d_model {preset['d_model']} is not "
                                     f"divisible by heads {preset['heads']}")

        preset["causal"] = bool(entry.get("causal"))
        note = entry.get("note")
        if isinstance(note, str) and note.strip():
            preset["note"] = note.strip()
        clean.append(preset)

    return (clean, problems) if not problems else ([], problems)


def save(entries: Any) -> Dict[str, Any]:
    """Validate and write presets.json. Returns {"ok", "problems", "error"}.

    The file's `_comment` block is carried over rather than rewritten: it
    documents the format for whoever opens the file in an editor, and losing it
    to a UI save would be a slow-acting papercut.
    """
    clean, problems = validate(entries)
    if problems:
        return {"ok": False, "problems": problems, "error": None}

    comment = None
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            comment = existing.get("_comment")
    except (OSError, ValueError):
        pass

    payload: Dict[str, Any] = {}
    if comment is not None:
        payload["_comment"] = comment
    payload["presets"] = clean

    directory = os.path.dirname(PRESETS_PATH)
    try:
        os.makedirs(directory, exist_ok=True)
        # Keep the previous file before replacing it. Cheap, and the only thing
        # standing between a bad save and retyping fourteen shapes.
        if os.path.exists(PRESETS_PATH):
            backup = PRESETS_PATH + ".bak"
            with open(PRESETS_PATH, "r", encoding="utf-8") as src:
                previous = src.read()
            with open(backup, "w", encoding="utf-8") as dst:
                dst.write(previous)

        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix="presets-", suffix=".tmp",
            delete=False)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        # Atomic on both Windows and POSIX: readers see either file whole.
        os.replace(handle.name, PRESETS_PATH)
    except OSError as exc:
        return {"ok": False, "problems": [], "error": f"could not write: {exc}"}

    return {"ok": True, "problems": [], "error": None, "count": len(clean)}
