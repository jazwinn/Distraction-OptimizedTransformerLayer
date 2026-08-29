"""
Run the harness once per attention backend and put the speedups side by side,
so it's obvious which backend actually wins for each shape. The custom backend
gets one column per kernel (scalar vs tensor-core) rather than one column for
whichever "auto" happened to choose.

    cmd.exe /c scripts\\devenv.bat python scripts\\compare_backends.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "torch_transformer_benchmark.py")

CASES = [
    ("default",            []),
    ("causal",             ["--causal"]),
    ("padded",             ["--padding-ratio", "0.3"]),
    ("causal+padded",      ["--causal", "--padding-ratio", "0.3"]),
    ("seq512",             ["--seq-len", "512", "--batch-size", "4"]),
    ("seq2048",            ["--seq-len", "2048", "--batch-size", "1"]),
    ("seq2048 causal",     ["--seq-len", "2048", "--batch-size", "1", "--causal"]),
    ("small b1 s32",       ["--batch-size", "1", "--seq-len", "32"]),
    ("wide d1024",         ["--d-model", "1024", "--heads", "16", "--ffn-dim", "4096"]),
    ("deep 12L",           ["--layers", "12"]),
]

# (column label, extra flags). Every column is a kernel this project wrote:
# --attn-backend has one legal value now, so the columns vary --attn-impl and
# the scalar kernel is timed separately from the tensor-core one, which is the
# comparison that matters. There used to be an "sdpa" column here; it went with
# the backend flag that selected it, since a prebuilt attention is no longer
# reachable from the model.
BACKENDS = (
    ("custom scalar", ["--attn-backend", "custom", "--attn-impl", "scalar"]),
    ("custom wmma",   ["--attn-backend", "custom", "--attn-impl", "wmma"]),
    ("custom tile",   ["--attn-backend", "custom", "--attn-impl", "tile"]),
    ("custom tile-tf32", ["--attn-backend", "custom", "--attn-impl", "tile-tf32"]),
    ("custom tile-bf16", ["--attn-backend", "custom", "--attn-impl", "tile-bf16"]),
)

# A full sweep is len(CASES) * len(BACKENDS) harness invocations, so restrict the
# columns when only some are interesting:
#   COMPARE_BACKENDS="custom wmma,custom tile-tf32"
# Names must match the column labels above; an unknown one is an error rather
# than a silently empty table.
_want = os.environ.get("COMPARE_BACKENDS", "").strip()
if _want:
    _names = [n.strip() for n in _want.split(",") if n.strip()]
    _known = {b for b, _ in BACKENDS}
    _bad = [n for n in _names if n not in _known]
    if _bad:
        raise SystemExit(f"unknown backend(s) {_bad}; choose from {sorted(_known)}")
    BACKENDS = tuple((b, a) for b, a in BACKENDS if b in _names)


# The harness is re-entered once per (config, backend); its own defaults spend
# most of that on repeated accuracy trials, which this table does not report
# beyond PASS/FAIL. FAST trims them so a full sweep is minutes rather than an
# hour -- set COMPARE_FULL=1 to use the harness defaults instead.
FAST = [] if os.environ.get("COMPARE_FULL") else ["--accuracy-trials", "2"]


def run(backend_args: list[str], args: list[str]):
    proc = subprocess.run(
        [sys.executable, SCRIPT, *backend_args, *FAST, *args],
        capture_output=True, text=True, cwd=REPO,
    )
    out = proc.stdout
    status = re.search(r"summary: (PASS|FAIL)", out)
    speed = re.search(r"speedup\s+: ([\d.]+)x", out)
    return (
        status.group(1) if status else "ERR",
        f"{speed.group(1)}x" if speed else "-",
    )


def main() -> int:
    header = f"{'config':<18}" + "".join(f"{b:>18}" for b, _ in BACKENDS)
    print(header)
    print("-" * len(header))
    for label, args in CASES:
        cells = []
        for _, backend_args in BACKENDS:
            status, speed = run(backend_args, args)
            cells.append(f"{speed} ({status})")
        print(f"{label:<18}" + "".join(f"{c:>18}" for c in cells), flush=True)
    print("-" * len(header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
