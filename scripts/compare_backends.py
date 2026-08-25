"""
Run the harness once per attention backend and put the speedups side by side,
so it's obvious which backend actually wins for each shape.

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

BACKENDS = ("sdpa", "custom")


def run(backend: str, args: list[str]):
    env = dict(os.environ, TTB_ATTN_BACKEND=backend)
    proc = subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True, text=True, cwd=REPO, env=env,
    )
    out = proc.stdout
    status = re.search(r"summary: (PASS|FAIL)", out)
    speed = re.search(r"speedup\s+: ([\d.]+)x", out)
    return (
        status.group(1) if status else "ERR",
        f"{speed.group(1)}x" if speed else "-",
    )


def main() -> int:
    header = f"{'config':<18}" + "".join(f"{b:>18}" for b in BACKENDS)
    print(header)
    print("-" * len(header))
    for label, args in CASES:
        cells = []
        for backend in BACKENDS:
            status, speed = run(backend, args)
            cells.append(f"{speed} ({status})")
        print(f"{label:<18}" + "".join(f"{c:>18}" for c in cells))
    print("-" * len(header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
