"""Run the harness over the grading shapes, one child process at a time.

Reads dashboard/presets.json, runs torch_transformer_benchmark.py per shape and
parses its summary lines. Nothing is timed in this process -- the numbers are
the harness's own -- so the usual rule still holds: compare rows only against
rows from the same invocation of this script.

  python scripts/sweep_shapes.py                 # shapes 1-13
  python scripts/sweep_shapes.py --shapes 1,9,13
  python scripts/sweep_shapes.py -- --attn-impl wmma
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEEDUP_RE = re.compile(r"^speedup\s*:\s*([0-9.]+)x")
SUMMARY_RE = re.compile(r"^summary:\s*(\w+)\s*\|\s*max_abs=([0-9.eE+-]+)")
OPT_RE = re.compile(r"^optimized:\s*median=([0-9.]+) ms")
BASE_RE = re.compile(r"^baseline\s*:\s*median=([0-9.]+) ms")


def run_shape(preset, extra, rtol, atol):
    cmd = [
        sys.executable, "torch_transformer_benchmark.py",
        "--batch-size", str(preset["batch_size"]),
        "--seq-len", str(preset["seq_len"]),
        "--d-model", str(preset["d_model"]),
        "--heads", str(preset["heads"]),
        "--ffn-dim", str(preset["ffn_dim"]),
        "--layers", str(preset["layers"]),
        "--rtol", str(rtol), "--atol", str(atol),
    ]
    if preset.get("causal"):
        cmd.append("--causal")
    cmd += extra
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    row = {"speedup": None, "verdict": "?", "max_abs": None, "opt": None, "base": None}
    for line in out.splitlines():
        m = SPEEDUP_RE.match(line)
        if m:
            row["speedup"] = float(m.group(1))
        m = SUMMARY_RE.match(line)
        if m:
            row["verdict"], row["max_abs"] = m.group(1), float(m.group(2))
        m = OPT_RE.match(line)
        if m:
            row["opt"] = float(m.group(1))
        m = BASE_RE.match(line)
        if m:
            row["base"] = float(m.group(1))
    if p.returncode != 0 and row["speedup"] is None:
        row["verdict"] = "CRASH"
        sys.stderr.write(out[-2000:] + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1-13")
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--atol", type=float, default=0.002)
    ap.add_argument("--tag", default="")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    wanted = set()
    for part in args.shapes.split(","):
        if "-" in part:
            a, b = part.split("-")
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))

    with open(os.path.join(ROOT, "dashboard", "presets.json")) as f:
        presets = json.load(f)["presets"]

    print(f"# sweep {args.tag} gate rtol={args.rtol} atol={args.atol} extra={extra}")
    print(f"{'shape':<18} {'base ms':>9} {'opt ms':>9} {'speedup':>8} {'acc':>6} {'max_abs':>11}")
    rows = []
    for i, preset in enumerate(presets, 1):
        if i not in wanted:
            continue
        row = run_shape(preset, extra, args.rtol, args.atol)
        rows.append((i, preset["name"], row))
        print(f"{preset['name']:<18} "
              f"{row['base'] if row['base'] is not None else float('nan'):>9.4f} "
              f"{row['opt'] if row['opt'] is not None else float('nan'):>9.4f} "
              f"{(row['speedup'] or float('nan')):>7.3f}x "
              f"{row['verdict']:>6} "
              f"{(row['max_abs'] if row['max_abs'] is not None else float('nan')):>11.3e}",
              flush=True)

    good = [r[2]["speedup"] for r in rows if r[2]["speedup"]]
    if good:
        gm = math.exp(sum(math.log(s) for s in good) / len(good))
        print(f"\ngeomean speedup over {len(good)} shapes: {gm:.4f}x")
    bad = [r[1] for r in rows if r[2]["verdict"] not in ("PASS",)]
    print("non-PASS: " + (", ".join(bad) if bad else "none"))


if __name__ == "__main__":
    main()
