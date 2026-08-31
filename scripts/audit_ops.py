"""Per-kernel GPU time for the optimized model on one grading shape.

Answers "where does the remaining time actually go" -- which is what decides
what is worth optimizing next. CUDA graphs are forced off first: a captured
graph reports as one opaque `cudaGraphLaunch` and every kernel inside it
vanishes from the table, so an audit with graphs on silently under-reports.

    python scripts/audit_ops.py --shape 1
    python scripts/audit_ops.py --shape 8 --shape 9 --shape 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from optimized import config as ocfg  # noqa: E402

ocfg.CUDA_GRAPH = "off"

# Set before any model is built, same as optimized/cli.py does it.
for _i, _a in enumerate(sys.argv):
    if _a == "--linear-bias":
        ocfg.LINEAR_BIAS = sys.argv[_i + 1]
    if _a == "--linear-gelu":
        ocfg.LINEAR_GELU = sys.argv[_i + 1]

from torch_transformer_benchmark import (  # noqa: E402
    TransformerConfig, UserOptimizedTransformer, generate_random_case)


def audit(preset, iters, warmup):
    cfg = TransformerConfig(
        batch_size=preset["batch_size"], seq_len=preset["seq_len"],
        d_model=preset["d_model"], num_heads=preset["heads"],
        ffn_dim=preset["ffn_dim"], num_layers=preset["layers"],
        causal=bool(preset.get("causal")),
    )
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = UserOptimizedTransformer(cfg).to(dev).eval()
    x, mask = generate_random_case(cfg, dev, torch.float32, 0, 0.0, 1.0)

    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()

        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0) or 0
        if t > 0:
            rows.append((t / iters, e.count / iters, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"\n=== {preset['name']}  (graphs off, {iters} iters) ===")
    print(f"{'us/iter':>9} {'%':>6} {'n/iter':>7}  kernel")
    for us, n, key in rows[:18]:
        print(f"{us:>9.1f} {100*us/total:>5.1f}% {n:>7.1f}  {key[:70]}")
    print(f"{total:>9.1f} 100.0% {sum(r[1] for r in rows):>7.1f}  TOTAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, action="append", default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--linear-bias", default=None)
    ap.add_argument("--linear-gelu", default=None)
    args = ap.parse_args()
    shapes = args.shape or [1]
    with open(os.path.join(ROOT, "dashboard", "presets.json")) as f:
        presets = json.load(f)["presets"]
    for s in shapes:
        audit(presets[s - 1], args.iters, args.warmup)


if __name__ == "__main__":
    main()
