"""
Benchmark the attention op alone: scalar kernel vs SDPA, with accuracy against
the exact reference alongside, so a speed win that came from losing precision is
visible rather than hidden.

    cmd.exe /c scripts\\devenv.bat python scripts\\bench_attention.py
"""

from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402
from verify_kernel import CASES, build_case, reference_attention  # noqa: E402


def timed(fn, iters=40):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def main() -> int:
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    head = (f"{'case':<18}{'sdpa ms':>10}{'scalar ms':>11}"
            f"{'scalar x':>10}{'scalar err':>12}")
    print(head)
    print("-" * len(head))

    for label, b, h, s, d, causal, padded in CASES:
        q, k, v, am, ic = build_case(b, h, s, d, causal, padded,
                                     torch.device("cuda"), torch.float32)
        scale = d ** -0.5

        with torch.inference_mode():
            ref = reference_attention(q, k, v, am, ic, scale).float()
            out = kernels.fused_attention_forward(q, k, v, am, ic, scale).float()
            err = (out - ref).abs().max().item()

            t_sdpa = timed(lambda: F.scaled_dot_product_attention(
                q, k, v, attn_mask=am, is_causal=ic, scale=scale))
            t_scalar = timed(lambda: kernels.fused_attention_forward(
                q, k, v, am, ic, scale))

        print(f"{label:<18}{t_sdpa:>10.3f}{t_scalar:>11.3f}"
              f"{t_sdpa / t_scalar:>9.2f}x{err:>12.1e}")

    print("-" * len(head))
    print("scalar x is speedup vs SDPA (>1 means the kernel is faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
