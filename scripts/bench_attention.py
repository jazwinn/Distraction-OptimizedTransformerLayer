"""
Benchmark the attention op alone: both custom kernels against SDPA, with
accuracy alongside, so a speed win that came from losing precision is visible
rather than hidden.

The reference here runs with TF32 enabled, because that is the arithmetic the
benchmark harness baseline actually uses. Measuring against an exact-fp32
reference instead makes the scalar kernel look better than it is: the harness
compares against the baseline output, not against ground truth.

    cmd.exe /c scripts\\devenv.bat python scripts\\bench_attention.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402
from verify_kernel import CASES, build_case, reference_attention  # noqa: E402


ROUNDS = 10
INNER = 5


def bench(fns):
    """Interleaved min-of-N timing, in ms.

    Candidates are timed round-robin rather than one after another, and the
    minimum is reported rather than the median: on a power-capped part the
    clock sags over a long run, which would otherwise penalise whichever
    candidate happened to be measured last.
    """
    for fn in fns.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    best = {name: float("inf") for name in fns}
    for _ in range(ROUNDS):
        for name, fn in fns.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(INNER):
                fn()
            end.record()
            torch.cuda.synchronize()
            best[name] = min(best[name], start.elapsed_time(end) / INNER)
    return best


def main() -> int:
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    # TF32 on: this is what the harness baseline runs, so it is what the custom
    # kernels have to match.
    torch.backends.cuda.matmul.allow_tf32 = True

    head = (f"{'case':<18}{'sdpa':>8}{'scalar':>8}{'wmma':>8}"
            f"{'wmma/sc':>9}{'wmma/sdpa':>11}{'sc err':>9}{'wmma err':>10}")
    print(head)
    print("-" * len(head))

    for label, b, h, s, d, causal, padded in CASES:
        q, k, v, am, ic = build_case(b, h, s, d, causal, padded,
                                     torch.device("cuda"), torch.float32)
        scale = d ** -0.5

        with torch.inference_mode():
            ref = reference_attention(q, k, v, am, ic, scale).float()
            err_scalar = (kernels.fused_attention_forward(
                q, k, v, am, ic, scale, 1).float() - ref).abs().max().item()
            err_wmma = (kernels.fused_attention_forward(
                q, k, v, am, ic, scale, 0).float() - ref).abs().max().item()

            t = bench({
                "sdpa": lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=am, is_causal=ic, scale=scale),
                "scalar": lambda: kernels.fused_attention_forward(
                    q, k, v, am, ic, scale, 1),
                "wmma": lambda: kernels.fused_attention_forward(
                    q, k, v, am, ic, scale, 0),
            })

        print(f"{label:<18}{t['sdpa']:>8.3f}{t['scalar']:>8.3f}{t['wmma']:>8.3f}"
              f"{t['scalar'] / t['wmma']:>8.2f}x{t['sdpa'] / t['wmma']:>10.2f}x"
              f"{err_scalar:>9.1e}{err_wmma:>10.1e}")

    print("-" * len(head))
    print("ratios >1 mean the tensor-core kernel is faster; the wmma column "
          "falls back to")
    print("the scalar kernel on shapes it does not cover (head_dim 8 and 128).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
