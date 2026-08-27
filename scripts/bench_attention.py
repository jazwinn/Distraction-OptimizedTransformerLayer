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

    # Tile modes are optional columns: each raises rather than falling back on
    # a shape it does not specialize (head_dim 128 in this table), and the whole
    # tile kernel is absent from builds that found no CUDA 13.3+. Both show as
    # n/a. Kept as a list so a new math mode is one entry, not another set of
    # hand-written column branches.
    TILE_COLS = (("tile", 3), ("tile-tf32", 5), ("tile-bf16", 4))

    head = (f"{'case':<18}{'sdpa':>10}{'scalar':>10}{'wmma':>10}"
            + "".join(f"{name:>10}" for name, _ in TILE_COLS)
            + f"{'wmma/sdpa':>11}"
            + "".join(f"{name + '/sdpa':>16}" for name, _ in TILE_COLS)
            + f"{'wmma err':>10}"
            + "".join(f"{name + ' err':>14}" for name, _ in TILE_COLS))
    print(head)
    print("-" * len(head))

    for label, b, h, s, d, causal, padded in CASES:
        q, k, v, am, ic = build_case(b, h, s, d, causal, padded,
                                     torch.device("cuda"), torch.float32)
        scale = d ** -0.5

        def run(impl):
            return kernels.fused_attention_forward(q, k, v, am, ic, scale, impl)

        with torch.inference_mode():
            ref = reference_attention(q, k, v, am, ic, scale).float()
            err_scalar = (run(1).float() - ref).abs().max().item()
            err_wmma = (run(0).float() - ref).abs().max().item()

            timed = {
                "sdpa": lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=am, is_causal=ic, scale=scale),
                "scalar": lambda: run(1),
                "wmma": lambda: run(0),
            }
            errs = {}
            for name, impl in TILE_COLS:
                try:
                    errs[name] = (run(impl).float() - ref).abs().max().item()
                except RuntimeError:
                    errs[name] = None
                    continue
                # Bind impl per iteration; a bare closure over the loop
                # variable would time the last mode under every name.
                timed[name] = (lambda i: lambda: run(i))(impl)
            t = bench(timed)

        def cell(name, width, fmt):
            return (f"{fmt(name):>{width}}" if errs[name] is not None
                    else f"{'n/a':>{width}}")

        print(f"{label:<18}{t['sdpa']:>10.3f}{t['scalar']:>10.3f}{t['wmma']:>10.3f}"
              + "".join(cell(n, 10, lambda n=n: f"{t[n]:.3f}") for n, _ in TILE_COLS)
              + f"{t['sdpa'] / t['wmma']:>10.2f}x"
              + "".join(cell(n, 16, lambda n=n: f"{t['sdpa'] / t[n]:.2f}x")
                        for n, _ in TILE_COLS)
              + f"{err_wmma:>10.1e}"
              + "".join(cell(n, 14, lambda n=n: f"{errs[n]:.1e}")
                        for n, _ in TILE_COLS))

    print("-" * len(head))
    print("ratios >1 mean the custom kernel is faster than sdpa. wmma now "
          "covers every")
    print("head_dim in the table; the tile columns report n/a at head_dim 128, "
          "which they")
    print("do not specialize. tile runs on the CUDA cores; tile-tf32 and "
          "tile-bf16 are the")
    print("same kernel with its GEMM operands narrowed onto the tensor cores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
