"""
Check the custom fused attention kernels against SDPA and against the baseline's
own attention math, on the attention op alone -- no transformer stack, no
weight loading, no benchmark harness in the way.

Both kernels are checked on every case, each against its own tolerance. They are
not trying to hit the same number: the scalar kernel does everything in fp32 and
should land within a few ULP of an exact reference, while the tensor-core kernel
rounds its operands to TF32 on purpose, which is worth about 1e-3 on these
magnitudes. Holding both to one loose threshold would mean a scalar-kernel
regression of three orders of magnitude still passed.

The reference is computed with TF32 explicitly off, so it does not move with the
ambient torch settings.

Use this while writing the kernels. It fails fast and points at the exact
(config, impl) that broke, which the full harness can't do.

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_kernel.py
"""

from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernel_ext  # noqa: E402

# (label, batch, heads, seq_len, head_dim, causal, padded)
CASES = [
    ("tiny",              1,  1,   8,  8, False, False),
    ("tiny causal",       1,  1,   8,  8, True,  False),
    ("tiny padded",       2,  1,   8,  8, False, True),
    ("tiny causal+pad",   2,  1,   8,  8, True,  True),
    ("default",           8,  8, 128, 64, False, False),
    ("default causal",    8,  8, 128, 64, True,  False),
    ("default padded",    8,  8, 128, 64, False, True),
    ("default caus+pad",  8,  8, 128, 64, True,  True),
    ("long seq",          1,  8, 2048, 64, False, False),
    ("long seq causal",   1,  8, 2048, 64, True,  False),
    ("odd shape",         3,  5,  37, 32, True,  True),
    ("wide head_dim",     2,  4,  64, 128, False, False),
]


def reference_attention(q, k, v, attn_mask, is_causal, scale):
    """Mirrors BaselineSelfAttention.forward's arithmetic exactly."""
    S = q.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        blocked = torch.ones(S, S, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(blocked, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def build_case(b, h, s, d, causal, padded, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, h, s, d, generator=g, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, generator=g, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, generator=g, device=device, dtype=dtype)

    attn_mask = None
    is_causal = causal
    if padded:
        lengths = torch.randint(1, s + 1, (b,), generator=g, device=device)
        valid = torch.arange(s, device=device)[None, :] < lengths[:, None]  # [B,S]
        key_mask = valid[:, None, None, :]
        if causal:
            causal_allowed = torch.ones(s, s, device=device, dtype=torch.bool).tril()
            attn_mask = key_mask & causal_allowed
            is_causal = False  # folded into attn_mask
        else:
            attn_mask = key_mask.expand(b, 1, s, s).contiguous()
    return q, k, v, attn_mask, is_causal


def timed(fn, iters=30):
    for _ in range(8):
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


# (impl code, label, tolerance against an exact-fp32 reference).
#
# The scalar kernel accumulates entirely in fp32, so anything past a few ULP of
# the reference is a bug rather than rounding. The tensor-core kernel rounds Q,
# K, P and V to TF32 (10 mantissa bits) before every multiply, exactly as cuBLAS
# does for the baseline; on unit-scale inputs that is worth ~1e-3.
# The tile kernel does its matmuls in plain fp32, so it holds the same budget
# as the scalar kernel rather than the tensor-core one.
# tile-tf32 rounds the same operands the wmma kernel does to the same 10
# mantissa bits, so it gets the same budget.
# tile-bf16 narrows both GEMM operands to bfloat16 (8 significand bits), so
# its budget is an order of magnitude looser than the tf32 paths', not tighter.
# Impls that raise on a case they do not cover rather than falling back, so a
# raise from one of these is coverage rather than a failure. Everything in
# IMPLS except the scalar kernel, which has always fallen back silently.
DECLINING_IMPLS = frozenset({2, 3, 4, 5})

IMPLS = (
    (1, "scalar", 5e-6),
    (2, "wmma", 3e-3),
    (3, "tile", 5e-6),
    (5, "tile-tf32", 3e-3),
    (4, "tile-bf16", 3e-2),
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1

    kernels = kernel_ext.get_kernels(verbose=False)
    if kernels is None:
        print(f"custom kernel failed to build: {kernel_ext.load_error()}")
        print("run through scripts/devenv.bat so cl.exe is on PATH")
        return 1

    # Pin the reference: left at the ambient setting, the thing every kernel is
    # measured against would move between runs.
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda")
    dtype = torch.float32
    row = "{:<18} {:>8} {:>11} {:>11} {:>9} {:>9} {:>8}"
    print(row.format("case", "impl", "vs_ref", "vs_sdpa",
                     "custom_ms", "sdpa_ms", "speedup"))
    print("-" * 80)

    failures = []
    for label, b, h, s, d, causal, padded in CASES:
        q, k, v, attn_mask, is_causal = build_case(
            b, h, s, d, causal, padded, device, dtype
        )
        scale = d ** -0.5

        with torch.inference_mode():
            ref = reference_attention(q, k, v, attn_mask, is_causal, scale)
            sdpa = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
            )
            t_sdpa = timed(lambda: F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale))

            for impl, name, tol in IMPLS:
                try:
                    custom = kernels.fused_attention_forward(
                        q, k, v, attn_mask, is_causal, scale, impl
                    )
                except RuntimeError as exc:
                    if impl in DECLINING_IMPLS:
                        # No tensor-core / tile specialization for this
                        # head_dim (or no tile support in this build). The
                        # scalar row above already covered the case.
                        print(row.format(label, name, "n/a", "-", "-", "-", "-"))
                        continue
                    print(row.format(label, name, "RAISED", str(exc)[:11],
                                     "-", "-", "-"))
                    failures.append(f"{label}/{name}")
                    continue

                if custom.shape != ref.shape:
                    print(row.format(label, name, f"SHAPE{tuple(custom.shape)}",
                                     "-", "-", "-", "-"))
                    failures.append(f"{label}/{name}")
                    continue

                d_ref = (custom.float() - ref.float()).abs().max().item()
                d_sdpa = (custom.float() - sdpa.float()).abs().max().item()
                t_custom = timed(lambda: kernels.fused_attention_forward(
                    q, k, v, attn_mask, is_causal, scale, impl))

                bad = d_ref > tol or not torch.isfinite(custom).all()
                if bad:
                    failures.append(f"{label}/{name}")
                print(row.format(
                    label,
                    name,
                    f"{d_ref:.2e}" + ("!" if bad else ""),
                    f"{d_sdpa:.2e}",
                    f"{t_custom:.3f}",
                    f"{t_sdpa:.3f}",
                    f"{t_sdpa / t_custom:.2f}x",
                ))

    print("-" * 80)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("every kernel matches the reference on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
