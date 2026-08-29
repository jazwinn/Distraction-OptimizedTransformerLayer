"""Correctness of the wmma split-KV path against the single-pass one.

Split-KV changes the summation order of the online softmax: each split reduces
its own slice of the key range and the combine pass rebases every partial onto
the max over all slices. The result should differ from the single-pass kernel
only by float reassociation, so the bar here is "agrees with the single-pass
kernel to well inside the harness tolerance, and matches float64 no worse".

`--split-count` is forced rather than left to the rule, because the interesting
cases are the ones the rule declines: more splits than a block has key tiles
exercises the empty-split path, which stores (-inf, 0, 0) and must be weighted
to exactly zero rather than producing a NaN.

    python scripts/verify_wmma_split_kv.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402

from verify_kernel import reference_attention_f64  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2

# (label, B, H, S, head_dim, causal, mask_kind, out_layout)
#   mask_kind: none | keypad | rowkill
# S values deliberately include non-multiples of BLOCK_M (64) and of BLOCK_N.
CASES = [
    ("dense  d16  S128",   2, 4,  128,  16, False, "none",    1),
    ("causal d16  S128",   2, 4,  128,  16, True,  "none",    1),
    ("causal d32  S128",   1, 8,  128,  32, True,  "none",    1),
    ("causal d32  S100",   1, 8,  100,  32, True,  "none",    1),
    ("causal d32  S512",   1, 8,  512,  32, True,  "none",    1),
    ("dense  d32  S512",   1, 8,  512,  32, False, "none",    1),
    ("causal d64  S128",   1, 8,  128,  64, True,  "none",    1),
    ("causal d64  S192",   1, 8,  192,  64, True,  "none",    1),
    ("dense  d64  S256",   2, 4,  256,  64, False, "none",    1),
    ("causal d128 S256",   1, 4,  256, 128, True,  "none",    1),
    ("causal d32  S128 L0", 1, 8, 128,  32, True,  "none",    0),
    ("dense  d32  S128 L0", 1, 8, 128,  32, False, "none",    0),
    ("keypad d32  S128",   2, 4,  128,  32, False, "keypad",  1),
    ("keypad d32  S192 c", 2, 4,  192,  32, True,  "keypad",  1),
    ("rowkill d32 S128",   2, 4,  128,  32, False, "rowkill", 1),
    ("rowkill d64 S128 c", 2, 4,  128,  64, True,  "rowkill", 1),
]

SPLIT_COUNTS = [2, 3, 4, 8]


def make_mask(kind, B, H, S):
    if kind == "none":
        return None
    m = torch.ones(B, 1, S, S, dtype=torch.bool, device=DEV)
    if kind in ("keypad", "rowkill"):
        m[:, :, :, int(S * 0.7):] = False        # last 30% of keys invalid
    if kind == "rowkill":
        m[:, :, S // 3:S // 3 + 5, :] = False    # 5 query rows fully masked
    return m


def reference(q, k, v, mask, causal, scale, layout):
    """float64 SDPA. Rows with no admissible key are zeros, as the kernels emit.

    SDPA refuses attn_mask together with is_causal, so when the case has both
    the triangle is folded into the mask here. The kernel takes them separately
    and is expected to agree, which is part of what this checks.
    """
    S = q.shape[-2]
    bool_mask = mask
    if causal and mask is not None:
        tri = torch.ones(S, S, dtype=torch.bool, device=DEV).tril()
        bool_mask = mask & tri
    elif causal:
        bool_mask = None

    # reference_attention_f64 takes the bool mask directly -- True means
    # "may attend" -- so the float -inf mask this used to build for SDPA is
    # not needed. It also handles the fully-masked row, which is 0/0 in the
    # softmax and which the kernels emit as 0.
    return reference_attention_f64(
        q, k, v, bool_mask, causal and mask is None, scale, layout=layout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atol", type=float, default=2e-3)
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_split_count"):
        print(f"need a build with split-KV: {kernel_ext.load_error()}")
        return 1

    print(f"{torch.cuda.get_device_properties(DEV).name}: wmma split-KV vs the "
          f"single-pass kernel, fp32 tensors in fp16 fragments")
    print()
    print(f"  {'case':<22} {'splits':>6} {'vs 1-pass':>10} {'vs fp64':>10} "
          f"{'1-pass fp64':>12} {'nan':>4}  verdict")

    worst = 0.0
    failures = 0
    for label, B, H, S, D, causal, mkind, layout in CASES:
        g = torch.Generator(device="cuda").manual_seed(20260829)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        scale = D ** -0.5
        mask = make_mask(mkind, B, H, S)
        ref = reference(q, k, v, mask, causal, scale, layout)

        def run():
            return K.fused_attention_forward(q, k, v, mask, causal, scale,
                                             IMPL_WMMA, layout)

        K.wmma_set_split_count(1)
        base = run().double()
        base_err = (base - ref).abs().max().item()

        for n in SPLIT_COUNTS:
            K.wmma_set_split_count(n)
            got = run().double()
            d_pass = (got - base).abs().max().item()
            d_ref = (got - ref).abs().max().item()
            nans = int(torch.isnan(got).sum().item())
            ok = (d_ref <= args.atol) and nans == 0
            worst = max(worst, d_pass)
            if not ok:
                failures += 1
            print(f"  {label:<22} {n:>6} {d_pass:10.2e} {d_ref:10.2e} "
                  f"{base_err:12.2e} {nans:>4}  {'ok' if ok else 'FAIL'}")

    K.wmma_set_split_count(0)
    print()
    print(f"  worst split-vs-single-pass disagreement: {worst:.2e}")
    if failures:
        print(f"  {failures} FAILURES")
        return 1
    print("  every split count matches the single-pass kernel and float64 "
          f"within atol {args.atol:g}, with no NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
