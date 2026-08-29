"""Per-tile mask classification must be BIT-identical to testing every element.

Unlike a precision change, this one removes tests whose outcome was already
"pass": an interior key tile is one where every bounds, causal and mask
predicate would have been true. So the arithmetic is not merely equivalent, it
is the same arithmetic in the same order, and the bar is exact equality. Any
difference at all is a classification bug -- a tile called interior that was
not.

The cases exist to make each clause fail in turn:

  rows_in   S not a multiple of BLOCK_M, so the last query block is ragged
  cols_in   S not a multiple of BLOCK_N, so the last key tile is ragged
  tri_free  causal, so every block has exactly one diagonal tile
  mask      an explicit mask, which disables the fast path outright
  DIM/PDIM  head_dim 8, where the operands are padded 8 -> 16

    python scripts/verify_mask_classify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2

# (label, B, H, S, head_dim, causal, mask_kind, layout)
CASES = []
for D in (8, 16, 32, 64, 128):
    for S in (64, 100, 128, 130, 192, 257, 512):
        for causal in (False, True):
            CASES.append((f"d{D:<3} S{S:<4} {'causal' if causal else 'dense '}",
                          2, 4, S, D, causal, "none", 1))
# mask variants, and the other output layout
for D in (16, 32, 64):
    for kind in ("keypad", "rowkill"):
        CASES.append((f"d{D:<3} S192  {kind}", 2, 4, 192, D, False, kind, 1))
        CASES.append((f"d{D:<3} S192  {kind} c", 2, 4, 192, D, True, kind, 1))
    CASES.append((f"d{D:<3} S130  layout0", 2, 4, 130, D, True, "none", 0))


def make_mask(kind, B, S):
    if kind == "none":
        return None
    m = torch.ones(B, 1, S, S, dtype=torch.bool, device=DEV)
    m[:, :, :, int(S * 0.7):] = False
    if kind == "rowkill":
        m[:, :, S // 3:S // 3 + 5, :] = False
    return m


def main() -> int:
    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_mask_classify"):
        print(f"need a build with mask classification: {kernel_ext.load_error()}")
        return 1

    print(f"{torch.cuda.get_device_properties(DEV).name}: classification on vs "
          f"off, {len(CASES)} cases, bar is EXACT equality")
    print()

    bad, checked = [], 0
    for label, B, H, S, D, causal, mkind, layout in CASES:
        g = torch.Generator(device="cuda").manual_seed(20260829)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        mask = make_mask(mkind, B, S)
        scale = D ** -0.5

        outs = {}
        for on in (False, True):
            K.wmma_set_mask_classify(on)
            outs[on] = K.fused_attention_forward(q, k, v, mask, causal, scale,
                                                 IMPL_WMMA, layout).clone()
        checked += 1
        same = torch.equal(outs[False], outs[True])
        nan = int(torch.isnan(outs[True]).sum().item())
        if not same or nan:
            d = (outs[True].double() - outs[False].double()).abs().max().item()
            bad.append((label, d, nan))
            print(f"  MISMATCH  {label}  max_diff={d:.3e}  nan={nan}")

    K.wmma_set_mask_classify(True)
    print()
    if bad:
        print(f"  {len(bad)} of {checked} cases DIFFER -- classification is wrong")
        return 1
    print(f"  all {checked} cases bit-identical with classification on and off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
