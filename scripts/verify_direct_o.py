"""Storing O straight to global must be BIT-identical to staging it in shared.

The direct epilogue hands the same accumulator fragments to the same
wmma::store_matrix_sync; only the destination changes, from a shared tile that
is then copied out to the output row itself. No arithmetic is added, removed or
reordered, so the bar is exact equality and any difference is an addressing
bug.

The cases exist to make each clause of the eligibility test fail in turn:

  rows      S not a multiple of BLOCK_M, so the last query block is ragged and
            falls back -- a fragment store would write past the sequence
  DIM/PDIM  head_dim 8, where the fragment is 16 wide and only 8 columns
            exist, so a direct store would spill into the next row
  layout    [B,H,S,D] steps a row by D, [B,S,H*D] by H*D; both are the ldm the
            fragment store takes, and getting them the wrong way round is the
            most likely mistake here
  split     the split-KV partials go to part_o, which is fp32 and packed
            [B,H,splits,S,D] -- a different destination with its own stride

    python scripts/verify_direct_o.py
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

# (B, H, S, head_dim, causal, mask_kind, layout)
CASES = []
for D in (8, 16, 32, 64, 128):
    for S in (16, 64, 100, 128, 130, 192, 257, 512):
        for causal in (False, True):
            for layout in (0, 1):
                CASES.append((2, 4, S, D, causal, "none", layout))
for D in (16, 32, 64):
    for kind in ("keypad", "rowkill"):
        CASES.append((2, 4, 192, D, False, kind, 1))
        CASES.append((2, 4, 192, D, True, kind, 1))
# H > 1 with the packed layout is what makes ldm = H*head_dim rather than
# head_dim, so an ldm mix-up shows up here and nowhere else.
for H in (1, 3, 8):
    for D in (16, 64):
        CASES.append((2, H, 128, D, True, "none", 1))
        CASES.append((2, H, 128, D, True, "none", 0))


def make_mask(kind, B, S):
    if kind == "none":
        return None
    m = torch.ones(B, 1, S, S, dtype=torch.bool, device=DEV)
    m[:, :, :, int(S * 0.7):] = False
    if kind == "rowkill":
        m[:, :, S // 3:S // 3 + 5, :] = False
    return m


def sweep(K, label, force_splits):
    bad, checked = [], 0
    for B, H, S, D, causal, mkind, layout in CASES:
        g = torch.Generator(device="cuda").manual_seed(20260829)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        mask = make_mask(mkind, B, S)
        scale = D ** -0.5

        outs = {}
        for on in (False, True):
            K.wmma_set_direct_o(on)
            outs[on] = K.fused_attention_forward(q, k, v, mask, causal, scale,
                                                 IMPL_WMMA, layout).clone()
        checked += 1
        nan = int(torch.isnan(outs[True]).sum().item())
        if not torch.equal(outs[False], outs[True]) or nan:
            d = (outs[True].double() - outs[False].double()).abs().max().item()
            name = (f"B{B} H{H} S{S} d{D} "
                    f"{'causal' if causal else 'dense'} {mkind} layout{layout}")
            bad.append(name)
            print(f"  MISMATCH  {label}  {name}  max_diff={d:.3e}  nan={nan}")
    return bad, checked


def main() -> int:
    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_direct_o"):
        print(f"need a build with the direct-O epilogue: {kernel_ext.load_error()}")
        return 1

    print(f"{torch.cuda.get_device_properties(DEV).name}: direct O on vs off, "
          f"{len(CASES)} cases x 2 split settings, bar is EXACT equality")
    print()

    total_bad, total = [], 0
    # splits == 1 is the ordinary path into `out`; a forced split count sends
    # the same fragments to part_o instead and then through the combine pass.
    for splits, label in ((0, "splits=auto"), (4, "splits=4")):
        K.wmma_set_split_count(splits)
        bad, checked = sweep(K, label, splits)
        print(f"  {label:<12} {checked - len(bad)}/{checked} bit-identical")
        total_bad += bad
        total += checked
    K.wmma_set_split_count(0)
    K.wmma_set_direct_o(True)

    print()
    if total_bad:
        print(f"  {len(total_bad)} of {total} cases DIFFER -- the direct "
              f"epilogue is wrong")
        return 1
    print(f"  all {total} cases bit-identical with the direct epilogue on and off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
