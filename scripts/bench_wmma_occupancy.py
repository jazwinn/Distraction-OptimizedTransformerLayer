"""How much of the card the wmma attention grid actually fills, and what that
costs -- the measurement the split-KV decision rests on.

The grid is (ceil(S/BLOCK_M), H, B): query side only. Nothing in it scales with
the KEY length, so a shape with few queries and many keys launches a small grid
that each do a lot of serial work. Flash-Decoding fixes exactly that case and
only that case, so the first question is which shapes are in it.

Two columns carry the argument:

  waves    blocks / resident. Below 1.0 the card is not full and the kernel is
           latency-bound; at or above 1.0 there is nothing for a split to buy.
  us/block op time divided by blocks, normalised against the same (H, S, D) at
           a batch large enough to fill the card. A ratio well above 1.0 means
           each block is waiting rather than working.

`split cap` is min(resident/blocks, n_kt): a split cannot use more capacity
than the card has spare, and cannot cut a key range into more pieces than it
has key tiles.

    python scripts/bench_wmma_occupancy.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1

# (label, B, H, S, head_dim, causal). Chosen to span the occupancy range: the
# first rows are the grading shapes whose grids cannot fill the card, the last
# are controls that already do.
CASES = [
    ("B1  H8  S128  d32 c",   1,  8,  128,  32, True),
    ("B1  H8  S512  d32 c",   1,  8,  512,  32, True),
    ("B1  H8  S1024 d32 c",   1,  8, 1024,  32, True),
    ("B2  H8  S128  d32 c",   2,  8,  128,  32, True),
    ("B4  H8  S128  d32 c",   4,  8,  128,  32, True),
    ("B1  H4  S128  d32 c",   1,  4,  128,  32, True),
    ("B1  H8  S128  d64 c",   1,  8,  128,  64, True),
    ("B1  H8  S2048 d64 c",   1,  8, 2048,  64, True),
    ("B2  H2  S2048 d64 c",   2,  2, 2048,  64, True),
    ("B8  H4  S128  d8  c",   8,  4,  128,   8, True),
    ("B8  H16 S128  d16 c",   8, 16,  128,  16, True),
    ("B8  H8  S128  d32 c",   8,  8,  128,  32, True),
    ("B16 H8  S128  d32 c",  16,  8,  128,  32, True),
    ("B64 H8  S128  d32 c",  64,  8,  128,  32, True),
]

# (H, S, D, causal) -> a batch big enough to saturate, for the us/block baseline
SATURATED_B = 64


def graph_timed(fn, iters=30, reps=5, per_graph=10):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
            for _ in range(per_graph):
                fn()
    except Exception:
        return None
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) / iters * 1e3 / per_graph)
    return best


def time_shape(K, B, H, S, D, causal, iters):
    g = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(B, H, S, D, device=DEV, generator=g)
    k = torch.randn(B, H, S, D, device=DEV, generator=g)
    v = torch.randn(B, H, S, D, device=DEV, generator=g)
    scale = D ** -0.5
    return graph_timed(lambda: K.fused_attention_forward(
        q, k, v, None, causal, scale, IMPL_WMMA, BSHD), iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_grid_info"):
        print("need a build with wmma_grid_info; rebuild")
        return 1

    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, fp32 tensors in "
          f"fp16 fragments, graph-timed")
    print()
    print(f"  {'shape':<20} {'BM':>3} {'blocks':>7} {'resid':>6} {'waves':>6} "
          f"{'n_kt':>5} {'cap':>4} {'op us':>8} {'us/blk':>7} {'vs full':>8}")

    # warm
    time_shape(K, 8, 8, 128, 64, True, 10)

    saturated = {}
    rows = []
    for label, B, H, S, D, causal in CASES:
        blocks, resident, bm, bn = K.wmma_grid_info(B, H, S, D)
        if blocks == 0:
            print(f"  {label:<20}  (no fp16 kernel at head_dim {D})")
            continue
        us = time_shape(K, B, H, S, D, causal, args.iters)

        # per-block cost at a batch that certainly fills the card, same (H,S,D)
        key = (H, S, D, causal)
        if key not in saturated:
            sb = K.wmma_grid_info(SATURATED_B, H, S, D)[0]
            sus = time_shape(K, SATURATED_B, H, S, D, causal, args.iters)
            saturated[key] = (sus / sb) if sb else float("nan")
        full_per_block = saturated[key]

        # key tiles a block walks. Under causal the average m-tile walks about
        # half the range, so this is the dense count halved -- it bounds how
        # many pieces a split has to work with.
        n_kt_dense = (S + bn - 1) // bn
        n_kt = max(1, n_kt_dense // 2 if causal else n_kt_dense)

        waves = blocks / resident if resident else float("nan")
        cap = min(resident // blocks if blocks else 0, n_kt)
        per_block = us / blocks
        rows.append((label, bm, blocks, resident, waves, n_kt, cap, us,
                     per_block, per_block / full_per_block))
        print(f"  {label:<20} {bm:>3} {blocks:>7} {resident:>6} {waves:>6.2f} "
              f"{n_kt:>5} {cap:>4} {us:>8.1f} {per_block:>7.3f} "
              f"{per_block/full_per_block:>7.2f}x")

    print()
    print("  waves < 1.0 and vs-full >> 1.0 together mean idle capacity a split")
    print("  could fill. `cap` is the most splits that shape can use.")
    hot = [r for r in rows if r[4] < 1.0 and r[6] >= 2]
    if hot:
        print()
        print(f"  {len(hot)} of {len(rows)} shapes are candidates:")
        for r in hot:
            print(f"    {r[0]:<20} waves {r[4]:.2f}, cap {r[6]}, "
                  f"per-block {r[9]:.2f}x the saturated cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
