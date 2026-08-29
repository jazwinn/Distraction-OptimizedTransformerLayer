"""A/B storing O straight to global against staging it through shared memory.

O leaves the key loop in accumulator registers, and the original epilogue put
it in shared first: a fragment store into a block-sized fp32 tile, a barrier,
then sixteen rows read back and written out a lane at a time.
wmma::store_matrix_sync takes a generic pointer, and both output layouts step
one row by a constant, so the fragment can go straight to `out` instead.

Two separate wins, and the flag covers both because it picks the shared-memory
layout and the launch size as well as the code path:

  the epilogue    one fragment store per tile instead of a shared write, a
                  barrier, and sixteen shared reads plus sixteen global writes
                  per lane
  the block tile  which then only has to hold the fallback cases -- one
                  fragment per warp instead of BLOCK_M x PDIM

The second is the larger one, and it is the one that decides the table: the
head_dims whose shared memory does not shrink gain nothing, and head_dim 256,
whose block count goes from one to two, nearly doubles. So the occupancy
section prints what the driver says each layout is worth before any timing.

Sampling AND ordering are symmetric, via ab_common.balanced_order. Two timings
a side is not enough on its own: if the sides always occupy the same slots and
the later slots are faster, the bias survives. See csrc/TUNING.md.

    python scripts/ab_direct_o.py
    python scripts/ab_direct_o.py --self-control
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_common import balanced_order  # noqa: E402

import kernel_ext  # noqa: E402

import torch  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1
CONTROL_FLOOR_US = 15.0

# (label, B, H, S, head_dim, causal, mask_kind)
#
# Ordered by what the change should track. The epilogue is a fixed cost per
# block, so its share falls as the key loop lengthens: short sequences should
# gain most. The shared-memory saving is the opposite -- it is worth a fourth
# resident block at head_dim 64 and 128 and nothing below 32, so it should show
# up as a floor that does not fade with S.
CASES = [
    ("hd64  B8 H8  S128",   8,  8,  128,  64, False, "none"),
    ("hd64  B8 H8  S512",   8,  8,  512,  64, False, "none"),
    ("hd64  B8 H8  S1024",  8,  8, 1024,  64, False, "none"),
    ("hd64  B8 H8  S512  c", 8, 8,  512,  64, True,  "none"),
    ("hd64  B4 H8  S2048 c", 4, 8, 2048,  64, True,  "none"),
    ("hd32  B8 H8  S128",   8,  8,  128,  32, False, "none"),
    ("hd32  B8 H8  S512",   8,  8,  512,  32, False, "none"),
    ("hd32  B8 H8  S1024",  8,  8, 1024,  32, False, "none"),
    ("hd32  B8 H8  S512  c", 8, 8,  512,  32, True,  "none"),
    ("hd32  B4 H8  S2048 c", 4, 8, 2048,  32, True,  "none"),
    ("hd16  B8 H16 S512",   8, 16,  512,  16, False, "none"),
    ("hd16  B8 H16 S512  c", 8, 16, 512,  16, True,  "none"),
    ("hd128 B8 H8  S512",   8,  8,  512, 128, False, "none"),
    ("hd128 B4 H8  S1024 c", 4, 8, 1024, 128, True,  "none"),
    # head_dim 256 is where the saving is largest -- 16 KB, and the single
    # block per SM it was getting becomes two. It is also the only head_dim
    # that loses to SDPA, which makes it the point of the change rather than a
    # footnote, so leaving it out understates the whole table.
    ("hd256 B4 H8  S512",   4,  8,  512, 256, False, "none"),
    ("hd256 B2 H8  S1024 c", 2, 8, 1024, 256, True,  "none"),
    # ragged S: the last query block of every head falls back to the staged
    # path, so this is the shape that pays for the fallback being slower
    ("hd64  B8 H8  S500  c", 8, 8,  500,  64, True,  "none"),
    ("hd32  B8 H8  S500  c", 8, 8,  500,  32, True,  "none"),
    # head_dim 8 can never store direct: the fragment is 16 wide and the row is
    # 8. It still leaves the staged path for the tiled fallback though, so it
    # is NOT a control -- see the note under the group means.
    ("hd8   B8 H8  S512  c", 8, 8,  512,   8, True,  "none"),
    ("mask  B8 H8  S512 d32", 8, 8, 512,  32, False, "keypad"),
]


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


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def make_mask(kind, B, S):
    if kind == "none":
        return None
    m = torch.ones(B, 1, S, S, dtype=torch.bool, device=DEV)
    m[:, :, :, int(S * 0.8):] = False
    return m


def occupancy_section(K):
    print("=== what each layout is worth, from the occupancy API ===")
    print(f"  {'head_dim':>8} {'block':>8} {'staged':>8} {'direct':>8} {'gain':>6}")
    for D in (8, 16, 32, 64, 128, 256):
        res = {}
        for on in (False, True):
            K.wmma_set_direct_o(on)
            info = K.wmma_grid_info(1, 1, 4096, D)   # resident is shape-free
            res[on] = info
        if res[True][2] == 0:
            print(f"  {D:>8}   unsupported")
            continue
        sms = torch.cuda.get_device_properties(DEV).multi_processor_count
        a, b = res[False][1] // sms, res[True][1] // sms
        note = "" if a else "   (API declines: needs the carveout)"
        print(f"  {D:>8} {res[True][2]:>3}x{res[True][3]:<4} {a:>8} {b:>8} "
              f"{('+' + str(b - a)) if b > a else '-':>6}{note}")
    K.wmma_set_direct_o(True)
    print()


def op_section(K, args, self_control):
    print("=== attention op, graph-timed, interleaved ===")
    if self_control:
        print("  SELF-CONTROL: both columns are direct-off, true ratio 1.000x")
    print(f"  {'shape':<24} {'off us':>9} {'on us':>9} {'ratio':>8} {'ctrl':>6}")

    gains, ctrls, groups = [], [], {}
    for label, B, H, S, D, causal, mkind in CASES:
        g = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        mask = make_mask(mkind, B, S)
        scale = D ** -0.5

        def run():
            return K.fused_attention_forward(q, k, v, mask, causal, scale,
                                             IMPL_WMMA, BSHD)

        best_off = math.inf
        best_on = math.inf
        ctrl = math.inf
        for rnd in range(args.rounds):
            t = {False: [], True: []}
            for on in balanced_order((False, True), rnd):
                K.wmma_set_direct_o(on and not self_control)
                t[on].append(graph_timed(run, args.iters))
            best_off = min([best_off] + t[False])
            best_on = min([best_on] + t[True])
            ctrl = min(ctrl, abs(t[False][0] / t[False][1] - 1.0))
        K.wmma_set_direct_o(True)

        r = best_off / best_on
        gains.append(r)
        groups.setdefault(label.split()[0], []).append(r)
        if best_off >= CONTROL_FLOOR_US:
            ctrls.append(ctrl)
        print(f"  {label:<24} {best_off:9.1f} {best_on:9.1f} {r:7.3f}x "
              f"{ctrl*100:5.1f}%")

    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    for name in ("hd8", "hd16", "hd32", "hd64", "hd128", "hd256", "mask"):
        if name in groups:
            # head_dim 8 is not a control, which a first pass assumed. It can
            # never store direct -- the fragment is 16 wide, the row is 8 --
            # but it does not take the staged path either. It takes the tiled
            # fallback, and that alone is faster there: the old row loop ran
            # sixteen times with eight of thirty-two lanes live.
            #
            # head_dim 32 is the real control. Its O tile is already no bigger
            # than the Q staging it sits on, so neither its shared memory nor
            # its occupancy moves, and it isolates the epilogue on its own.
            note = ""
            if name == "hd8":
                note = "  (tiled fallback only, never direct)"
            elif name == "hd32":
                note = "  (control: shared memory and occupancy unchanged)"
            print(f"    {name:<6} {gm(groups[name]):.3f}x over "
                  f"{len(groups[name])} shapes{note}")
    if ctrls:
        print(f"  worst control over rows above {CONTROL_FLOOR_US:g} us: "
              f"+/-{max(ctrls)*100:.1f}%")


MODEL_SHAPES = [
    ("B8  S512  d256 h8",   8,  512,  256,  8),
    ("B8  S1024 d256 h8",   8, 1024,  256,  8),
    ("B8  S128  d256 h8",   8,  128,  256,  8),
    ("B16 S128  d256 h8",  16,  128,  256,  8),
    ("B8  S512  d512 h8",   8,  512,  512,  8),
    ("B8  S128  d32  h4",   8,  128,   32,  4),
]


def model_section(K, args):
    import torch_transformer_benchmark as bench

    print()
    print("=== whole model, 6 layers, causal, ffn_dim == d_model ===")
    print(f"  {'shape':<20} {'off ms':>9} {'on ms':>9} {'ratio':>8} "
          f"{'ctrl':>6} {'max_abs':>10}")

    gains = []
    for label, b, sq, d, h in MODEL_SHAPES:
        cfg = bench.TransformerConfig(batch_size=b, seq_len=sq, d_model=d,
                                      num_heads=h, ffn_dim=d, num_layers=6,
                                      causal=True)
        base = bench.BaselineTransformer(cfg)
        models = {}
        for on in (False, True):
            opt = bench.UserOptimizedTransformer(cfg)
            bench.copy_model_weights(base, opt)
            models[on] = opt.to(DEV).eval()
        x, m = bench.generate_random_case(config=cfg, device=DEV,
                                          dtype=torch.float32, seed=1234,
                                          padding_ratio=0.0, input_scale=1.0)

        def run(on, iters):
            K.wmma_set_direct_o(on)
            with torch.inference_mode():
                torch.cuda.synchronize()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(iters):
                    models[on](x, m)
                en.record()
                torch.cuda.synchronize()
            return st.elapsed_time(en) / iters

        for on in (False, True):
            run(on, 5)

        best = {False: math.inf, True: math.inf}
        ctrl = math.inf
        for rnd in range(args.rounds):
            t = {False: [], True: []}
            for on in balanced_order((False, True), rnd):
                t[on].append(run(on, args.iters))
            best[False] = min([best[False]] + t[False])
            best[True] = min([best[True]] + t[True])
            ctrl = min(ctrl, abs(t[False][0] / t[False][1] - 1.0))

        outs = {}
        for on in (False, True):
            K.wmma_set_direct_o(on)
            with torch.inference_mode():
                outs[on] = models[on](x, m).clone()
        err = (outs[True] - outs[False]).abs().max().item()

        r = best[False] / best[True]
        gains.append(r)
        print(f"  {label:<20} {best[False]:9.3f} {best[True]:9.3f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {err:10.2e}")

    K.wmma_set_direct_o(True)
    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    print("  max_abs is 0.0 everywhere by construction: the same fragments "
          "reach the same addresses, only not via shared memory.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--self-control", action="store_true")
    ap.add_argument("--op-only", action="store_true")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_direct_o"):
        print(f"need a build with the direct-O epilogue: {kernel_ext.load_error()}")
        return 1

    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, fp32 tensors in "
          f"fp16 fragments, graph-timed")
    print()

    warm = torch.randn(8, 8, 128, 64, device=DEV)
    graph_timed(lambda: K.fused_attention_forward(
        warm, warm, warm, None, True, 0.125, IMPL_WMMA, BSHD), 10, 2, 5)

    if not args.self_control:
        occupancy_section(K)
    op_section(K, args, args.self_control)
    if not args.op_only and not args.self_control:
        model_section(K, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
