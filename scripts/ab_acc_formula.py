"""A/B the closed-form accumulator map against probing for it once per block.

Keeping O in accumulator registers means applying the per-row softmax rescale
to fragment elements, which needs to know which row each element holds. That is
architecture-defined and undocumented, so the kernel discovered it by probing:
store a tagged fragment, read back where each tag landed, invert. Correct, and
paid once per block -- a store_matrix_sync, two __syncwarp barriers and sixteen
shared accesses, to learn something that does not vary between blocks.

The closed form costs eight shifts. The probe still runs, once per process, to
confirm the two agree; if they ever do not the kernel goes back to probing.

**This is a fixed per-block cost, so it only shows up where blocks are short.**
A block's time is roughly `a + b*n_kt`; measured on this card at head_dim 32,
a is 47.3 ns and b is 15.4 ns, so at seq 128 the fixed part is ~42% of a block
and by seq 1024 it is ~9%. The probe is one slice of that `a` -- most of it is
staging Q -- so the ceiling here is single-digit percent at short sequences and
near zero at long ones. The table is ordered by sequence length to show that,
and the long rows are the control: they should read ~1.000x.

Eleven of the fourteen appendix shapes are seq 128, which is why a fixed
per-block cost is worth attacking at all.

Sampling AND ordering are symmetric: two timings a side per round, and which
side goes first alternates between rounds. Two timings a side alone is not
enough -- it fixes comparing min-of-2 against min-of-1, but if the sides always
occupy the same slots and the later slots are faster, the bias survives. A
self-control here read 1.115x at the smallest shape before the ordering was
alternated. See csrc/TUNING.md.

    python scripts/ab_acc_formula.py
    python scripts/ab_acc_formula.py --self-control
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

from ab_common import balanced_order  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1
CONTROL_FLOOR_US = 15.0

# (label, B, H, S, head_dim, causal)
#
# Grouped by sequence length, because that is what decides the share. The
# short-sequence rows are where a per-block cost can matter; the long ones are
# the built-in control.
CASES = [
    # seq 32-128: the appendix's own territory, 11 of its 14 shapes
    ("S32   hd32  B64 H4", 64,  4,   32,  32, True),
    ("S128  hd32  B64 H4", 64,  4,  128,  32, True),
    ("S128  hd32  B16 H4", 16,  4,  128,  32, True),
    ("S128  hd32  B4  H4",  4,  4,  128,  32, True),
    ("S128  hd8   B64 H4", 64,  4,  128,   8, True),
    ("S128  hd8   B64 H16", 64, 16, 128,   8, True),
    ("S128  hd64  B64 H2", 64,  2,  128,  64, True),
    ("S128  hd128 B64 H1", 64,  1,  128, 128, True),
    ("S128  hd256 B16 H4", 16,  4,  128, 256, True),
    ("S128  hd32  B8  H8 d", 8,  8,  128,  32, False),
    # seq 512 and up: the fixed cost is amortised away, so these are controls
    ("S512  hd32  B8  H8",  8,  8,  512,  32, True),
    ("S1024 hd32  B8  H8",  8,  8, 1024,  32, True),
    ("S1024 hd64  B4  H8",  4,  8, 1024,  64, True),
    ("S2048 hd32  B4  H8",  4,  8, 2048,  32, True),
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


def op_section(K, args, self_control):
    print("=== attention op, graph-timed, interleaved ===")
    if self_control:
        print("  SELF-CONTROL: both columns are probe-per-block, true ratio "
              "1.000x")
    print(f"  {'shape':<22} {'probe us':>9} {'formula':>9} {'ratio':>8} "
          f"{'ctrl':>6}")

    gains, ctrls, short, long_ = [], [], [], []
    for label, B, H, S, D, causal in CASES:
        g = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        scale = D ** -0.5

        def run():
            return K.fused_attention_forward(q, k, v, None, causal, scale,
                                             IMPL_WMMA, BSHD)

        best_off = math.inf
        best_on = math.inf
        ctrl = math.inf
        for rnd in range(args.rounds):
            t = {False: [], True: []}
            for on in balanced_order((False, True), rnd):
                K.wmma_set_acc_formula(on and not self_control)
                t[on].append(graph_timed(run, args.iters))
            best_off = min([best_off] + t[False])
            best_on = min([best_on] + t[True])
            ctrl = min(ctrl, abs(t[False][0] / t[False][1] - 1.0))
        K.wmma_set_acc_formula(True)

        r = best_off / best_on
        gains.append(r)
        (short if S <= 128 else long_).append(r)
        if best_off >= CONTROL_FLOOR_US:
            ctrls.append(ctrl)
        print(f"  {label:<22} {best_off:9.1f} {best_on:9.1f} {r:7.3f}x "
              f"{ctrl*100:5.1f}%")

    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    print(f"    seq <= 128 : {gm(short):.3f}x over {len(short)} shapes  "
          f"(where a per-block cost can show)")
    print(f"    seq >= 512 : {gm(long_):.3f}x over {len(long_)} shapes  "
          f"(control: the fixed cost is amortised away)")
    if ctrls:
        print(f"  worst control over rows above {CONTROL_FLOOR_US:g} us: "
              f"+/-{max(ctrls)*100:.1f}%")


MODEL_SHAPES = [
    ("#1  B64 S128 d128 h4",  64,  128, 128,  4),
    ("#4  B16 S128 d128 h4",  16,  128, 128,  4),
    ("#12 B64 S32  d128 h4",  64,   32, 128,  4),
    ("#13 B64 S1024 d128 h4", 64, 1024, 128,  4),
    ("#9  B16 S128 d128 h1",  16,  128, 128,  1),
]


def model_section(K, args):
    import torch_transformer_benchmark as bench

    print()
    print("=== whole model, appendix shapes, 4 layers, causal ===")
    print(f"  {'shape':<22} {'probe ms':>9} {'formula':>9} {'ratio':>8} "
          f"{'ctrl':>6} {'max_abs':>10}")

    gains = []
    for label, b, sq, d, h in MODEL_SHAPES:
        cfg = bench.TransformerConfig(batch_size=b, seq_len=sq, d_model=d,
                                      num_heads=h, ffn_dim=d, num_layers=4,
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
            K.wmma_set_acc_formula(on)
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
            K.wmma_set_acc_formula(on)
            with torch.inference_mode():
                outs[on] = models[on](x, m).clone()
        err = (outs[True] - outs[False]).abs().max().item()

        r = best[False] / best[True]
        gains.append(r)
        print(f"  {label:<22} {best[False]:9.3f} {best[True]:9.3f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {err:10.2e}")

    K.wmma_set_acc_formula(True)
    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    print("  max_abs is 0.0 by construction: the closed form is only used when "
          "the one-time")
    print("  check confirms it reproduces the probe, so every rescale reads "
          "the same row.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--self-control", action="store_true")
    ap.add_argument("--op-only", action="store_true")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_acc_formula"):
        print(f"need a build with the closed-form accumulator map: "
              f"{kernel_ext.load_error()}")
        return 1

    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, fp32 tensors in "
          f"fp16 fragments, graph-timed")
    print()

    warm = torch.randn(8, 8, 128, 64, device=DEV)
    graph_timed(lambda: K.fused_attention_forward(
        warm, warm, warm, None, True, 0.125, IMPL_WMMA, BSHD), 10, 2, 5)

    op_section(K, args, args.self_control)
    if not args.op_only and not args.self_control:
        model_section(K, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
