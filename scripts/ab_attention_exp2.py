"""A/B the wmma attention softmax: base-2 with a folded scale vs __expf.

What changes is where the softmax's two per-score-element multiplies are paid.
`__expf(x)` is not one instruction -- it is `ex2.approx(x * log2e)` -- so the
original does `s * scale` and then, inside __expf, another `* log2e`, both once
per score element. Folding `scale * log2e` into Q at staging time prices them
once per staged Q element instead: BLOCK_M*head_dim rather than
BLOCK_M*BLOCK_N per key tile. It also drops the `sv == -inf` test, because
ex2.approx(-inf) is defined to return +0.

SASS says this removes 49 FMUL and 16 FSETP from the head_dim 32 key-tile loop
body -- and that the compiler hands most of it back as LOP3/P2R at the head_dims
whose BLOCK_N is 32. Which of those two wins is exactly what this measures.

Method follows csrc/TUNING.md: one process, interleaved, best-of-rounds, and a
control row -- the OFF path timed against itself, true value exactly 1.000x, so
whatever it reads is this machine's noise floor.

    python scripts/ab_attention_exp2.py
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

from verify_kernel import reference_attention_f64  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1

# (label, B, H, S, head_dim). All causal, as every grading shape is. head_dim
# 16 and 32 are over-represented: SASS says BLOCK_N 32 is where the compiler
# claws the savings back, and those are the grading set's common widths.
CASES = [
    ("B8   H4 S128  d8",    8, 4,  128,   8),
    ("B8   H16 S128 d16",   8, 16, 128,  16),
    ("B8   H8 S512  d16",   8, 8,  512,  16),
    ("B1   H8 S128  d32",   1, 8,  128,  32),
    ("B16  H8 S128  d32",  16, 8,  128,  32),
    ("B64  H8 S128  d32",  64, 8,  128,  32),
    ("B8   H8 S1024 d32",   8, 8, 1024,  32),
    ("B8   H8 S128  d64",   8, 8,  128,  64),
    ("B4   H8 S512  d64",   4, 8,  512,  64),
    ("B1   H8 S2048 d64",   1, 8, 2048,  64),
    ("B4   H8 S512  d128",  4, 8,  512, 128),
    ("B2   H8 S1024 d128",  2, 8, 1024, 128),
]


def graph_timed(fn, iters=30, reps=5, per_graph=10):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            for _ in range(per_graph):
                fn()
    except Exception:
        return None
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1e3 / per_graph)
    return best


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--self-control", action="store_true",
                    help="run mode 1 against itself: every mode slot gets the "
                         "same kernel, so every true ratio is exactly 1.000 "
                         "and whatever the table reads is this machine noise")
    ap.add_argument("--op-only", action="store_true",
                    help="skip the whole-model section")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1
    if not hasattr(K, "wmma_set_softmax_mode"):
        print("this build predates the softmax modes; rebuild")
        return 1

    # Under --self-control every slot below is handed mode 1, so the "1v0" and
    # "2v0" columns compare identical code and their true value is 1.000x.
    def setmode(m):
        K.wmma_set_softmax_mode(1 if args.self_control else m)

    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, causal, fp32 "
          f"tensors contracted in fp16, graph-timed\n")
    print(f"  {'shape':<20} {'mode0 us':>9} {'mode1 us':>9} {'mode2 us':>9} "
          f"{'1v0':>7} {'2v0':>7} {'ctrl':>6} {'err0':>9} {'err1':>9} {'err2':>9}")

    # Throwaway: the first measurement in a fresh process reads several percent
    # slow, and once turned a 1.10x into a reported 1.32x.
    warm = torch.randn(8, 8, 128, 64, device=DEV)
    graph_timed(lambda: K.fused_attention_forward(
        warm, warm, warm, None, True, 0.125, IMPL_WMMA, BSHD), 10, 2, 5)

    g1, g2, worst_ctrl = [], [], 0.0
    worst = {0: 0.0, 1: 0.0, 2: 0.0}
    for label, B, H, S, D in CASES:
        gen = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(B, H, S, D, device=DEV, generator=gen)
        k = torch.randn(B, H, S, D, device=DEV, generator=gen)
        v = torch.randn(B, H, S, D, device=DEV, generator=gen)
        scale = D ** -0.5

        ref = reference_attention_f64(q, k, v, None, True, scale, layout=1)

        def run():
            return K.fused_attention_forward(q, k, v, None, True, scale,
                                             IMPL_WMMA, BSHD)

        errs = {}
        for m in (0, 1, 2):
            setmode(m)
            errs[m] = (run().double() - ref).abs().max().item()
            worst[m] = max(worst[m], errs[m])

        best = {0: math.inf, 1: math.inf, 2: math.inf}
        ctrl = math.inf
        for _ in range(args.rounds):
            # mode 0 at both ends of the round; identical code, so their ratio
            # is this machine's noise and nothing else.
            timed = {}
            for m in (0, 1, 2):
                setmode(m)
                timed[m] = graph_timed(run, args.iters)
            setmode(0)
            c = graph_timed(run, args.iters)
            best[0] = min(best[0], timed[0], c)
            best[1] = min(best[1], timed[1])
            best[2] = min(best[2], timed[2])
            ctrl = min(ctrl, abs(timed[0] / c - 1.0))
        K.wmma_set_softmax_mode(1)
        worst_ctrl = max(worst_ctrl, ctrl)

        r1 = best[0] / best[1]
        r2 = best[0] / best[2]
        g1.append(r1)
        g2.append(r2)
        print(f"  {label:<20} {best[0]:9.1f} {best[1]:9.1f} {best[2]:9.1f} "
              f"{r1:6.3f}x {r2:6.3f}x {ctrl*100:5.1f}% "
              f"{errs[0]:9.2e} {errs[1]:9.2e} {errs[2]:9.2e}")

    print()
    print(f"  geometric mean over {len(g1)} shapes:")
    print(f"    mode 1 (base-2, Q untouched) vs mode 0 : {gm(g1):.3f}x   "
          f"best {max(g1):.3f}x  worst {min(g1):.3f}x")
    print(f"    mode 2 (scale folded into Q) vs mode 0 : {gm(g2):.3f}x   "
          f"best {max(g2):.3f}x  worst {min(g2):.3f}x")
    print(f"  worst control this run: +/-{worst_ctrl*100:.1f}% -- nothing "
          f"below this is a result")
    print(f"  worst max_abs, harness atol is 2e-3:")
    for m in (0, 1, 2):
        flag = "  <-- OVER BUDGET" if worst[m] > 2e-3 else ""
        print(f"    mode {m}: {worst[m]:.2e}{flag}")
    if not args.op_only:
        model_section(K, args)
    return 0



# (label, batch, seq, d_model, heads). ffn_dim == d_model, causal, as every
# shape in the grading appendix is. Weighted toward narrow head_dims and toward
# the shapes where csrc/TUNING.md measured attention as a large share of the
# forward -- those are the two things that decide what an op-level 1.04x is
# worth end to end.
MODEL_SHAPES = [
    ("B8   S128  d32   h4",    8,  128,   32,  4),   # head_dim 8,  attn 20.8%
    ("B8   S128  d256  h16",   8,  128,  256, 16),   # head_dim 16, attn 17.6%
    ("B1   S128  d256  h8",    1,  128,  256,  8),   # head_dim 32, attn 26.9%
    ("B16  S128  d256  h8",   16,  128,  256,  8),
    ("B64  S128  d256  h8",   64,  128,  256,  8),
    ("B8   S1024 d256  h8",    8, 1024,  256,  8),   # head_dim 32, attn 42.0%
    ("B8   S128  d512  h8",    8,  128,  512,  8),   # head_dim 64
]


def model_section(K, args):
    """What the change is worth on the whole model, which is the graded number.

    Two model instances, not one: `self._graphs` is per instance, so setting the
    mode before each model's warmup bakes that kernel into that model's captured
    graph. Flipping the flag after capture would do nothing to a replay -- the
    same trap tile_set_split_kv documents.
    """
    import torch_transformer_benchmark as bench

    dev = torch.device("cuda")
    print()
    print("=== whole model, 6 layers, causal, ffn_dim == d_model ===")
    print(f"  {'shape':<22} {'mode0 ms':>9} {'mode1 ms':>9} {'ratio':>8} "
          f"{'ctrl':>6}  {'max_abs':>9}")

    gains, worst_ctrl, worst_err = [], 0.0, 0.0
    for label, b, sq, d, h in MODEL_SHAPES:
        cfg = bench.TransformerConfig(batch_size=b, seq_len=sq, d_model=d,
                                      num_heads=h, ffn_dim=d, num_layers=6,
                                      causal=True)
        base = bench.BaselineTransformer(cfg)
        models = {}
        for mode in (0, 1):
            opt = bench.UserOptimizedTransformer(cfg)
            bench.copy_model_weights(base, opt)
            models[mode] = opt.to(dev).eval()
        x, m = bench.generate_random_case(config=cfg, device=dev,
                                          dtype=torch.float32, seed=1234,
                                          padding_ratio=0.0, input_scale=1.0)

        def run(mode, iters):
            K.wmma_set_softmax_mode(mode)
            with torch.inference_mode():
                torch.cuda.synchronize()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(iters):
                    models[mode](x, m)
                en.record()
                torch.cuda.synchronize()
            return st.elapsed_time(en) / iters

        for mode in (0, 1):
            run(mode, 5)   # warm, and capture outside the timed region

        best = {0: math.inf, 1: math.inf}
        ctrl = math.inf
        for _ in range(args.rounds):
            a = run(0, args.iters)
            f = run(1, args.iters)
            c = run(0, args.iters)
            best[0] = min(best[0], a, c)
            best[1] = min(best[1], f)
            ctrl = min(ctrl, abs(a / c - 1.0))
        worst_ctrl = max(worst_ctrl, ctrl)

        outs = {}
        for mode in (0, 1):
            K.wmma_set_softmax_mode(mode)
            with torch.inference_mode():
                outs[mode] = models[mode](x, m).clone()
        err = (outs[1] - outs[0]).abs().max().item()
        worst_err = max(worst_err, err)

        r = best[0] / best[1]
        gains.append(r)
        print(f"  {label:<22} {best[0]:9.3f} {best[1]:9.3f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {err:9.2e}")

    K.wmma_set_softmax_mode(1)
    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    print(f"  worst control this run: +/-{worst_ctrl*100:.1f}%")
    print(f"  worst mode1-vs-mode0 output difference: {worst_err:.2e}")

if __name__ == "__main__":
    raise SystemExit(main())
