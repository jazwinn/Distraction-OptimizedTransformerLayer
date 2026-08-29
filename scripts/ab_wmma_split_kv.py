"""A/B the wmma kernel's split-KV (Flash-Decoding) path.

The attention grid is (ceil(S/BLOCK_M), H, B) -- query side only -- so a shape
with few queries and many keys launches a grid too small to fill the card and
each block does a long serial key loop. Splitting that key range across extra
blocks buys parallelism that no block shape can, at the cost of a partial
workspace and a second pass.

Three sections, in the order the decisions were made:

  sweep   force every split count on the candidate shapes, so the gate's rule
          can be checked against what actually wins rather than assumed.
  A/B     the rule as shipped, against splitting off, with a self-control.
  model   what it is worth end to end, which is the graded number.

Two traps from csrc/TUNING.md are designed around here. Eager op timing is
dispatch-bound at these sizes and reported the whole optimization as a loss
once, so everything is timed under CUDA graph replay. And a control row under
~15 us is meaningless, so short rows are excluded from the control statistic
rather than being allowed to set it.

    python scripts/ab_wmma_split_kv.py
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
CONTROL_FLOOR_US = 15.0

# (label, B, H, S, head_dim, causal)
CASES = [
    ("B1  H4  S128  d32 c",   1,  4,  128,  32, True),
    ("B1  H8  S128  d32 c",   1,  8,  128,  32, True),
    ("B2  H8  S128  d32 c",   2,  8,  128,  32, True),
    ("B4  H8  S128  d32 c",   4,  8,  128,  32, True),
    ("B1  H8  S512  d32 c",   1,  8,  512,  32, True),
    ("B1  H8  S1024 d32 c",   1,  8, 1024,  32, True),
    ("B1  H8  S128  d64 c",   1,  8,  128,  64, True),
    ("B1  H8  S512  d64 c",   1,  8,  512,  64, True),
    ("B1  H4  S256  d64 c",   1,  4,  256,  64, True),
    ("B1  H8  S128  d16 c",   1,  8,  128,  16, True),
    ("B2  H2  S2048 d64 c",   2,  2, 2048,  64, True),
    # controls: grids that already fill the card, so the gate should decline
    ("B8  H8  S128  d32 c",   8,  8,  128,  32, True),
    ("B64 H8  S128  d32 c",  64,  8,  128,  32, True),
    ("B8  H8  S1024 d32 c",   8,  8, 1024,  32, True),
]

SWEEP_COUNTS = [1, 2, 3, 4, 6, 8]


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


def tensors(B, H, S, D):
    g = torch.Generator(device="cuda").manual_seed(1234)
    return (torch.randn(B, H, S, D, device=DEV, generator=g),
            torch.randn(B, H, S, D, device=DEV, generator=g),
            torch.randn(B, H, S, D, device=DEV, generator=g))


def sweep_section(K, args):
    print("=== forced split counts: what actually wins, per shape ===")
    head = "  {:<20} {:>6}".format("shape", "rule")
    for n in SWEEP_COUNTS:
        head += f" {('n=' + str(n)):>8}"
    head += f" {'best':>6}"
    print(head)

    for label, B, H, S, D, causal in CASES:
        q, k, v = tensors(B, H, S, D)
        scale = D ** -0.5
        rule = K.wmma_split_count(B, H, S, D, causal)

        def run():
            return K.fused_attention_forward(q, k, v, None, causal, scale,
                                             IMPL_WMMA, BSHD)

        # Every count gets its own n=1 baseline timed immediately next to it,
        # and the whole thing repeats. Timing each count once in a single
        # sequential pass produced an 8.2x on this very shape that alternation
        # showed to be flat -- the same failure mode csrc/TUNING.md records for
        # min-of-2 across processes.
        times = {n: math.inf for n in SWEEP_COUNTS}
        base = math.inf
        for _ in range(args.rounds):
            for n in SWEEP_COUNTS:
                K.wmma_set_split_count(n)
                times[n] = min(times[n], graph_timed(run, args.iters))
                K.wmma_set_split_count(1)
                base = min(base, graph_timed(run, args.iters))
        K.wmma_set_split_count(0)
        times[1] = min(times[1], base)
        row = f"  {label:<20} {rule:>6}"
        for n in SWEEP_COUNTS:
            row += f" {base / times[n]:7.3f}x"
        best_n = min(SWEEP_COUNTS, key=lambda n: times[n])
        row += f" {best_n:>6}"
        print(row)
    print()
    print("  ratios are against n=1. `rule` is what the gate picks, `best` what")
    print("  the sweep found; they should agree wherever the win is real.")
    print()


def ab_section(K, args, self_control=False):
    tag = "  SELF-CONTROL: both columns are split-off, true ratio 1.000x" \
        if self_control else ""
    print("=== the rule as shipped, against splitting off ===" +
          ("\n" + tag if tag else ""))
    print(f"  {'shape':<20} {'off us':>9} {'on us':>9} {'ratio':>8} "
          f"{'ctrl':>6} {'splits':>7}")

    gains, ctrls = [], []
    for label, B, H, S, D, causal in CASES:
        q, k, v = tensors(B, H, S, D)
        scale = D ** -0.5
        n = 1 if self_control else K.wmma_split_count(B, H, S, D, causal)

        def run():
            return K.fused_attention_forward(q, k, v, None, causal, scale,
                                             IMPL_WMMA, BSHD)

        best_off = math.inf
        best_on = math.inf
        ctrl = math.inf
        # off, on, off, on -- TWO samples each per round. Timing off twice and
        # on once and taking min-of-2 against min-of-1 biases the ratio against
        # the change by ~2.5%, which is enough to sink a 1.03x. Measured: with
        # the asymmetric form the DECLINED rows, which are identical code both
        # sides, read a geometric mean of 0.975x instead of 1.000x.
        for _ in range(args.rounds):
            K.wmma_set_split_kv(False)
            a = graph_timed(run, args.iters)
            K.wmma_set_split_kv(not self_control)
            f = graph_timed(run, args.iters)
            K.wmma_set_split_kv(False)
            c = graph_timed(run, args.iters)
            K.wmma_set_split_kv(not self_control)
            d = graph_timed(run, args.iters)
            best_off = min(best_off, a, c)
            best_on = min(best_on, f, d)
            ctrl = min(ctrl, abs(a / c - 1.0))
        K.wmma_set_split_kv(True)

        r = best_off / best_on
        gains.append(r)
        # A control row under ~15 us measures dispatch jitter, not the kernel.
        if best_off >= CONTROL_FLOOR_US:
            ctrls.append(ctrl)
        mark = "" if best_off >= CONTROL_FLOOR_US else "  (short)"
        print(f"  {label:<20} {best_off:9.1f} {best_on:9.1f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {n:>7}{mark}")

    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    split_only = [g for g, c in zip(gains, CASES)
                  if K.wmma_split_count(c[1], c[2], c[3], c[4], c[5]) > 1]
    if split_only and not self_control:
        print(f"  over the {len(split_only)} shapes the gate actually splits: "
              f"{gm(split_only):.3f}x")
    if ctrls:
        print(f"  worst control over rows above {CONTROL_FLOOR_US:g} us: "
              f"+/-{max(ctrls)*100:.1f}% -- nothing below this is a result")
    print()
    return gains


# head_dim is d_model/heads, and the gate only fires for small grids with a
# real key loop, so the list is weighted to head_dim 64 at batch 1. The rest
# are controls the gate should decline -- they show max_abs 0.0, which is how
# you can tell a row is identical code rather than a measured tie.
MODEL_SHAPES = [
    ("B1  S128  d512 h8",    1,  128,  512,  8),   # head_dim 64, splits
    ("B1  S256  d512 h8",    1,  256,  512,  8),   # head_dim 64
    ("B1  S128  d256 h4",    1,  128,  256,  4),   # head_dim 64
    ("B1  S256  d256 h4",    1,  256,  256,  4),   # head_dim 64
    ("B1  S128  d256 h8",    1,  128,  256,  8),   # head_dim 32, n_kt 2
    ("B1  S512  d256 h8",    1,  512,  256,  8),
    ("B2  S128  d256 h8",    2,  128,  256,  8),
    ("B8  S128  d256 h8",    8,  128,  256,  8),
    ("B8  S128  d32  h4",    8,  128,   32,  4),
]


def model_section(K, args):
    import torch_transformer_benchmark as bench

    print("=== whole model, 6 layers, causal, ffn_dim == d_model ===")
    print("  Two instances: _graphs is per instance, so the flag must be set")
    print("  before each one's warmup or the capture bakes in the wrong kernel.")
    print(f"  {'shape':<20} {'off ms':>9} {'on ms':>9} {'ratio':>8} "
          f"{'ctrl':>6} {'max_abs':>10}")

    gains, split, declined = [], [], []
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
            K.wmma_set_split_kv(on)
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
        # Two samples each, for the reason in ab_section: the declined rows are
        # identical code and must read 1.000x, and under asymmetric sampling
        # they read 0.975x.
        for _ in range(args.rounds):
            a = run(False, args.iters)
            f = run(True, args.iters)
            c = run(False, args.iters)
            d = run(True, args.iters)
            best[False] = min(best[False], a, c)
            best[True] = min(best[True], f, d)
            ctrl = min(ctrl, abs(a / c - 1.0))

        outs = {}
        for on in (False, True):
            K.wmma_set_split_kv(on)
            with torch.inference_mode():
                outs[on] = models[on](x, m).clone()
        err = (outs[True] - outs[False]).abs().max().item()

        r = best[False] / best[True]
        gains.append(r)
        (split if err > 0.0 else declined).append(r)
        print(f"  {label:<20} {best[False]:9.3f} {best[True]:9.3f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {err:10.2e}")

    K.wmma_set_split_kv(True)
    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    # Rows the gate declined ran identical code both sides, so their spread is
    # this harness's noise floor -- a control that costs nothing to collect.
    if declined:
        print(f"  the {len(declined)} DECLINED rows (identical code, true "
              f"ratio 1.000x): geomean {gm(declined):.3f}x, "
              f"{min(declined):.3f}x - {max(declined):.3f}x")
    if split:
        print(f"  the {len(split)} rows that actually split: "
              f"geomean {gm(split):.3f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--self-control", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--op-only", action="store_true")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_split_kv"):
        print(f"need a build with split-KV: {kernel_ext.load_error()}")
        return 1

    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, fp32 tensors in "
          f"fp16 fragments, graph-timed")
    print()

    warm = torch.randn(8, 8, 128, 64, device=DEV)
    graph_timed(lambda: K.fused_attention_forward(
        warm, warm, warm, None, True, 0.125, IMPL_WMMA, BSHD), 10, 2, 5)

    if args.sweep:
        sweep_section(K, args)
    ab_section(K, args, self_control=args.self_control)
    if not args.op_only and not args.self_control:
        model_section(K, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
