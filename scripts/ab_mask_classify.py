"""A/B per-tile mask classification against testing every score element.

Every score element used to evaluate four predicates -- two bounds tests, the
causal test, and an explicit-mask lookup -- plus the address arithmetic under
them. FlashAttention-2's `mask.h` does not: it templates so that interior tiles
carry no row-index computation at all and only the diagonal block pays. This is
the same split, decided per (warp, key tile) at run time.

The win should scale with the fraction of tiles that are interior, so the table
is ordered to show that: dense shapes where every tile qualifies, causal shapes
where one tile per block is diagonal, and masked shapes where the softmax fast
path is off but the K/V staging one still applies -- which isolates the staging
half of the change.

Sampling is symmetric -- two timings per side per round. Timing one side twice
and the other once and taking min of each compares min-of-2 against min-of-1
and biases the ratio by about 2.5%; see csrc/TUNING.md.

    python scripts/ab_mask_classify.py
    python scripts/ab_mask_classify.py --self-control
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

# (label, B, H, S, head_dim, causal, mask_kind)
CASES = [
    # dense: every interior tile qualifies, so this is the ceiling
    ("dense  B8 H8  S512  d32",  8,  8,  512,  32, False, "none"),
    ("dense  B8 H8  S1024 d32",  8,  8, 1024,  32, False, "none"),
    ("dense  B4 H8  S2048 d32",  4,  8, 2048,  32, False, "none"),
    ("dense  B8 H16 S512  d16",  8, 16,  512,  16, False, "none"),
    ("dense  B8 H8  S512  d64",  8,  8,  512,  64, False, "none"),
    # causal: one diagonal tile per block, the rest interior
    ("causal B8 H8  S128  d32",  8,  8,  128,  32, True,  "none"),
    ("causal B8 H8  S512  d32",  8,  8,  512,  32, True,  "none"),
    ("causal B8 H8  S1024 d32",  8,  8, 1024,  32, True,  "none"),
    ("causal B4 H8  S2048 d32",  4,  8, 2048,  32, True,  "none"),
    ("causal B8 H16 S512  d16",  8, 16,  512,  16, True,  "none"),
    ("causal B8 H8  S512  d64",  8,  8,  512,  64, True,  "none"),
    ("causal B4 H8  S1024 d64",  4,  8, 1024,  64, True,  "none"),
    # ragged S: the last query block and the last key tile never qualify
    ("causal B8 H8  S500  d32",  8,  8,  500,  32, True,  "none"),
    # masked: the SOFTMAX fast path is disabled outright (an explicit mask
    # cannot be classified away), but the K/V staging one is not -- it only
    # asks whether the tile is inside S. So this group isolates the staging
    # half of the change rather than acting as a control.
    ("mask   B8 H8  S512  d32",  8,  8,  512,  32, False, "keypad"),
    ("mask   B8 H8  S512  d32 c", 8, 8,  512,  32, True,  "keypad"),
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


def op_section(K, args, self_control):
    if self_control:
        print("  SELF-CONTROL: both columns are classification-off, true "
              "ratio 1.000x")
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
        for _ in range(args.rounds):
            K.wmma_set_mask_classify(False)
            a = graph_timed(run, args.iters)
            K.wmma_set_mask_classify(not self_control)
            f = graph_timed(run, args.iters)
            K.wmma_set_mask_classify(False)
            c = graph_timed(run, args.iters)
            K.wmma_set_mask_classify(not self_control)
            d = graph_timed(run, args.iters)
            best_off = min(best_off, a, c)
            best_on = min(best_on, f, d)
            ctrl = min(ctrl, abs(a / c - 1.0))
        K.wmma_set_mask_classify(True)

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
    for name in ("dense", "causal", "mask"):
        if name in groups:
            note = ("  (softmax fast path off; the K/V staging half "
                    "alone)") if name == "mask" else ""
            print(f"    {name:<7} {gm(groups[name]):.3f}x over "
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
            K.wmma_set_mask_classify(on)
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
        for _ in range(args.rounds):
            a = run(False, args.iters)
            f = run(True, args.iters)
            c = run(False, args.iters)
            dd = run(True, args.iters)
            best[False] = min(best[False], a, c)
            best[True] = min(best[True], f, dd)
            ctrl = min(ctrl, abs(a / c - 1.0))

        outs = {}
        for on in (False, True):
            K.wmma_set_mask_classify(on)
            with torch.inference_mode():
                outs[on] = models[on](x, m).clone()
        err = (outs[True] - outs[False]).abs().max().item()

        r = best[False] / best[True]
        gains.append(r)
        print(f"  {label:<20} {best[False]:9.3f} {best[True]:9.3f} {r:7.3f}x "
              f"{ctrl*100:5.1f}% {err:10.2e}")

    K.wmma_set_mask_classify(True)
    print()
    print(f"  geometric mean over {len(gains)} shapes: {gm(gains):.3f}x   "
          f"best {max(gains):.3f}x  worst {min(gains):.3f}x")
    print("  max_abs is 0.0 everywhere by construction: classification skips "
          "only tests that would have passed.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--self-control", action="store_true")
    ap.add_argument("--op-only", action="store_true")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None or not hasattr(K, "wmma_set_mask_classify"):
        print(f"need a build with mask classification: {kernel_ext.load_error()}")
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
