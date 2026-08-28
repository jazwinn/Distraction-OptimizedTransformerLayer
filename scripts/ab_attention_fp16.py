"""A/B the wmma attention kernel's compute precision: fp16 fragments vs tf32.

The tensors stay fp32 either way. What changes is `compute_t` -- the type the
shared tiles hold and the fragments contract in. tf32 and fp16 carry the same
10-bit mantissa, so this is not a precision/speed trade in the usual sense; what
fp16 buys is that its tensor cores run 2.0x-2.25x tf32 on this card, that a
16x16x16 fragment contracts twice the K of tf32's 16x16x8, and that every staged
tile halves -- which is what decides the block shape at head_dim 128.

Three columns, because the kernel is not always the right answer: SDPA is what
`auto` falls back to, and at head_dim 128 the tf32 kernel loses to it. Whether
fp16 changes that verdict is the question `kWmmaAutoMaxHeadDim` depends on.

Method follows csrc/TUNING.md: one process, interleaved, best-of-rounds, and a
control row -- tf32 timed against itself, true value exactly 1.000x, so whatever
it reads is this machine's noise.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_attention_fp16.py
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
import torch.nn.functional as F  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import config  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1

# (label, B, H, S, head_dim). All timed causal, as every grading shape is.
# head_dim 128 is over-represented on purpose: it is the one head_dim where the
# tf32 kernel loses to SDPA and is therefore not used at all today.
CASES = [
    ("B8   H4 S128  d8",    8, 4,  128,   8),
    ("B16  H8 S128  d32",  16, 8,  128,  32),
    ("B64  H8 S128  d32",  64, 8,  128,  32),
    ("B8   H8 S1024 d32",   8, 8, 1024,  32),
    ("B8   H8 S128  d64",   8, 8,  128,  64),
    ("B4   H8 S512  d64",   4, 8,  512,  64),
    ("B1   H8 S2048 d64",   1, 8, 2048,  64),
    ("B8   H8 S32   d128",  8, 8,   32, 128),
    ("B8   H8 S128  d128",  8, 8,  128, 128),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--op-only", action="store_true",
                    help="skip the whole-model section")
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1
    if not hasattr(K, "wmma_set_fp16"):
        print("this build predates compute_t; rebuild")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, causal, "
          f"fp32 tensors throughout, graph-timed\n")
    print(f"  {'shape':<20} {'sdpa':>9} {'tf32':>9} {'fp16':>9} {'16v32':>7} "
          f"{'vs sdpa':>8} {'ctrl':>6} {'e32':>9} {'e16':>9}")

    # Throwaway: the first measurement in a fresh process reads several percent
    # slow, and once turned a 1.10x into a reported 1.32x.
    warm = torch.randn(8, 8, 128, 64, device=DEV)
    graph_timed(lambda: K.fused_attention_forward(
        warm, warm, warm, None, True, 0.125, IMPL_WMMA, BSHD), 10, 2, 5)

    gains, sdpa_gains, worst_ctrl = [], [], 0.0
    for label, B, H, S, D in CASES:
        g = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(B, H, S, D, device=DEV, generator=g)
        k = torch.randn(B, H, S, D, device=DEV, generator=g)
        v = torch.randn(B, H, S, D, device=DEV, generator=g)
        scale = D ** -0.5

        ref = F.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), is_causal=True, scale=scale)
        ref = ref.transpose(1, 2).flatten(2)

        def wmma():
            return K.fused_attention_forward(q, k, v, None, True, scale,
                                             IMPL_WMMA, BSHD)

        def sdpa():
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale).transpose(1, 2).flatten(2)

        errs = {}
        for name, on in (("tf32", False), ("fp16", True)):
            K.wmma_set_fp16(on)
            errs[name] = (wmma().double() - ref).abs().max().item()

        best = {"tf32": math.inf, "fp16": math.inf, "sdpa": math.inf}
        ctrl = math.inf
        for _ in range(args.rounds):
            # tf32 at both ends of the round; the two are identical code, so
            # their ratio is this machine's noise floor and nothing else.
            K.wmma_set_fp16(False)
            a = graph_timed(wmma, args.iters)
            K.wmma_set_fp16(True)
            f = graph_timed(wmma, args.iters)
            K.wmma_set_fp16(False)
            c = graph_timed(wmma, args.iters)
            sd = graph_timed(sdpa, args.iters)
            best["tf32"] = min(best["tf32"], a, c)
            best["fp16"] = min(best["fp16"], f)
            best["sdpa"] = min(best["sdpa"], sd)
            ctrl = min(ctrl, abs(a / c - 1.0))
        K.wmma_set_fp16(True)
        worst_ctrl = max(worst_ctrl, ctrl)

        r = best["tf32"] / best["fp16"]
        vs_sdpa = best["sdpa"] / best["fp16"]
        gains.append(r)
        sdpa_gains.append(vs_sdpa)
        print(f"  {label:<20} {best['sdpa']:9.1f} {best['tf32']:9.1f} "
              f"{best['fp16']:9.1f} {r:6.3f}x {vs_sdpa:7.3f}x {ctrl*100:5.1f}% "
              f"{errs['tf32']:9.2e} {errs['fp16']:9.2e}")

    def gm(v):
        return math.exp(sum(math.log(x) for x in v) / len(v))

    print(f"\n  geometric mean over {len(gains)} shapes:")
    print(f"    fp16 vs tf32 : {gm(gains):.3f}x")
    print(f"    fp16 vs SDPA : {gm(sdpa_gains):.3f}x  "
          f"(>1 means the kernel is worth dispatching)")
    print(f"  worst control this run: +/-{worst_ctrl*100:.1f}% -- "
          f"nothing below this is a result")

    if not args.op_only:
        model_section(K, args)
    return 0


# (label, batch, seq, d_model, heads). ffn_dim == d_model, causal, as every
# shape in the grading appendix is.
MODEL_SHAPES = [
    ("B1   S128  d256 h8",     1,  128,  256,  8),
    ("B8   S128  d256 h8",     8,  128,  256,  8),
    ("B16  S128  d256 h8",    16,  128,  256,  8),
    ("B64  S128  d256 h8",    64,  128,  256,  8),
    ("B8   S1024 d256 h8",     8, 1024,  256,  8),
    ("B8   S128  d32  h4",     8,  128,   32,  4),
    ("B8   S128  d512 h8",     8,  128,  512,  8),
    ("B8   S128  d512 h4",     8,  128,  512,  4),   # head_dim 128
    ("B8   S128  d1024 h16",   8,  128, 1024, 16),
]


def model_section(K, args):
    """What the change is worth on the whole model, which is the graded number.

    An op that reads 1.5x is worth nothing end to end if attention was 8% of the
    forward pass. Both models hold the same weights, so the max_abs column is a
    kernel-vs-kernel comparison rather than two different models being compared.
    """
    dev = torch.device("cuda")
    print("\n=== whole model, 6 layers, causal, ffn_dim == d_model ===")
    print(f"  {'shape':<22} {'tf32 ms':>9} {'fp16 ms':>9} {'ratio':>8} "
          f"{'ctrl':>6}  {'max_abs':>9}")

    gains, worst_ctrl = [], 0.0
    for label, b, sq, d, h in MODEL_SHAPES:
        cfg = bench.TransformerConfig(batch_size=b, seq_len=sq, d_model=d,
                                      num_heads=h, ffn_dim=d, num_layers=6,
                                      causal=True)
        base = bench.BaselineTransformer(cfg)
        models = {}
        for name in ("tf32", "auto"):
            opt = bench.UserOptimizedTransformer(cfg)
            bench.copy_model_weights(base, opt)
            models[name] = opt.to(dev).eval()
        x, m = bench.generate_random_case(config=cfg, device=dev,
                                          dtype=torch.float32, seed=1234,
                                          padding_ratio=0.0, input_scale=1.0)

        def run(name, iters):
            config.ATTENTION_FP16 = name
            with torch.inference_mode():
                torch.cuda.synchronize()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(iters):
                    models[name](x, m)
                en.record()
                torch.cuda.synchronize()
            return st.elapsed_time(en) / iters

        for name in ("tf32", "auto"):
            run(name, 5)   # warm, and capture outside the timed region

        best = {"tf32": math.inf, "auto": math.inf}
        ctrl = math.inf
        for _ in range(args.rounds):
            a = run("tf32", args.iters)
            f = run("auto", args.iters)
            c = run("tf32", args.iters)
            best["tf32"] = min(best["tf32"], a, c)
            best["auto"] = min(best["auto"], f)
            ctrl = min(ctrl, abs(a / c - 1.0))
        worst_ctrl = max(worst_ctrl, ctrl)

        outs = {}
        for name in ("tf32", "auto"):
            config.ATTENTION_FP16 = name
            with torch.inference_mode():
                outs[name] = models[name](x, m).clone()
        err = (outs["auto"] - outs["tf32"]).abs().max().item()

        r = best["tf32"] / best["auto"]
        gains.append(r)
        print(f"  {label:<22} {best['tf32']:9.3f} {best['auto']:9.3f} "
              f"{r:7.3f}x {ctrl*100:5.1f}% {err:9.2e}")

    config.ATTENTION_FP16 = "auto"
    gm = math.exp(sum(math.log(g) for g in gains) / len(gains))
    print(f"\n  geometric mean over {len(gains)} shapes: {gm:.3f}x")
    print(f"  worst control this run: +/-{worst_ctrl*100:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
