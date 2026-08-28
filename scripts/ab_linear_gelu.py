"""A/B the fused Linear+GELU against cuBLAS + F.gelu, on the whole model.

Three ways, because two separate changes are being scored: the fusion itself
(one kernel instead of two) and the precision the fused kernel computes in
(fp16 fragments instead of tf32). `off` is cuBLAS + F.gelu, `tf32` is the fused
kernel at the old precision, `auto` is the fused kernel at fp16. So tf32-vs-off
isolates the fusion, and auto-vs-tf32 isolates the precision.

scripts/tune_linear_gelu.py answers "is the kernel faster than the pair it
replaces". That is not the question the grading number asks. This one is: what
does the *model* do, running the way the harness runs it -- six layers, causal,
CUDA graphs on, dispatch already amortised. An op that reads 3.19x can be worth
almost nothing end to end if it was 4% of the forward pass, and an op-level win
measured eager can be a regression once dispatch is gone. Both have happened
here, so neither table substitutes for the other.

Method follows csrc/TUNING.md:

  * One process, interleaved, best-of-rounds. LINEAR_GELU is flipped between
    timings rather than compared across runs.
  * One model instance per setting, so a capture made under one cannot serve
    another. (_graph_key includes LINEAR_GELU too, so a shared instance would
    also be correct -- this is belt and braces.)
  * A control row: "off" timed against itself, once at each end of every round.
    Identical code both sides, true value exactly 1.000x, so whatever it reads
    is this machine's noise. Nothing smaller than that is a result.

Shapes are the grading appendix's: all causal, ffn_dim == d_model.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_linear_gelu.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402,F401

import torch  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import config  # noqa: E402

DEV = torch.device("cuda")

SETTINGS = ["off", "tf32", "auto"]

# (label, batch, seq, d_model, heads). ffn_dim == d_model, as every shape in the
# grading appendix has it. With fp16 fragments the kernel wins at every shape,
# so there is no gate left to test -- what varies here is how much of the
# forward pass the FFN is, which is what decides the end-to-end payoff.
SHAPES = [
    ("B1   S128  d256 h8",     1,  128,  256,  8),
    ("B4   S128  d256 h8",     4,  128,  256,  8),
    ("B8   S128  d256 h8",     8,  128,  256,  8),
    ("B16  S128  d256 h8",    16,  128,  256,  8),
    ("B64  S128  d256 h8",    64,  128,  256,  8),
    ("B128 S128  d256 h8",   128,  128,  256,  8),
    ("B8   S1024 d256 h8",     8, 1024,  256,  8),
    ("B8   S128  d32  h4",     8,  128,   32,  4),
    ("B32  S128  d32  h4",    32,  128,   32,  4),
    ("B8   S128  d512 h8",     8,  128,  512,  8),
    ("B8   S128  d1024 h16",   8,  128, 1024, 16),
]


def build_models(b, s, d, h, layers=6):
    """One model per LINEAR_GELU setting, all holding the SAME weights.

    Separate instances so a capture made under one setting cannot serve another
    -- but all loaded from one baseline, because they are also the sides of the
    accuracy comparison. An earlier version constructed a fresh randomly
    initialised baseline per model and reported a max_abs of ~3.5 on shapes
    where both sides run identical code; that was two different models being
    compared, not two kernels.
    """
    cfg = bench.TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                  ffn_dim=d, num_layers=layers, causal=True)
    base = bench.BaselineTransformer(cfg)
    models = {}
    for name in SETTINGS:
        opt = bench.UserOptimizedTransformer(cfg)
        bench.copy_model_weights(base, opt)
        models[name] = opt.to(DEV).eval()
    x, m = bench.generate_random_case(config=cfg, device=DEV, dtype=torch.float32,
                                      seed=1234, padding_ratio=0.0, input_scale=1.0)
    return models, x, m


def time_ms(model, x, m, iters, setting):
    config.LINEAR_GELU = setting
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(x, m)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if kernel_ext.get_kernels() is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, "
          f"causal, ffn_dim == d_model, 6 layers, CUDA_GRAPH=auto\n")

    print(f"  {'shape':<22} {'off ms':>9} {'tf32 ms':>9} {'fp16 ms':>9} "
          f"{'fuse':>7} {'fp16':>7} {'total':>7} {'ctrl':>6}  {'max_abs':>9}")

    fuse_g, prec_g, tot_g = [], [], []
    worst_ctrl = 0.0
    for label, b, s, d, h in SHAPES:
        models, x, m = build_models(b, s, d, h)

        # Warm each under its own setting, so capture happens outside the timed
        # region and no model's first call is being measured.
        for name in SETTINGS:
            config.LINEAR_GELU = name
            with torch.inference_mode():
                for _ in range(5):
                    models[name](x, m)

        best = {name: math.inf for name in SETTINGS}
        best_ctrl = math.inf
        for _ in range(args.rounds):
            # Round-robin, with "off" timed at both ends. The two off timings
            # are identical code, so their ratio is this machine's noise floor.
            first_off = time_ms(models["off"], x, m, args.iters, "off")
            for name in SETTINGS[1:]:
                best[name] = min(best[name],
                                 time_ms(models[name], x, m, args.iters, name))
            second_off = time_ms(models["off"], x, m, args.iters, "off")
            best["off"] = min(best["off"], first_off, second_off)
            best_ctrl = min(best_ctrl, abs(first_off / second_off - 1.0))
        worst_ctrl = max(worst_ctrl, best_ctrl)

        outs = {}
        for name in SETTINGS:
            config.LINEAR_GELU = name
            with torch.inference_mode():
                outs[name] = models[name](x, m).clone()
        err = (outs["auto"] - outs["off"]).abs().max().item()

        fuse = best["off"] / best["tf32"]
        prec = best["tf32"] / best["auto"]
        total = best["off"] / best["auto"]
        fuse_g.append(fuse)
        prec_g.append(prec)
        tot_g.append(total)
        print(f"  {label:<22} {best['off']:9.3f} {best['tf32']:9.3f} "
              f"{best['auto']:9.3f} {fuse:6.3f}x {prec:6.3f}x {total:6.3f}x "
              f"{best_ctrl * 100:5.1f}% {err:9.2e}")

    config.LINEAR_GELU = "auto"

    def gm(v):
        return math.exp(sum(math.log(g) for g in v) / len(v))

    print(f"\n  geometric mean over {len(tot_g)} shapes:")
    print(f"    fusion   (off  -> tf32): {gm(fuse_g):.3f}x")
    print(f"    fp16     (tf32 -> fp16): {gm(prec_g):.3f}x")
    print(f"    together (off  -> fp16): {gm(tot_g):.3f}x")
    print(f"  worst control this run: +/-{worst_ctrl * 100:.1f}% -- "
          f"nothing below this is a result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
