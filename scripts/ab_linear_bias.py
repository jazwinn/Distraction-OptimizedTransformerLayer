"""A/B the custom GEMM against cuBLAS on the projections that have no activation.

Companion to ab_linear_gelu.py, same method, different call site: that one
scores the FFN's Linear+GELU fusion, this one scores replacing plain `F.linear`
-- the fused QKV projection, and later out_proj and ffn_out -- with the same
kernel compiled without its activation.

Why this is worth a flag of its own: cuBLAS does not serve these in one way. At
the wide grading shapes it picks a TF32 tensorop tile and runs near the card's
TF32 roofline, where there is little to win. At the small-K ones -- shapes 7 and
12 -- it picks `cutlass_80_simt_sgemm_128x256`, which is not using tensor cores
at all, and there is a great deal to win. A single geomean over the set hides
both facts, so the per-shape column is the result and the geomean is a summary.

  * off  -> F.linear, i.e. cuBLAS
  * tf32 -> the custom kernel with tf32 fragments (isolates the *kernel*)
  * auto -> the custom kernel with fp16 fragments (isolates the *precision*)

Method follows csrc/TUNING.md and ab_common.balanced_order: one process,
interleaved, best-of-rounds, one model instance per setting, and a control row
that times "off" against itself so this machine's noise floor is printed next to
every result.

    cmd.exe /c scripts\devenv.bat python scripts\ab_linear_bias.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_common import balanced_order  # noqa: E402

import kernel_ext  # noqa: E402,F401

import torch  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import config  # noqa: E402

DEV = torch.device("cuda")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which config global this run flips, and the values it takes. Every arm gets
# its own model instance, so a graph captured under one setting can never serve
# another; config._graph_key names these too, as belt and braces.
KNOB = "LINEAR_BIAS"
SETTINGS = ["off", "tf32", "auto"]


def load_shapes(wanted):
    with open(os.path.join(ROOT, "dashboard", "presets.json")) as f:
        presets = json.load(f)["presets"]
    return [(i, presets[i - 1]) for i in wanted]


def build_models(preset):
    """One model per setting, all holding the SAME weights.

    Separate instances so a graph captured under one setting cannot serve
    another; identical weights because these are also the sides of the accuracy
    comparison.
    """
    cfg = bench.TransformerConfig(
        batch_size=preset["batch_size"], seq_len=preset["seq_len"],
        d_model=preset["d_model"], num_heads=preset["heads"],
        ffn_dim=preset["ffn_dim"], num_layers=preset["layers"],
        causal=bool(preset.get("causal")))
    base = bench.BaselineTransformer(cfg)
    models = {}
    for name in SETTINGS:
        opt = bench.UserOptimizedTransformer(cfg)
        bench.copy_model_weights(base, opt)
        models[name] = opt.to(DEV).eval()
    x, m = bench.generate_random_case(config=cfg, device=DEV, dtype=torch.float32,
                                      seed=1234, padding_ratio=0.0, input_scale=1.0)
    return models, x, m


def _coerce(setting):
    """Settings arrive from argparse as strings, but numeric knobs are compared
    against integers -- `rows <= config._LINEAR_BIAS_MAX_ROWS` raises TypeError
    against a str. So a digit-only setting becomes an int; everything else stays
    the string the string-valued knobs ("auto", "off", "tf32") expect."""
    return int(setting) if str(setting).lstrip("-").isdigit() else setting


def time_ms(model, x, m, iters, setting):
    setattr(config, KNOB, _coerce(setting))
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
    global KNOB, SETTINGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=0,
                    help="forward passes per timing. 0 (the default) picks it "
                         "per shape from --target-ms, so a 0.13 ms shape and a "
                         "28 ms one get the same wall clock per sample and "
                         "therefore comparable noise")
    ap.add_argument("--target-ms", type=float, default=60.0)
    ap.add_argument("--shapes", default="1-5,7-13",
                    help="grading shape numbers; 6 is excluded by default "
                         "because one timed iteration of it is ~1.8 s")
    ap.add_argument("--knob", default=KNOB,
                    help="the optimized/config.py global to flip. Defaults to "
                         "LINEAR_BIAS; --knob LAYERNORM --settings off,auto "
                         "scores the entry LayerNorm instead")
    ap.add_argument("--settings", default=",".join(SETTINGS),
                    help="comma-separated values for --knob, slowest first. "
                         "The FIRST is the reference arm and the one the "
                         "control times against itself; the LAST is what the "
                         "'total' column reports")
    ap.add_argument("--self-control", action="store_true",
                    help="time 'off' in every slot. The true ratio is then "
                         "1.000x by construction and anything else is this "
                         "harness lying; run it before trusting a small win")
    args = ap.parse_args()

    KNOB = args.knob
    SETTINGS = args.settings.split(",")
    ref = SETTINGS[0]

    wanted = []
    for part in args.shapes.split(","):
        if "-" in part:
            a, b = part.split("-")
            wanted += list(range(int(a), int(b) + 1))
        else:
            wanted.append(int(part))

    if kernel_ext.get_kernels() is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    props = torch.cuda.get_device_properties(DEV)
    mode = "SELF-CONTROL (every arm is 'off')" if args.self_control else "live"
    print(f"{props.name}: {props.multi_processor_count} SMs, "
          f"grading shapes, CUDA_GRAPH=auto -- {mode}\n")
    cols = "".join(f"{n + ' ms':>10}" for n in SETTINGS)
    steps = "".join(f"{SETTINGS[i] + '/' + SETTINGS[i - 1]:>13}"
                    for i in range(1, len(SETTINGS)))
    print(f"  {'shape':<18}{cols}{steps}{'total':>8}{'ctrl':>7}  {'max_abs':>9}")

    step_g, tot_g = [], []
    worst_ctrl = 0.0
    for _, preset in load_shapes(wanted):
        models, x, m = build_models(preset)
        run_as = {name: (ref if args.self_control else name) for name in SETTINGS}

        for name in SETTINGS:
            setattr(config, KNOB, _coerce(run_as[name]))
            with torch.inference_mode():
                for _ in range(5):
                    models[name](x, m)

        # Fixed iters gave the 0.13 ms shapes a per-sample window a few hundred
        # microseconds wide, where one stray context switch is a 10% reading --
        # the self-control run measured +/-9.6% on shape 4 that way. Scaling to
        # a fixed wall clock instead puts every shape at the same noise floor.
        iters = args.iters
        if iters <= 0:
            # `ref`, not a literal "off": the reference arm is SETTINGS[0],
            # which is "off" only for the string-valued knobs. A numeric
            # knob such as _LINEAR_BIAS_MAX_ROWS has no "off" setting and
            # used to KeyError here.
            probe = time_ms(models[ref], x, m, 3, run_as[ref])
            iters = max(3, min(400, int(round(args.target_ms / max(probe, 1e-3)))))

        best = {name: math.inf for name in SETTINGS}
        best_ctrl = math.inf
        for rnd in range(args.rounds):
            t = {name: [] for name in SETTINGS}
            for name in balanced_order(SETTINGS, rnd):
                t[name].append(time_ms(models[name], x, m, iters, run_as[name]))
            for name in SETTINGS:
                best[name] = min([best[name]] + t[name])
            best_ctrl = min(best_ctrl, abs(t[ref][0] / t[ref][1] - 1.0))
        worst_ctrl = max(worst_ctrl, best_ctrl)

        outs = {}
        for name in SETTINGS:
            setattr(config, KNOB, _coerce(run_as[name]))
            with torch.inference_mode():
                outs[name] = models[name](x, m).clone()
        err = (outs[SETTINGS[-1]] - outs[ref]).abs().max().item()

        step = [best[SETTINGS[i - 1]] / best[SETTINGS[i]]
                for i in range(1, len(SETTINGS))]
        total = best[ref] / best[SETTINGS[-1]]
        step_g.append(step)
        tot_g.append(total)
        cols = "".join(f"{best[n]:10.3f}" for n in SETTINGS)
        steps = "".join(f"{v:12.3f}x" for v in step)
        print(f"  {preset['name']:<18}{cols}{steps}{total:7.3f}x "
              f"{best_ctrl * 100:6.1f}% {err:9.2e}")

    setattr(config, KNOB, _coerce(SETTINGS[-1]))

    def gm(v):
        return math.exp(sum(math.log(g) for g in v) / len(v))

    print(f"\n  geometric mean over {len(tot_g)} shapes:")
    for i in range(1, len(SETTINGS)):
        leg = f"{SETTINGS[i - 1]} -> {SETTINGS[i]}"
        print(f"    {leg:<24}: {gm([v[i - 1] for v in step_g]):.3f}x")
    print(f"    {'together (' + ref + ' -> ' + SETTINGS[-1] + ')':<24}: "
          f"{gm(tot_g):.3f}x")
    # Two different noise statements, and the spread is the honest one.
    # `best_ctrl` is the *minimum* disagreement between the reference arm's two
    # slots across rounds, so it is the most optimistic reading available and
    # routinely prints 0.1% on a shape whose total column is 5% off 1.000x.
    # Under --self-control every total is 1.000x by construction, so their
    # spread is what this harness can actually resolve.
    print(f"  worst per-round control: +/-{worst_ctrl * 100:.1f}%")
    print(f"  total column spans {min(tot_g):.3f}x .. {max(tot_g):.3f}x"
          + ("  <-- under --self-control every one of these should be 1.000x; "
             "the spread is the real noise floor" if args.self_control else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
