"""
A/B eager against CUDA-graph replay, interleaved, to find where graphs stop
paying -- which is the number _GRAPH_MAX_ACTIVATION should be set
from -- and how much memory each capture pins in exchange.

Methodology, which matters more than the numbers: run-to-run variance on this
card is +/-10-15%, larger than several of the effects being measured, and a
cross-run comparison has already inverted a ranking in this project once. So
both variants are timed in one process, round-robin, several rounds, and each
keeps its *best* round. Every group also carries a **control** row -- eager
timed against eager -- which is identical code on both sides and therefore a
direct reading of this harness's own noise. No result smaller than the control's
spread means anything, and since one of the headline numbers here is a 1.03x,
having the noise floor printed beside it is not optional.

Four axes, because "how big is the input" has more than one answer and the
obvious one turned out to be wrong:

  * tokens (batch*seq), densely across the crossover.
  * depth. Launches scale with layers (~79 kernels at 6, ~157 at 12) while the
    token count does not move. Measured: 1.031x / 1.037x / 1.040x / 1.040x at 3,
    6, 12 and 24 layers -- eager and replay scale together, so depth does not
    move the ratio and does not belong in the gate.
  * width. This is the one that mattered: at 512 tokens, d_model 256 gave 2.708x
    and d_model 512 gave 1.036x. Same tokens, 2.6x difference in payoff, which
    kills the token count as a gate on its own.
  * the product, tokens*d_model, held constant in pairs to test whether it is the
    real predictor. It is, near the boundary: at 524288 two different shapes came
    out 1.038x and 1.030x, inside the noise floor of each other. Further down it
    is looser (1.02x-1.32x at 262144), but down there everything is a win anyway
    and the gate does not have to choose.

Reserved-memory delta is reported next to the speedup rather than in a separate
table, because the decision needs both. A 1.03x that pins 84 MiB is a different
call from a 4.2x that pins 22 MiB.

If you just want the number for your machine, skip the tables:

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_graph.py --recommend

which sweeps activation volume, works out the noise floor from its own control
rows, and prints the _GRAPH_MAX_ACTIVATION to set -- or refuses to answer and
tells you to close whatever is making the machine noisy.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_graph.py
    cmd.exe /c scripts\\devenv.bat python scripts\\ab_graph.py --rounds 7
    cmd.exe /c scripts\\devenv.bat python scripts\\ab_graph.py --axis depth
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import config  # noqa: E402

# The runtime knobs live in optimized/config.py, and the model reads them
# from there on every call. Setting them on the harness module instead
# would create a dead attribute that nothing reads -- which is worse than
# an error here, because both sides of every comparison below would then
# be the *same* path and every delta would come out zero.

DEV = torch.device("cuda")

# (batch, seq, d_model, heads, ffn, layers)
TOKEN_AXIS = [
    (1,   32, 512,  8, 2048, 6),
    (1,  128, 512,  8, 2048, 6),
    (8,   32, 512,  8, 2048, 6),
    (2,  128, 512,  8, 2048, 6),
    (8,  128, 512,  8, 2048, 6),     # the harness default: 1024 tokens
    (8,  256, 512,  8, 2048, 6),
    (4,  512, 512,  8, 2048, 6),
    (8,  512, 512,  8, 2048, 6),
    (2, 1024, 512,  8, 2048, 6),
    (8, 1024, 512,  8, 2048, 6),
    (1, 2048, 512,  8, 2048, 6),
]

DEPTH_AXIS = [
    (4, 128, 512, 8, 2048, 3),
    (4, 128, 512, 8, 2048, 6),
    (4, 128, 512, 8, 2048, 12),
    (4, 128, 512, 8, 2048, 24),
]

WIDTH_AXIS = [
    (4, 128,  256,  4, 1024, 6),
    (4, 128,  512,  8, 2048, 6),
    (4, 128, 1024, 16, 4096, 6),
]

# Does tokens*d_model predict the payoff better than tokens alone? The width
# axis says it might: d_model 256 and d_model 512 at the same 512 tokens came
# out 2.708x and 1.036x. These pairs hold tokens*d_model constant while moving
# tokens and d_model in opposite directions, so if the product is what matters
# each pair should land on the same ratio.
PRODUCT_AXIS = [
    (2,  128,  256,  4, 1024, 6),   # 256 tok  x d256  =  65536
    (1,  128,  512,  8, 2048, 6),   # 128 tok  x d512  =  65536
    (8,  128,  256,  4, 1024, 6),   # 1024 tok x d256  = 262144
    (4,  128,  512,  8, 2048, 6),   # 512 tok  x d512  = 262144
    (1,  256, 1024, 16, 4096, 6),   # 256 tok  x d1024 = 262144
    (8,  256,  256,  4, 1024, 6),   # 2048 tok x d256  = 524288
    (8,  128,  512,  8, 2048, 6),   # 1024 tok x d512  = 524288
]

# The sweep behind --recommend: activation volume (batch*seq*d_model) in powers of
# two, two shapes at each level with different d_model.
#
# Two per level on purpose. The gate is on the product, but the product is not a
# perfect predictor -- at 262144 this machine measured 1.02x and 1.32x for
# different (tokens, d_model) splits of the same volume. Sampling two and taking
# the *worse* one makes the recommendation conservative in the direction that
# matters: it errs toward capturing less, which costs a little latency, rather
# than toward pinning memory on shapes that will not benefit.
#
# (batch, seq, d_model, heads, ffn, layers)
GATE_SWEEP = [
    (16384,   [(1,   32, 512,  8, 2048, 6), (1,   64, 256,  4, 1024, 6)]),
    (32768,   [(2,   32, 512,  8, 2048, 6), (1,  128, 256,  4, 1024, 6)]),
    (65536,   [(1,  128, 512,  8, 2048, 6), (2,  128, 256,  4, 1024, 6)]),
    (131072,  [(2,  128, 512,  8, 2048, 6), (4,  128, 256,  4, 1024, 6)]),
    (262144,  [(4,  128, 512,  8, 2048, 6), (8,  128, 256,  4, 1024, 6)]),
    (524288,  [(8,  128, 512,  8, 2048, 6), (8,  256, 256,  4, 1024, 6)]),
    (1048576, [(4,  512, 512,  8, 2048, 6), (8,  512, 256,  4, 1024, 6)]),
    (2097152, [(8,  512, 512,  8, 2048, 6), (8, 1024, 256,  4, 1024, 6)]),
]

# A control row is eager timed against eager: identical code both sides, so its
# true value is exactly 1.000x and whatever it actually reads is this machine's
# noise. Below this much deviation the machine is quiet enough to trust; above
# it, --recommend refuses to answer rather than reporting a number derived from
# noise. Calibrated from observation: idle here reads 0.96-1.02x, and with a game
# running it read 0.849-1.340x.
NOISE_REFUSE_ABOVE = 0.06

# Never claim a gain smaller than this even on a perfectly quiet machine. Guards
# against a run where the control rows happen to land near 1.000x by luck.
MIN_CREDIBLE_GAIN = 0.015


# Under this, a single forward is short enough that CUDA event resolution and
# the timing loop itself contribute meaningfully, and the ratio stops being a
# measurement. Same reasoning as ab_split_kv.py's floor, scaled to whole-model
# latencies rather than single-kernel ones.
RESOLUTION_MS = 0.05

ROW = "{:<22} {:>7} {:>10} {:>10} {:>8} {:>9} {:>8}"


def build(b, s, d, h, ffn, layers):
    cfg = bench.TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                  ffn_dim=ffn, num_layers=layers, causal=False)
    base = bench.BaselineTransformer(cfg)
    opt = bench.UserOptimizedTransformer(cfg)
    bench.copy_model_weights(base, opt)
    x, m = bench.generate_random_case(config=cfg, device=DEV, dtype=torch.float32,
                                      seed=1234, padding_ratio=0.0, input_scale=1.0)
    return opt.to(DEV).eval(), x, m


def time_ms(model, x, m, iters):
    """Mean over iters, timed with one CUDA event pair around the whole loop."""
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


def measure(shape, rounds, iters):
    """Best-of-rounds for eager, graph, and an eager-vs-eager control."""
    b, s, d, h, ffn, layers = shape

    # Two separate model instances so neither can be perturbed by the other's
    # captured state, and so the eager side is guaranteed never to hold a graph.
    eager_model, x, m = build(*shape)
    graph_model, _, _ = build(*shape)

    config.CUDA_GRAPH = "off"
    with torch.inference_mode():
        for _ in range(5):
            eager_model(x, m)

    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved(DEV)
    config.CUDA_GRAPH = "always"
    with torch.inference_mode():
        for _ in range(5):
            graph_model(x, m)
    pool_mib = (torch.cuda.memory_reserved(DEV) - reserved_before) / (1 << 20)
    captured = len(graph_model._graphs) > 0

    best_eager = best_graph = math.inf
    best_ctrl_a = best_ctrl_b = math.inf
    for _ in range(rounds):
        # Round-robin, so a thermal or clock drift over the run hits both sides
        # equally rather than whichever went first. The second eager timing is
        # the control: identical code to the first, so its ratio against the
        # first is this machine's noise floor and nothing else.
        config.CUDA_GRAPH = "off"
        e1 = time_ms(eager_model, x, m, iters)
        config.CUDA_GRAPH = "always"
        gt = time_ms(graph_model, x, m, iters)
        config.CUDA_GRAPH = "off"
        e2 = time_ms(eager_model, x, m, iters)

        best_eager = min(best_eager, e1)
        best_graph = min(best_graph, gt)
        best_ctrl_a = min(best_ctrl_a, e1)
        best_ctrl_b = min(best_ctrl_b, e2)

    for r in graph_model._graphs.values():
        r.release()
    del eager_model, graph_model, x, m
    torch.cuda.empty_cache()

    return {
        "tokens": b * s,
        "eager": best_eager,
        "graph": best_graph,
        "ratio": best_eager / best_graph,
        "control": best_ctrl_a / best_ctrl_b,
        "pool": pool_mib,
        "captured": captured,
    }


def run_axis(name, shapes, rounds, iters, label):
    print(f"\n=== {name} ===\n")
    print(ROW.format("case", "tokens", "eager ms", "graph ms", "ratio",
                     "pool MiB", "control"))
    print("-" * 82)
    ratios = []
    for shape in shapes:
        r = measure(shape, rounds, iters)
        tag = label(shape)
        flag = ""
        if not r["captured"]:
            flag = "  (NOT CAPTURED)"
        elif r["graph"] < RESOLUTION_MS or r["eager"] < RESOLUTION_MS:
            flag = "  (below resolution)"
        print(ROW.format(tag, r["tokens"], f"{r['eager']:.3f}", f"{r['graph']:.3f}",
                         f"{r['ratio']:.3f}x", f"{r['pool']:.1f}",
                         f"{r['control']:.3f}x") + flag)
        if r["captured"] and r["graph"] >= RESOLUTION_MS:
            ratios.append(r["ratio"])
    if ratios:
        geo = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        print(f"{'':<22} {'':>7} {'':>10} {'':>10} {geo:>7.3f}x   geometric mean")


def recommend(rounds, iters):
    """Measure the crossover on THIS machine and print the value to paste."""
    print("\n=== finding _GRAPH_MAX_ACTIVATION for this machine ===\n")
    print("Two shapes per activation volume, different d_model, worse of the two "
          "taken.\n'control' is eager timed against eager, so it should read "
          "1.000x; how far it\nstrays is this machine's noise floor.\n")
    print("{:>10}  {:>18}  {:>9}  {:>9}  {:>9}".format(
        "activation", "shapes", "worst", "best", "control"))
    print("-" * 64)

    results = []
    worst_control = 0.0
    for volume, shapes in GATE_SWEEP:
        ratios, controls, labels = [], [], []
        for shape in shapes:
            r = measure(shape, rounds, iters)
            ratios.append(r["ratio"])
            controls.append(r["control"])
            labels.append(f"b{shape[0]}s{shape[1]}d{shape[2]}")
        ctrl_dev = max(abs(c - 1.0) for c in controls)
        worst_control = max(worst_control, ctrl_dev)
        results.append((volume, min(ratios), max(ratios), ctrl_dev))
        print("{:>10}  {:>18}  {:>8.3f}x  {:>8.3f}x  {:>8.3f}x".format(
            volume, labels[0] + " +1", min(ratios), max(ratios),
            1.0 + ctrl_dev))

    print("-" * 64)
    print(f"\nnoise floor from the control rows: +/-{worst_control:.1%}")

    if worst_control > NOISE_REFUSE_ABOVE:
        print(f"\n*** REFUSING TO RECOMMEND: this machine is too noisy right now.")
        print(f"    A control row should read 1.000x, and the worst here was off")
        print(f"    by {worst_control:.1%}, which is larger than several of the")
        print(f"    effects being measured. Close whatever else is using the GPU")
        print(f"    or CPU -- a game, a video call, a browser doing something --")
        print(f"    and run this again. Nothing else about the harness is")
        print(f"    affected; only this measurement is.")
        return 1

    # A gain only counts if it clears the noise floor. Take the largest volume
    # that clears it, and require every smaller volume to clear it too -- the
    # payoff should fall monotonically with size, and if it does not, the data is
    # not clean enough to draw a line through.
    threshold = 1.0 + max(worst_control, MIN_CREDIBLE_GAIN)
    print(f"a gain counts as real above {threshold:.3f}x")

    winners = [v for v, lo, _, _ in results if lo >= threshold]
    if not winners:
        print("\nNo activation volume showed a gain clear of the noise floor.")
        print("On this machine CUDA graphs are not worth capturing at any size")
        print("tested; set CUDA_GRAPH = \"off\" and skip the feature.")
        return 0

    gate = max(winners)
    smaller = [v for v, lo, _, _ in results if v <= gate and lo < threshold]

    print(f"\n  set _GRAPH_MAX_ACTIVATION = {gate}"
          f"{'    # ' + hex(gate) if gate & (gate - 1) else '    # 1 << ' + str(gate.bit_length() - 1)}")
    print(f"\nin optimized/config.py. That is the largest activation")
    print(f"volume whose worse shape still beat eager by more than the noise")
    print(f"floor. Above it, replay measured nothing worth the pinned memory.")

    # How much daylight was there between the winning row and the threshold? A
    # thin margin is what produces a one-bucket disagreement between runs, and
    # the reader should know before pasting the number into the source.
    margin = next(lo for v, lo, _, _ in results if v == gate) - threshold
    if margin < 0.01:
        print(f"\nThat was a close call: the winning row cleared the threshold by")
        print(f"only {margin:.3f}x, so a repeat run could pick the bucket either")
        print(f"side of it. Re-run with --rounds 9 to settle it, or just take the")
        print(f"lower value -- capturing slightly less costs a little latency and")
        print(f"nothing else.")

    if smaller:
        print(f"\nNote: {smaller} did NOT clear the threshold despite being smaller,")
        print(f"so the payoff is not falling cleanly with size on this run. That")
        print(f"usually means the machine was busier for part of it -- worth a")
        print(f"second run before trusting the number above.")

    if gate == config._GRAPH_MAX_ACTIVATION:
        print(f"\nThis matches the value already set. Nothing to change.")
    else:
        cur = config._GRAPH_MAX_ACTIVATION
        direction = "larger" if gate > cur else "smaller"
        print(f"\nCurrently set to {cur}, so this machine wants a {direction} gate.")
        print(f"Leaving it as-is is safe either way -- too low costs some latency,")
        print(f"too high costs some pinned memory, and replay is bit-identical to")
        print(f"eager at any setting, so neither can produce a wrong answer.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    # Default deferred so --recommend can ask for more rounds than the browsing
    # tables need. It has to: at 2 rounds --recommend returned 262144 and at 5 it
    # returned 524288 on the same machine, one bucket apart. A tuning run happens
    # once and its answer gets written into the source, so it is the wrong place
    # to economise on samples.
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--axis",
                    choices=("all", "tokens", "depth", "width", "product"),
                    default="all")
    ap.add_argument("--recommend", action="store_true",
                    help="measure the crossover on this machine and print the "
                         "_GRAPH_MAX_ACTIVATION value to use. Refuses to answer "
                         "if the control rows say the machine is too noisy.")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rounds = args.rounds if args.rounds is not None else (5 if args.recommend else 3)

    if args.recommend:
        # --recommend prints its own header and does the interpreting itself, so
        # the "read the crossover off the table" preamble below would only be
        # telling the reader to do a job the script is about to do for them.
        return recommend(rounds, args.iters)

    print(f"rounds={rounds} iters={args.iters}  "
          f"_GRAPH_MAX_ACTIVATION={config._GRAPH_MAX_ACTIVATION} -- not applied "
          f"here: this script forces capture with CUDA_GRAPH='always' so the "
          f"shapes auto declines can still be measured. To have the crossover "
          f"read off for you and turned into a value to set, use --recommend.")
    print("ratio > 1 means the graph is faster. 'control' is eager vs eager, so "
          "it should read 1.000x;\nhow far it strays is this machine's noise "
          "floor, and nothing closer to 1 than that is a result.")

    if args.axis in ("all", "tokens"):
        run_axis("tokens (batch x seq)", TOKEN_AXIS, rounds, args.iters,
                 lambda sh: f"b{sh[0]} s{sh[1]}")
    if args.axis in ("all", "depth"):
        run_axis("depth (layers, tokens fixed at 512)", DEPTH_AXIS, rounds,
                 args.iters, lambda sh: f"{sh[5]} layers")
    if args.axis in ("all", "width"):
        run_axis("width (d_model, tokens fixed at 512)", WIDTH_AXIS, rounds,
                 args.iters, lambda sh: f"d_model {sh[2]}")
    if args.axis in ("all", "product"):
        run_axis("tokens x d_model held constant in pairs", PRODUCT_AXIS,
                 rounds, args.iters,
                 lambda sh: f"{sh[0]*sh[1]}tok d{sh[2]} = {sh[0]*sh[1]*sh[2]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
