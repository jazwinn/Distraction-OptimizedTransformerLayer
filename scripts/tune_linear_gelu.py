"""Pick the block tile for gemm_bias_gelu_kernel, and decide which shapes it wins.

The kernel fuses `F.gelu(F.linear(x, W, b))` into one pass. What it has to beat
is cuBLAS's GEMM plus a separate elementwise GELU, so that pair is timed here as
the incumbent and every tile is timed against it in the same process.

Three csrc/TUNING.md rules shape this script, and each of them cost a wrong
conclusion once:

  * **Graph-timed, not eager.** At M=128, K=N=256 an eager `F.linear` measures
    32.3 us of which ~27 is PyTorch dispatch -- the same 32 us the M=1024 call
    measures for eight times the work. The model runs these shapes under a
    captured CUDA graph, where that dispatch is already gone, so an eager
    comparison is measuring a cost the model does not pay. Eager is printed too,
    to keep the size of the gap visible.
  * **Interleaved, best-of-rounds.** Candidates are cycled within each round
    rather than timed one after another, so a drift in clocks hits all of them.
  * **A control row.** cuBLAS timed against itself on the same shapes. Nothing
    smaller than its spread is a result.

Shapes are the FFN's first GEMM: M = batch*seq_len, K = d_model, N = ffn_dim.
The grading appendix has ffn_dim == d_model at every shape; the harness default
is ffn_dim == 4*d_model. Both are swept, because the answer differs -- with a 4x
expansion the GELU pass is a quarter of the pair, and with no expansion it is
about a third, which is what makes the fusion worth wiring in at all.

    cmd.exe /c scripts\\devenv.bat python scripts\\tune_linear_gelu.py
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

# Tile ids and math ids, matching kGemmTile*/kGemmMath* in
# csrc/fused_attention.cu. Tile and precision are independent axes, so the sweep
# is their product -- the best tile is not necessarily the same for both, since
# an fp16 fragment contracts 16 elements of K where tf32 contracts 8.
TILES = [("128x128", 0), ("64x64", 1), ("64x32", 2)]
MATHS = [("tf32", 0), ("fp16", 1)]

# (label, M, K, N). K == N is the appendix's FFN; N == 4*K is the harness default.
SHAPES = [
    ("B1   S128  d256",     128,  256,  256),
    ("B4   S128  d256",     512,  256,  256),
    ("B8   S128  d32",     1024,   32,   32),
    ("B8   S128  d256",    1024,  256,  256),
    ("B16  S128  d256",    2048,  256,  256),
    ("B64  S128  d256",    8192,  256,  256),
    ("B128 S128  d256",   16384,  256,  256),
    ("B8   S1024 d256",    8192,  256,  256),
    ("B10k S32   d256",  320000,  256,  256),
    ("B8   S128  d512",    1024,  512,  512),
    ("B8   S128  d1024",   1024, 1024, 1024),
    ("B4   S128  d2048",    512, 2048, 2048),
    ("B8   S128  d512 ffn2048", 1024, 512, 2048),
    # Extremes, here to bound the gate rather than because the model issues
    # them: the smallest grid a real shape could produce, the narrowest N, and
    # a K past anything in the appendix.
    ("edge M32   d256",       32,  256,  256),
    ("edge M128  d64",       128,   64,   64),
    ("edge M2048 d4096",    2048, 4096, 4096),
]


def graph_timed(fn, iters=50, reps=5, per_graph=20):
    """Kernel time with PyTorch dispatch taken out of the loop.

    per_graph calls go into one graph: a replay costs a few microseconds
    whatever it contains, which at these sizes is comparable to the kernel.
    """
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


def eager_timed(fn, iters=50, reps=5):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1e3)
    return best


def operands(M, K, N, dev):
    g = torch.Generator(device="cuda").manual_seed(1234)
    return (torch.randn(M, K, device=dev, generator=g),
            torch.randn(N, K, device=dev, generator=g) * 0.05,
            torch.randn(N, device=dev, generator=g) * 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    # The harness runs with TF32 on, which is what makes this kernel's tf32
    # fragments comparable to cuBLAS's. Measuring under highest precision would
    # compare a tf32 kernel against an fp32 one and call the difference a win.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"{props.name}: {props.multi_processor_count} SMs\n")

    # Throwaway. The first measurement in a fresh process reads several percent
    # slow, and once turned a 1.10x into a reported 1.32x.
    x, w, b = operands(1024, 256, 256, dev)
    graph_timed(lambda: F.linear(x, w, b), 10, 2, 5)

    print(f"  {'shape':<24} {'M':>7} {'K':>5} {'N':>5} {'blocks':>7} "
          f"{'cuBLAS':>8} {'tf32':>8} {'tile':>8} {'fp16':>8} {'tile':>8} "
          f"{'16v32':>6} {'vs cuBLAS':>7} {'max_abs':>9}")

    per_tile_wins = {name: [] for name, _ in TILES}
    for label, M, Kd, N in SHAPES:
        x, w, b = operands(M, Kd, N, dev)

        # max_abs against cuBLAS, per precision. Reported for fp16 because tf32
        # is bit-identical to cuBLAS wherever cuBLAS picks a tf32 kernel with
        # the same k-order, so its column is 0.00e+00 and says nothing.
        ref = F.gelu(F.linear(x, w, b), approximate="none")
        errs = {}
        for mname, mid in MATHS:
            worst = 0.0
            for _, tid in TILES:
                got = K.linear_gelu(x, w, b, tid, mid)
                worst = max(worst, (got - ref).abs().max().item())
            errs[mname] = worst

        # Interleaved: one round times every candidate, and each candidate keeps
        # its best round. Timing all of cuBLAS then all of the kernel would fold
        # any clock drift between the two halves into the ratio.
        cands = [("cuBLAS", lambda: F.gelu(F.linear(x, w, b), approximate="none"))]
        for mname, mid in MATHS:
            for tname, tid in TILES:
                cands.append((
                    f"{mname}/{tname}",
                    (lambda t, m: lambda: K.linear_gelu(x, w, b, t, m))(tid, mid)))
        best = {name: float("inf") for name, _ in cands}
        for _ in range(args.rounds):
            for name, fn in cands:
                t = graph_timed(fn, args.iters)
                if t is not None:
                    best[name] = min(best[name], t)

        base = best["cuBLAS"]
        # Best tile within each precision, so the two are compared at their own
        # optimum rather than at a tile that happens to suit one of them.
        picks = {}
        for mname, _ in MATHS:
            tname = min((t for t, _ in TILES), key=lambda t: best[f"{mname}/{t}"])
            picks[mname] = (tname, best[f"{mname}/{tname}"])
        t32, f16 = picks["tf32"], picks["fp16"]
        gain = base / f16[1]
        per_tile_wins[f16[0]].append(gain)

        blocks = ((M + 63) // 64) * ((N + 63) // 64)
        print(f"  {label:<24} {M:7d} {Kd:5d} {N:5d} {blocks:7d} "
              f"{base:8.1f} {t32[1]:8.1f} {t32[0]:>8} {f16[1]:8.1f} {f16[0]:>8} "
              f"{t32[1]/f16[1]:6.3f}x {gain:7.3f}x {errs['fp16']:9.2e}")

    print("\n  tile chosen, and its geometric-mean ratio where chosen:")
    for name, gains in per_tile_wins.items():
        if gains:
            gm = math.exp(sum(math.log(g) for g in gains) / len(gains))
            print(f"    {name:>8}: {len(gains):2d} shapes, {gm:.3f}x")

    # Control: cuBLAS against itself, same harness, same shapes. Any tile margin
    # below this is a tie, and the incumbent (cuBLAS) keeps the shape.
    print("\n  control -- cuBLAS timed against itself:")
    worst = 0.0
    for label, M, Kd, N in SHAPES:
        x, w, b = operands(M, Kd, N, dev)
        fn = lambda: F.gelu(F.linear(x, w, b), approximate="none")  # noqa: E731
        a = min(graph_timed(fn, args.iters) for _ in range(args.rounds))
        c = min(graph_timed(fn, args.iters) for _ in range(args.rounds))
        worst = max(worst, abs(a / c - 1.0))
    print(f"    +/-{worst * 100:.1f}% -- nothing below this is a result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
