"""Where the custom bias GEMM crosses over cuBLAS, as a function of M.

This reproduces the measurement behind row 1b of docs/OPTIMIZATION_LEDGER.md, which
is what set `_LINEAR_BIAS_MAX_ROWS = 262144`. On the QKV shape (K=128, N=384)
the custom kernel read 1.284x at M=32768, 1.145x at 65536, 1.085x at 131072,
**0.959x at 262144**, 0.926x at 524288 and 0.977x at 1280000 -- so above ~262k
rows it lost and the dispatcher hands those shapes to `F.linear`. Shape 6
(batch 10000) is the one grading shape above that gate, and it is therefore the
one shape that gets none of ledger rows 1, 2, 6, 7 or 8.

Row 1a's diagnosis for the loss is the epilogue's scalar stores. So this script
is the instrument for asking whether vectorizing them moved the crossover, and
by how much.

Both arms are timed in one process, interleaved via ab_common.balanced_order,
with the incumbent re-timed alongside the challenger every round -- per the
project's measurement rules. `--self-control` puts cuBLAS in *both* slots, which
must read ~1.000x; anything it does not read is the harness's floor and no
result smaller than that spread is readable.
"""
import argparse
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import kernel_ext  # noqa: E402  (must precede torch)
sys.path.insert(0, "scripts")
from ab_common import balanced_order  # noqa: E402


def time_op(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(kern, M, N, K, rounds, target_ms, self_control, out_half,
          tiles=None):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    def cublas():
        out = F.linear(a, w, b)
        return out.half() if out_half else out

    def custom():
        return kern.linear_bias(a, w, b, -1, -1, out_half)

    if tiles is not None:
        # Compare two of the kernel's own block tiles instead of racing cuBLAS.
        # `tile` is an explicit argument to linear_bias, so this needs no config
        # knob and no rebuild -- both arms are the same .pyd, timed round-robin.
        ref_t, cand_t = tiles

        def arm(t):
            return lambda: kern.linear_bias(a, w, b, t, -1, out_half)

        arms = {f"tile{ref_t}": arm(ref_t),
                f"tile{cand_t}": arm(ref_t) if self_control else arm(cand_t)}
    else:
        arms = {"cublas": cublas, "custom": cublas if self_control else custom}
    ref_name, cand_name = list(arms)

    # Scale iterations to a fixed wall clock, not a fixed count: a standing
    # measurement fact is that a fixed --iters gives the small shapes a
    # sub-millisecond sample window and reads +-9.6% noise.
    probe = time_op(arms[ref_name], 3)
    iters = max(3, min(2000, int(target_ms / max(probe, 1e-4))))

    per_arm = {k: [] for k in arms}
    for rnd in range(rounds):
        for name in balanced_order(list(arms), rnd):
            per_arm[name].append(time_op(arms[name], iters, warmup=1))

    med = {k: statistics.median(v) for k, v in per_arm.items()}
    ratio = med[ref_name] / med[cand_name]
    return med[ref_name], med[cand_name], ratio, iters


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--target-ms", type=float, default=60.0)
    p.add_argument("--self-control", action="store_true",
                   help="put cuBLAS in both slots; must read ~1.000x")
    p.add_argument("--out-half", action="store_true",
                   help="fp16 C, which is what the model asks for above "
                        "_QKV_FP16_MIN_ROWS rows")
    p.add_argument("-N", type=int, default=384, help="QKV projection width")
    p.add_argument("-K", type=int, default=128, help="d_model")
    p.add_argument("--tiles", default=None,
                   help="compare two of the kernel's own block tiles instead of "
                        "cuBLAS, e.g. --tiles 1,0 for 64x64 against 128x128. "
                        "Codes: 0=128x128, 1=64x64, 2=64x32, -1=auto. The first "
                        "is the reference arm")
    p.add_argument("--rows", default="32768,65536,131072,262144,524288,1280000",
                   help="comma-separated M values -- row 1b's own ladder")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1

    # cuBLAS must be given TF32, which is what the harness gives it. Torch
    # defaults matmul allow_tf32 to False, and without this the cuBLAS arm runs
    # a true FP32 SGEMM -- about half the throughput of the TF32 path. The
    # custom kernel then "wins" a flat ~2.2x at every M, which is the tell: row
    # 1b measured a strong M dependence (1.284x down to 0.926x), and a ratio
    # that does not move across a 40x range of M means the two arms are in
    # different arithmetic regimes rather than the same one with different
    # epilogues. Every other ab_*.py in this repo sets these two lines.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    kern = kernel_ext.get_kernels()
    if kern is None:
        print("extension failed to load:", kernel_ext.load_error())
        return 1

    tiles = None
    if args.tiles:
        parts = [int(x) for x in args.tiles.split(",")]
        assert len(parts) == 2, "--tiles takes exactly two codes"
        tiles = tuple(parts)
    if args.self_control:
        mode = "SELF-CONTROL (both arms identical)"
    elif tiles:
        mode = f"tile {tiles[1]} vs tile {tiles[0]}"
    else:
        mode = "custom vs cuBLAS"
    print(f"{torch.cuda.get_device_name(0)}   {mode}")
    print(f"N={args.N} K={args.K} out={'fp16' if args.out_half else 'fp32'} "
          f"rounds={args.rounds} target={args.target_ms}ms\n")
    lhs = f"tile{tiles[0]} ms" if tiles else "cuBLAS ms"
    rhs = f"tile{tiles[1]} ms" if tiles else "custom ms"
    print(f"  {'M':>9} {lhs:>11} {rhs:>11} {'ratio':>8}  {'iters':>6}")

    ratios = []
    for M in (int(x) for x in args.rows.split(",")):
        try:
            cb, cu, r, iters = bench(kern, M, args.N, args.K, args.rounds,
                                     args.target_ms, args.self_control,
                                     args.out_half, tiles)
        except torch.cuda.OutOfMemoryError:
            print(f"  {M:>9}  out of memory")
            torch.cuda.empty_cache()
            continue
        flag = ""
        if not args.self_control:
            win, lose = (rhs, lhs) if r > 1.0 else (lhs, rhs)
            flag = f"  <- {win.replace(' ms', '')} wins"
        print(f"  {M:>9} {cb:>11.4f} {cu:>11.4f} {r:>8.4f}{flag}")
        ratios.append(r)
        torch.cuda.empty_cache()

    if ratios:
        lo, hi = min(ratios), max(ratios)
        print(f"\n  ratio spans {lo:.4f}..{hi:.4f}")
        if args.self_control:
            print("  ^ this spread is the floor; ignore any effect smaller than it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
