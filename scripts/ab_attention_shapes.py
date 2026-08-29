"""Sweep the wmma kernel's block shapes for 16-bit compute types.

WMMA_M_*/WMMA_N_* were tuned when every staged tile was fp32. Narrowing the
compute type to fp16 frees roughly a third of the block's shared memory, so the
shape that was optimal at 4 bytes per element is not necessarily optimal at 2 --
and at head_dim 128 the old shape was not even chosen on merit. It was the only
one that fit.

The budget, per block, at head_dim 128:

    fp32 32x16   29.9 KB    fp16 32x16   29.9 KB   <- today
    fp32 32x32   too big    fp16 32x32   41.4 KB   <- newly affordable
    fp32 64x32   75.8 KB    fp16 64x32   65.8 KB   <- still too big

O is always fp32 and alone wants 33.8 KB at BLOCK_M 64, which is why halving the
operand width does not buy a 64-row block here.

Each candidate is a separate build with -D overrides into its own directory, so
this is the one comparison in the repo that CANNOT be interleaved in a single
process. Everything else follows csrc/TUNING.md: the incumbent is rebuilt and
re-timed alongside every challenger in the same invocation, each shape is timed
best-of-rounds, and an incumbent-vs-incumbent control is printed.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_attention_shapes.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402

from verify_kernel import reference_attention_f64  # noqa: E402

from tune_block_shapes import build  # noqa: E402

DEV = torch.device("cuda")
IMPL_WMMA = 2
BSHD = 1

# head_dim -> [(BLOCK_M, BLOCK_N)], incumbent first. Only the head_dims where a
# narrower element could plausibly change the answer are swept: 8/16 already run
# the widest shape the general case allows.
CANDIDATES = {
    64:  [(64, 16), (64, 32), (32, 32)],
    128: [(32, 16), (32, 32), (16, 32)],
}

# Shapes to score each candidate on, per head_dim. Short and long together, so a
# shape that tanks seq 128 cannot win on the sum of raw milliseconds.
CASES = {
    64:  [(8, 8, 128), (4, 8, 512), (1, 8, 2048)],
    128: [(8, 8, 32), (8, 8, 128), (4, 8, 512), (2, 8, 1024)],
}


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


def score(mod, head_dim, iters):
    """Best time per case for this build, plus its worst error against fp64."""
    out, worst_err = [], 0.0
    for B, H, S in CASES[head_dim]:
        g = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(B, H, S, head_dim, device=DEV, generator=g)
        k = torch.randn(B, H, S, head_dim, device=DEV, generator=g)
        v = torch.randn(B, H, S, head_dim, device=DEV, generator=g)
        scale = head_dim ** -0.5

        def call():
            return mod.fused_attention_forward(q, k, v, None, True, scale,
                                               IMPL_WMMA, BSHD)

        ref = reference_attention_f64(q, k, v, None, True, scale, layout=1)
        try:
            worst_err = max(worst_err, (call().double() - ref).abs().max().item())
        except Exception:
            return None, float("nan")   # shape declined: over budget
        t = graph_timed(call, iters)
        if t is None:
            return None, worst_err
        out.append(t)
    return out, worst_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--head-dims", type=int, nargs="*", default=[128, 64])
    args = ap.parse_args()

    kernel_ext._ensure_msvc_on_path()
    tile_home = kernel_ext._find_tile_cuda_home()
    if tile_home is not None:
        os.environ["CUDA_HOME"] = tile_home
        os.environ["PATH"] = os.path.join(tile_home, "bin") + os.pathsep + os.environ["PATH"]

    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(DEV)
    print(f"{props.name}: {props.multi_processor_count} SMs, causal, "
          f"fp16 compute, one build per candidate\n")

    root = tempfile.mkdtemp(prefix="wmma16_")
    for head_dim in args.head_dims:
        cases = CASES[head_dim]
        print(f"=== head_dim {head_dim} ===")
        print(f"  {'shape':>10} " + " ".join(f"{f'{m}x{n}':>10}"
                                             for m, n in CANDIDATES[head_dim]))
        results = {}
        for i, (m, n) in enumerate(CANDIDATES[head_dim]):
            defines = {f"WMMA16_M_{head_dim}": m, f"WMMA16_N_{head_dim}": n}
            name = f"wmma16_d{head_dim}_{m}x{n}"
            workdir = os.path.join(root, name)
            os.makedirs(workdir, exist_ok=True)
            try:
                mod = build(name, defines, workdir)
            except Exception as exc:
                print(f"  {m}x{n}: build failed: {str(exc)[:80]}")
                results[(m, n)] = (None, float("nan"))
                continue
            results[(m, n)] = score(mod, head_dim, args.iters)

        for j, (B, H, S) in enumerate(cases):
            cells = []
            for m, n in CANDIDATES[head_dim]:
                ts, _ = results[(m, n)]
                cells.append(f"{ts[j]:10.1f}" if ts else f"{'declined':>10}")
            print(f"  B{B} H{H} S{S:<5} " + " ".join(cells))

        base = CANDIDATES[head_dim][0]
        base_ts, base_err = results[base]
        print(f"\n  vs the incumbent {base[0]}x{base[1]}, per shape and geomean:")
        for m, n in CANDIDATES[head_dim]:
            ts, err = results[(m, n)]
            if not ts or not base_ts:
                print(f"    {m:3d}x{n:<3d}  declined")
                continue
            ratios = [b / t for b, t in zip(base_ts, ts)]
            g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            per = " ".join(f"{r:.3f}x" for r in ratios)
            print(f"    {m:3d}x{n:<3d}  {per}   geomean {g:.3f}x   "
                  f"max_abs {err:.2e}")
        print()
    print(f"builds in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
