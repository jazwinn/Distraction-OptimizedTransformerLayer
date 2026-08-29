"""A/B the warp-per-row add+LayerNorm against the block-per-row one.

Follows csrc/TUNING.md's two rules: both kernels are timed inside one process,
interleaved, via layernorm_set_warp_width(). Every row in the sweeps below is a
genuine warp-vs-block comparison -- the sweep forces width 0 and 256 regardless
of the default -- so none of them is a control. The control is measured
separately at the bottom, block kernel against itself on the same shapes.

Three sweeps, because three separate questions have to be answered before a
default can be set:

  1. does it win in the bandwidth regime, and by how much (the roofline claim)
  2. how many rows per block -- occupancy against the sm_86 16-blocks/SM cap,
     which is the reading that motivated the kernel and turned out not to be
     what it wins on; the sweep is flat, and rows/block=1 (identical occupancy
     to the block kernel) already wins
  3. how wide should the threshold be. The first guess was 32, from occupancy:
     only D=32 is starved and only D=32 was off the roofline. Both true, both
     the wrong lever -- what the kernel removes is shared-memory staging and
     barriers, a latency win that pays at any moderate row count at any width.
     D=64 and D=256 measure 2.2x.

Timed both eager and under CUDA graph replay, with 20 calls captured per graph.
Eager includes PyTorch dispatch, which at these sizes can be several times the
kernel; the model runs the shapes in question under a graph, so replay is what
it actually gets.

Run a throwaway shape first if you add one: the first measurement in a fresh
process reads several percent slow and once turned a 1.10x into a reported
1.32x.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_layernorm_warp.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_common import balanced_order  # noqa: E402

import kernel_ext  # noqa: E402

import torch  # noqa: E402

WIDTHS = [32, 64, 128, 256, 512]
ROWS_CANDIDATES = [1, 2, 4, 8, 16]
ELEMENTS = 1 << 24

# (rows, D) the harness issues: rows = batch*seq_len, D = d_model.
SHAPES = [
    (1024, 32), (256, 32), (4096, 32), (16384, 32),
    (1024, 64), (1024, 256), (8192, 256), (1024, 512),
]


def timed(fn, iters=50, reps=7):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end) / iters * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def graph_timed(fn, iters=50, reps=7, per_graph=20):
    """Kernel time with PyTorch dispatch taken out of the loop.

    per_graph calls are captured INTO one graph rather than one. Replaying a
    graph costs several microseconds whatever it contains, and at these sizes
    that is larger than the kernel: a first pass captured one call each and
    reported 8.02 vs 8.95 us at 1024x32 -- a 0.90x "regression" that was almost
    entirely the fixed replay cost, since 256x32 (a quarter of the work) came
    out at 7.99 vs 8.84. Amortising over 20 calls puts the kernel back in view.
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
    ts = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end) / iters * 1e3 / per_graph)
    ts.sort()
    return ts[len(ts) // 2]


def make(rows, D, dev):
    g = torch.Generator(device="cuda").manual_seed(1234)
    return (torch.randn(rows, D, device=dev, generator=g),
            torch.randn(rows, D, device=dev, generator=g),
            torch.randn(D, device=dev, generator=g),
            torch.randn(D, device=dev, generator=g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    K = kernel_ext.get_kernels()
    if K is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1
    if not hasattr(K, "layernorm_set_warp_width"):
        print("this build predates the warp-per-row kernel; rebuild")
        return 1

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"{props.name}: {props.multi_processor_count} SMs, "
          f"{props.max_threads_per_multi_processor // 32} warps/SM max\n")

    big = torch.empty(1 << 25, device=dev)
    dst = torch.empty_like(big)
    peak = (2 * big.numel() * 4) / (timed(lambda: dst.copy_(big), 30) * 1e-6) / 1e9
    print(f"measured copy bandwidth: {peak:.0f} GB/s\n")

    def both(x, sub, w, b, width):
        K.layernorm_set_warp_width(width)
        r = K.fused_add_layernorm(x, sub, w, b, 1e-5)
        return r

    # ---- 1. bandwidth regime -------------------------------------------------
    print(f"=== bandwidth regime, {ELEMENTS/1e6:.0f}M elements at every D ===")
    print(f"  {'rows':>8} {'D':>5} {'floor':>8} {'block':>9} {'warp':>9} "
          f"{'ratio':>7} {'blk GB/s':>9} {'wrp GB/s':>9}")
    for D in WIDTHS:
        rows = ELEMENTS // D
        x, sub, w, b = make(rows, D, dev)
        traffic = 4 * rows * D * 4
        floor = traffic / (peak * 1e9) * 1e6
        blk = timed(lambda: both(x, sub, w, b, 0), args.iters)
        wrp = timed(lambda: both(x, sub, w, b, 256), args.iters)
        K.layernorm_set_warp_width(-1)
        print(f"  {rows:8d} {D:5d} {floor:8.1f} {blk:9.1f} {wrp:9.1f} "
              f"{blk / wrp:6.3f}x {traffic/(blk*1e-6)/1e9:9.0f} "
              f"{traffic/(wrp*1e-6)/1e9:9.0f}")

    # ---- 2. rows per block ---------------------------------------------------
    print("\n=== rows per block, warp kernel, D=32 ===")
    print(f"  {'rows/blk':>9} {'warps/SM':>9} " +
          " ".join(f"{'r'+str(r):>9}" for r in (1024, 16384, 524288)))
    K.layernorm_set_warp_width(256)
    for rpb in ROWS_CANDIDATES:
        K.layernorm_set_warp_rows(rpb)
        cells = []
        for nrows in (1024, 16384, 524288):
            x, sub, w, b = make(nrows, 32, dev)
            cells.append(timed(lambda: K.fused_add_layernorm(x, sub, w, b, 1e-5),
                               args.iters))
        # 16 blocks/SM is the sm_86 cap; ROWS warps per block against 48.
        print(f"  {rpb:9d} {min(16 * rpb, 48):9d} " +
              " ".join(f"{c:9.2f}" for c in cells))
    K.layernorm_set_warp_rows(0)
    K.layernorm_set_warp_width(-1)

    # ---- 3. the shapes the model issues, eager and in-graph -------------------
    print("\n=== shapes the model issues ===")
    print(f"  {'rows':>6} {'D':>5} {'eagerB':>8} {'eagerW':>8} {'eager':>7} "
          f"{'graphB':>8} {'graphW':>8} {'graph':>7}  default")
    rows_out = []
    for nrows, D in SHAPES:
        x, sub, w, b = make(nrows, D, dev)
        eb = timed(lambda: both(x, sub, w, b, 0), args.iters)
        ew = timed(lambda: both(x, sub, w, b, 256), args.iters)
        # Both widths timed twice, in both orders: block-first only would
        # give the warp kernel the later and faster slot every time. See
        # ab_common.balanced_order.
        g = {0: [], 256: []}
        for width in balanced_order((0, 256), 0):
            K.layernorm_set_warp_width(width)
            g[width].append(
                graph_timed(lambda: K.fused_add_layernorm(x, sub, w, b, 1e-5),
                            args.iters))
        K.layernorm_set_warp_width(-1)
        gb = min(t for t in g[0] if t) if all(g[0]) else None
        gw = min(t for t in g[256] if t) if all(g[256]) else None
        used = "warp" if D <= K.layernorm_warp_width() else "block"
        ratio = f"{gb / gw:.3f}x" if gb and gw else "-"
        rows_out.append((nrows, D, eb, ew, gb, gw, used))
        print(f"  {nrows:6d} {D:5d} {eb:8.2f} {ew:8.2f} {eb/ew:6.3f}x "
              f"{gb:8.2f} {gw:8.2f} {ratio:>7}  {used}")

    print("\n  geometric mean over shapes the default routes to the warp "
          "kernel:")
    sel = [r for r in rows_out if r[6] == "warp" and r[4] and r[5]]
    if sel:
        print(f"    eager {math.exp(sum(math.log(r[2]/r[3]) for r in sel)/len(sel)):.3f}x"
              f" / graph {math.exp(sum(math.log(r[4]/r[5]) for r in sel)/len(sel)):.3f}x"
              f"  ({len(sel)} rows)")
    # A real control: the block kernel timed against itself, same harness, same
    # shapes. Every row above is a genuine warp-vs-block comparison -- the sweep
    # forces width=0 and width=256 regardless of the default -- so none of them
    # is a control, and an earlier version of this script mislabelled the D>=64
    # rows as one and reported a meaningless +/-54%.
    print("\n  control -- block kernel timed against itself, same shapes:")
    worst = 0.0
    for nrows, D in SHAPES:
        x, sub, w, b = make(nrows, D, dev)
        K.layernorm_set_warp_width(0)
        a = graph_timed(lambda: K.fused_add_layernorm(x, sub, w, b, 1e-5),
                        args.iters)
        c = graph_timed(lambda: K.fused_add_layernorm(x, sub, w, b, 1e-5),
                        args.iters)
        K.layernorm_set_warp_width(-1)
        if a and c:
            worst = max(worst, abs(a / c - 1.0))
    print(f"    +/-{worst*100:.1f}% -- nothing below this is a result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
