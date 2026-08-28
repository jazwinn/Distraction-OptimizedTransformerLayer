"""Where fused_add_layernorm leaves the card idle, and how far off roofline.

The kernel is one block per row. After the block size was scaled to the row
width, a narrow row gets a one-warp block -- which fixed the idle-thread waste
but means one *block* per row, and an SM caps the number of resident blocks
regardless of how small they are. This prints the occupancy the driver actually
reports alongside the achieved bandwidth, so the diagnosis is measured rather
than asserted before anything is rewritten.

The floor is analytic: 4N floats (read x, read sub, write x_new, write normed)
at the card's measured copy bandwidth. Not a torch.add row -- torch.add is
dispatch-bound at these sizes and reports a "floor" above the kernel it is
supposed to bound.

    cmd.exe /c scripts\\devenv.bat python scripts\\bench_layernorm_occupancy.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

WIDTHS = [32, 64, 128, 256, 512, 1024, 2048]

# The (rows, D) pairs the harness issues: rows = batch*seq_len, D = d_model.
SHAPES = [
    (1024, 32), (128, 256), (256, 256), (1024, 256), (8192, 256),
    (16384, 256), (1024, 512), (1024, 1024), (1024, 2048),
]

ELEMENTS = 1 << 24   # held constant across D so every width is bandwidth-bound


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


def make(rows, D, dev):
    g = torch.Generator(device="cuda").manual_seed(1234)
    return (torch.randn(rows, D, device=dev, generator=g),
            torch.randn(rows, D, device=dev, generator=g),
            torch.randn(D, device=dev, generator=g),
            torch.randn(D, device=dev, generator=g))


def main() -> int:
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1
    if not hasattr(kernels, "layernorm_blocks_per_sm"):
        print("this build has no layernorm_blocks_per_sm; rebuild")
        return 1

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    sms = props.multi_processor_count
    warps_per_sm = props.max_threads_per_multi_processor // 32
    print(f"{props.name}: {sms} SMs, {warps_per_sm} warps/SM max, "
          f"sm_{props.major}{props.minor}\n")

    big = torch.empty(1 << 25, device=dev)
    dst = torch.empty_like(big)
    copy_us = timed(lambda: dst.copy_(big), iters=30)
    peak = (2 * big.numel() * 4) / (copy_us * 1e-6) / 1e9
    print(f"measured copy bandwidth: {peak:.0f} GB/s\n")

    print("=== occupancy the driver reports, per row width ===")
    print(f"  {'D':>5} {'threads':>8} {'warps/blk':>10} {'blocks/SM':>10} "
          f"{'warps/SM':>9} {'of max':>8}  limiter")
    for D in WIDTHS:
        threads = kernels.layernorm_block_threads(D)
        per_sm = kernels.layernorm_blocks_per_sm(D)
        nwarps = (threads + 31) // 32
        warps = per_sm * nwarps
        # 16 resident blocks is the sm_86 hardware cap; hitting it exactly with
        # room to spare in warps means the block count is what binds.
        limiter = ("blocks/SM cap" if per_sm >= 16 and warps < warps_per_sm
                   else "warps/threads" if warps >= warps_per_sm
                   else "smem or registers")
        print(f"  {D:5d} {threads:8d} {nwarps:10d} {per_sm:10d} {warps:9d} "
              f"{100.0 * warps / warps_per_sm:7.0f}%  {limiter}")

    print(f"\n=== achieved bandwidth, {ELEMENTS/1e6:.0f}M elements at every D ===")
    print(f"  {'rows':>8} {'D':>5} {'us':>9} {'GB/s':>8} {'floor us':>9} "
          f"{'x off':>7}")
    for D in WIDTHS:
        rows = ELEMENTS // D
        x, sub, w, b = make(rows, D, dev)
        us = timed(lambda: kernels.fused_add_layernorm(x, sub, w, b, 1e-5), iters=30)
        traffic = 4 * rows * D * 4
        floor = traffic / (peak * 1e9) * 1e6
        print(f"  {rows:8d} {D:5d} {us:9.2f} {traffic / (us * 1e-6) / 1e9:8.0f} "
              f"{floor:9.2f} {us / floor:6.2f}x")

    print("\n=== the shapes the model issues (eager; dispatch-bound, for scale) ===")
    print(f"  {'rows':>6} {'D':>5} {'us':>9} {'floor us':>9} {'blocks':>8} "
          f"{'card':>7}")
    for rows, D in SHAPES:
        x, sub, w, b = make(rows, D, dev)
        us = timed(lambda: kernels.fused_add_layernorm(x, sub, w, b, 1e-5))
        floor = (4 * rows * D * 4) / (peak * 1e9) * 1e6
        capacity = kernels.layernorm_blocks_per_sm(D) * sms
        print(f"  {rows:6d} {D:5d} {us:9.2f} {floor:9.2f} {rows:8d} "
              f"{100.0 * min(rows, capacity) / capacity:6.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
