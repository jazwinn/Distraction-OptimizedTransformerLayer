"""Cost of reading strided q/k/v, isolated from everything else.

Times fused_attention_forward on contiguous inputs against the same values in
the packed [B,S,3*d_model] layout the model produces. The two are alternated
call by call, so drift in clocks, contention or thermals lands on both equally.
The control column is a second contiguous buffer timed the same way: whatever
it deviates from 1.000 is the noise floor, and a ratio inside that is nothing.

This does NOT measure the win -- that is the 18 clone kernels that no longer
run. It measures the price paid for it.
"""
import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Above torch on purpose: importing kernel_ext preloads the driver's GPU
# compiler, which stops a cuTile run from exiting 0xC0000005 only if it happens
# before torch pulls the NVIDIA DLLs in. See kernel_ext.preload_tile_compiler().
import kernel_ext
import torch

K = kernel_ext.get_kernels(verbose=False)


def packed_views(q, k, v):
    b, h, s, d = q.shape
    p = torch.empty(b, s, 3, h, d, device=q.device, dtype=q.dtype)
    for i, t in enumerate((q, k, v)):
        p[:, :, i] = t.permute(0, 2, 1, 3)
    return p.permute(2, 0, 3, 1, 4).unbind(0)


def ab(fns, iters=200):
    for f in fns:
        for _ in range(20):
            f()
    torch.cuda.synchronize()
    ev = [[(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
          for _ in fns]
    for i in range(iters):
        for j, f in enumerate(fns):
            a, b = ev[j][i]
            a.record()
            f()
            b.record()
    torch.cuda.synchronize()
    return [statistics.median(a.elapsed_time(b) for a, b in col) for col in ev]


# "uncovered hd" is head_dim 96, which no kernel specialises, so auto lands on
# the ATen fallback -- the one consumer that pays for a row pitch it did not
# ask for. Without it this table only ever measured paths that read strides
# natively and reported the fallback as free. It is not: ATen measured 1.447x
# on strided input here before the fallback was given contiguous tensors.
CASES = [("default", 8, 8, 128, 64), ("long seq", 1, 8, 2048, 64),
         ("small", 1, 8, 32, 64), ("wide hd", 2, 4, 64, 128),
         ("uncovered hd", 8, 8, 128, 96)]
IMPLS = [(0, "auto"), (1, "scalar"), (2, "wmma"), (3, "tile"), (5, "tile-tf32")]

print(f"{'case':<14}{'impl':>10}{'contig_ms':>11}{'packed_ms':>11}"
      f"{'ratio':>8}{'control':>9}")
print("-" * 63)
dev = torch.device("cuda")
for label, b, h, s, d in CASES:
    g = torch.Generator(device=dev).manual_seed(0)
    q, k, v = (torch.randn(b, h, s, d, generator=g, device=dev) for _ in range(3))
    qp, kp, vp = packed_views(q, k, v)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    scale = d ** -0.5
    with torch.inference_mode():
        for impl, name in IMPLS:
            try:
                K.fused_attention_forward(q, k, v, None, False, scale, impl)
            except RuntimeError:
                print(f"{label:<14}{name:>10}{'n/a':>11}")
                continue

            def call(a, bb, c, impl=impl):
                return lambda: K.fused_attention_forward(
                    a, bb, c, None, False, scale, impl)

            tc, tp, tc2 = ab([call(q, k, v), call(qp, kp, vp), call(q2, k2, v2)])
            print(f"{label:<14}{name:>10}{tc:>11.4f}{tp:>11.4f}"
                  f"{tp / tc:>8.3f}{tc2 / tc:>9.3f}")
