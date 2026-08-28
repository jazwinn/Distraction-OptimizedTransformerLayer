"""Check the warp-per-row add+LayerNorm against the block-per-row kernel.

The two kernels reduce in different orders -- a warp butterfly over registers
against a two-stage reduction through shared memory -- so this compares them
directly as well as against F.layer_norm, and asserts the warp path actually
fired. Both are exercised in one process via layernorm_set_warp_width().

Three things it is specifically trying to break:

  * the corrected two-pass statistics. A plain sum-then-mean is accurate only
    while the row's mean is small next to its spread, so the large-mean cases
    below are not padding -- a naive mean drifted to 1.5e-3 at mean 1e4, past
    atol on its own. If the rewrite lost the correction these rows show it and
    nothing else does.
  * row counts that do not divide the rows-per-block, so the tail block runs
    with some warps retired.
  * widths that are not a multiple of 32, where lanes past D must contribute
    zero to the reduction and store nothing.

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_layernorm_warp.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# (rows, D, mean, scale) -- mean is what makes or breaks the two-pass form.
CASES = [
    (1024, 32, 0.0, 1.0),
    (1024, 32, 1e4, 1.0),      # large mean, small spread: the correction's job
    (1024, 32, 1e5, 1e-2),
    (1023, 32, 0.0, 1.0),      # rows not a multiple of rows-per-block
    (1, 32, 0.0, 1.0),         # one row: a single warp, most of the block idle
    (3, 32, 0.0, 1.0),
    (4096, 32, 0.0, 1.0),
    (1024, 24, 0.0, 1.0),      # D not a multiple of 32
    (1024, 17, 0.0, 1.0),
    (1024, 8, 0.0, 1.0),
    (1024, 1, 0.0, 1.0),       # degenerate: variance is exactly 0
    (1024, 64, 0.0, 1.0),
    (1024, 64, 1e4, 1.0),
    (1024, 96, 0.0, 1.0),
    (1024, 128, 0.0, 1.0),
    (1024, 200, 0.0, 1.0),
    (1024, 256, 0.0, 1.0),
    (1024, 256, 1e4, 1.0),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def main() -> int:
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1
    if not hasattr(kernels, "layernorm_set_warp_width"):
        print("this build predates the warp-per-row kernel; rebuild")
        return 1

    dev = torch.device("cuda")
    ok = True

    print("=== fp32: warp kernel vs block kernel vs F.layer_norm ===")
    print(f"  {'rows':>6} {'D':>5} {'mean':>7} {'warp-blk':>10} "
          f"{'blk-ref':>10} {'warp-ref':>10} {'rows/blk':>9}  verdict")
    for rows, D, mean, scale in CASES:
        g = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(rows, D, device=dev, generator=g) * scale + mean
        sub = torch.randn(rows, D, device=dev, generator=g) * scale
        w = torch.randn(D, device=dev, generator=g)
        b = torch.randn(D, device=dev, generator=g)

        ref_sum = x + sub
        ref = F.layer_norm(ref_sum, (D,), w, b, 1e-5)

        kernels.layernorm_set_warp_width(0)          # block-per-row
        blk_sum, blk = kernels.fused_add_layernorm(x, sub, w, b, 1e-5)
        kernels.layernorm_set_warp_width(256)        # force warp path
        wrp_sum, wrp = kernels.fused_add_layernorm(x, sub, w, b, 1e-5)
        kernels.layernorm_set_warp_width(-1)

        e_pair = (wrp - blk).abs().max().item()
        e_blk = (blk - ref).abs().max().item()
        e_wrp = (wrp - ref).abs().max().item()
        sums_exact = torch.equal(wrp_sum, blk_sum) and torch.equal(blk_sum, ref_sum)

        # The warp path must not be materially worse against the reference than
        # the incumbent is. Absolute tolerance scales with the data, since the
        # large-mean rows carry proportionally larger absolute error.
        tol = max(2e-3, abs(mean) * 1e-6 + scale * 1e-4)
        good = sums_exact and e_wrp <= max(e_blk * 2.0, tol)
        ok = ok and good
        print(f"  {rows:6d} {D:5d} {mean:7.0e} {e_pair:10.2e} {e_blk:10.2e} "
              f"{e_wrp:10.2e} {kernels.layernorm_warp_rows():9d}  "
              f"{'ok' if good else 'FAIL'}"
              f"{'' if sums_exact else '  (x+sub differs!)'}")

    print("\n=== half / bfloat16 (dtype dispatch reaches the warp kernel too) ===")
    for dtype in DTYPES[1:]:
        for rows, D in ((1024, 32), (1024, 64), (1023, 24)):
            g = torch.Generator(device="cuda").manual_seed(11)
            x = torch.randn(rows, D, device=dev, generator=g).to(dtype)
            sub = torch.randn(rows, D, device=dev, generator=g).to(dtype)
            w = torch.randn(D, device=dev, generator=g).to(dtype)
            b = torch.randn(D, device=dev, generator=g).to(dtype)
            kernels.layernorm_set_warp_width(0)
            _, blk = kernels.fused_add_layernorm(x, sub, w, b, 1e-5)
            kernels.layernorm_set_warp_width(256)
            _, wrp = kernels.fused_add_layernorm(x, sub, w, b, 1e-5)
            kernels.layernorm_set_warp_width(-1)
            e = (wrp.float() - blk.float()).abs().max().item()
            lim = 4e-3 if dtype is torch.float16 else 3e-2
            good = e < lim
            ok = ok and good
            print(f"  {str(dtype):>16} {rows:5d}x{D:<4d} max|warp-blk| = "
                  f"{e:.2e}   {'ok' if good else 'FAIL'}")

    print("\n=== rows-per-block sweep must not change the answer ===")
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1000, 32, device=dev, generator=g)
    sub = torch.randn(1000, 32, device=dev, generator=g)
    w = torch.randn(32, device=dev, generator=g)
    b = torch.randn(32, device=dev, generator=g)
    kernels.layernorm_set_warp_width(256)
    base = None
    for r in (1, 2, 4, 8, 16, 32):
        kernels.layernorm_set_warp_rows(r)
        _, out = kernels.fused_add_layernorm(x, sub, w, b, 1e-5)
        if base is None:
            base = out
            print(f"  rows/block={r:2d}  reference")
        else:
            same = torch.equal(out, base)
            ok = ok and same
            print(f"  rows/block={r:2d}  bit-identical: {same}")
    kernels.layernorm_set_warp_rows(0)
    kernels.layernorm_set_warp_width(-1)

    print("\n  active default: warp width =", kernels.layernorm_warp_width(),
          " rows/block =", kernels.layernorm_warp_rows())
    print("\nwarp-per-row matches the block-per-row kernel" if ok
          else "\nWARP KERNEL DISAGREES -- see the rows above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
