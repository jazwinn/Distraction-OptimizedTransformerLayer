"""Equivalence of the GEMM epilogue's two global-store paths.

The epilogue has two. The vector path takes 4 elements per lane and needs the
strip whole and `N % 4 == 0`; everything else -- a ragged last row block, a
ragged or unaligned N -- falls to a scalar loop. The fast path is selected per
*strip*, so one launch can use both.

**This does not compare against a float64 reference, deliberately.** These
kernels contract in fp16, so at K=128 with unit-variance inputs the dot product
has magnitude ~sqrt(K) and an inherent absolute error around 1e-2. Any element
whose result lands near zero then has an enormous relative error for reasons
that have nothing to do with the store path, and a reference-based check drowns
in them -- an earlier version of this file reported every case failing,
including the path it had not touched.

What is actually in question is narrower and admits an exact answer: **does the
vectorized store write the same bytes the scalar store would?** Both read the
same shared scratch, which holds the same accumulator, so the two must agree
*bit for bit* -- only the instruction width differs. So the test compares the
paths to each other:

    vector : linear_bias(A, W,        b)            N % 4 == 0
    scalar : linear_bias(A, W[:N-1],  b[:N-1])      N-1 is odd, forcing scalar

with the first N-1 columns of the vector result compared to the scalar result.
Same A, same weights, same accumulation order; if they differ at all, the store
is wrong. A reference comparison is still printed, but as information rather
than as the gate.
"""
import sys

import torch

sys.path.insert(0, ".")
import kernel_ext  # noqa: E402  (must precede torch; see preload_tile_compiler)


def fp16_reference(a, w, b, act):
    """The arithmetic the kernel actually promises: fp16 multiply, fp32 sum."""
    out = (a.half().float() @ w.half().float().t()) + b
    if act:
        out = out * 0.5 * (1.0 + torch.erf(out / (2.0 ** 0.5)))
    return out


def run(kern, a, w, b, act, out_half):
    if act:
        return kern.linear_gelu(a, w, b, -1, -1)
    return kern.linear_bias(a, w, b, -1, -1, out_half)


def check(kern, M, N, K, act, out_half, seed=0):
    """Vector path against scalar path on the same data. Must be identical."""
    assert N % 4 == 0, "the vector arm needs N % 4 == 0"
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float32)
    w = torch.randn(N, K, generator=g, device="cuda", dtype=torch.float32)
    b = torch.randn(N, generator=g, device="cuda", dtype=torch.float32)

    vec = run(kern, a, w, b, act, out_half)
    # N-1 is odd, so `(N & 3) == 0` is false and every strip takes the scalar
    # loop. Contiguous slices, so the kernel sees identical row strides.
    sca = run(kern, a, w[:N - 1].contiguous(), b[:N - 1].contiguous(),
              act, out_half)

    lhs, rhs = vec[:, :N - 1], sca
    identical = torch.equal(lhs, rhs)
    if identical:
        gap = 0.0
    else:
        gap = (lhs.double() - rhs.double()).abs().max().item()

    ref = fp16_reference(a, w, b, act)
    ref_err = (vec.float() - ref).abs().max().item()

    dt = "fp16" if out_half else "fp32"
    tag = "gelu" if act else "bias"
    ragged = " ragged-M" if (M % 16) else ""
    ok = "ok" if identical else "MISMATCH"
    print(f"  M={M:<7} N={N:<6} K={K:<5} {tag:<4} out={dt:<4}{ragged:<10} "
          f"paths-agree={ok:<9} gap={gap:.3e}  (vs fp16 ref {ref_err:.2e})")
    return identical


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    kern = kernel_ext.get_kernels()
    if kern is None:
        print("extension failed to load:", kernel_ext.load_error())
        return 1
    print(torch.cuda.get_device_name(0))
    print("\nvectorized store vs scalar store, same data -- must be bit-identical:")

    cases = []
    # Shapes the model runs: 384 is the fused QKV projection, 128/1024 d_model,
    # 3072 is 3*1024. Each also exercises a different block tile.
    for N in (128, 384, 1024, 3072):
        cases.append((256, N, 128))
    # Ragged M: the last row block is partial, so one launch mixes both paths.
    for M in (1, 17, 63, 130, 257):
        cases.append((M, 384, 128))
    # K off the block tile's BK, and a wide N.
    cases.append((512, 1536, 96))
    cases.append((1024, 512, 256))
    # Large M -- the regime row 1a is about.
    cases.append((65536, 384, 128))

    ok = True
    for M, N, K in cases:
        ok &= check(kern, M, N, K, act=False, out_half=False)
    print()
    for M, N, K in cases:
        ok &= check(kern, M, N, K, act=False, out_half=True)
    print()
    for M, N, K in cases:
        ok &= check(kern, M, N, K, act=True, out_half=False)

    print("\n" + "-" * 60)
    print("both store paths write identical bytes on every shape" if ok
          else "STORE PATHS DISAGREE -- the vectorized epilogue is wrong")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
