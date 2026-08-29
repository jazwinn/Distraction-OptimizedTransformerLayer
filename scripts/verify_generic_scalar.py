"""
Check the generic scalar kernel -- the catch-all that covers head_dims no
specialization takes.

Three things are being checked, and they are not the same claim:

  1. Coverage.   head_dims outside {8,16,32,64,128,256} now return an answer
                 instead of raising, in every dtype, with and without masks.
  2. Correctness. That answer matches a float64 reference to the same budget
                 the tuned scalar kernel is held to, because it is the same
                 fp32 arithmetic -- the padding columns must contribute
                 nothing, not merely something small.
  3. Equivalence. Run again with SCALAR_FORCE_GENERIC=1 and the generic kernel
                 serves the six head_dims the specializations own, so the two
                 can be diffed on identical shapes. A pass there means the
                 generalization did not change the algorithm, only who runs it.

Run it both ways -- the env var is read once per process:

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_generic_scalar.py
    cmd.exe /c scripts\\devenv.bat cmd /c "set SCALAR_FORCE_GENERIC=1 && python scripts\\verify_generic_scalar.py"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernel_ext  # noqa: E402  -- before torch, see verify_kernel.py

import torch  # noqa: E402

from scripts.verify_kernel import (  # noqa: E402
    build_case,
    reference_attention_f64,
    as_packed_views,
)

# The catch-all's own head_dims: none of these has a specialization, and every
# one of them raised before this kernel existed.
#
#  48/96/192  multiples of 16 that are not powers of two -- what a d_model of
#             768 over 8 heads actually produces
#  40/72/100  not multiples of 16 either, so a thread's 64-dim slice straddles
#             head_dim and the per-element bound test is load-bearing
#  1/2/7      narrower than one warp's worth of anything
#  320/512    past every specialization
#  1024/2048  TPR 16 and TPR 32, the block-size and warp-width ceilings
UNCOVERED = (1, 2, 7, 40, 48, 72, 96, 100, 129, 192, 320, 512, 1024, 2048)

# What the specializations own. Only reached when SCALAR_FORCE_GENERIC=1.
COVERED = (8, 16, 32, 64, 128, 256)

# (label, B, H, S, causal, padded)
SHAPES = (
    ("dense",     2, 3,  64, False, False),
    ("causal",    2, 3,  64, True,  False),
    ("padded",    2, 3,  64, False, True),
    ("caus+pad",  2, 3,  64, True,  True),
    ("ragged S",  3, 2,  37, True,  True),
    ("long",      1, 2, 512, True,  False),
)

# The generic kernel accumulates in fp32 exactly as the tuned one does, so it
# gets the tuned one's budget against an exact reference -- 5e-6 -- and not a
# looser one for being general. fp16/bf16 inputs carry their own input and
# output rounding on top; those budgets are the storage format's, not the
# kernel's, and were read off the measured maxima rather than guessed.
TOL = {
    torch.float32: 5e-6,
    torch.float64: 5e-6,
    torch.float16: 5e-3,
    torch.bfloat16: 4e-2,
}

IMPL_SCALAR = 1


def run(kernels, q, k, v, attn_mask, is_causal, scale, layout=0):
    return kernels.fused_attention_forward(
        q, k, v, attn_mask, is_causal, scale, IMPL_SCALAR, layout
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print("extension did not load:", kernel_ext.load_error())
        return 1

    forced = os.environ.get("SCALAR_FORCE_GENERIC") == "1"
    head_dims = (UNCOVERED + COVERED) if forced else UNCOVERED
    print(f"SCALAR_FORCE_GENERIC={'1' if forced else '0'} -- "
          f"{'generic serves every head_dim below' if forced else 'generic serves the uncovered head_dims below'}")
    print(f"{'shape':<10} {'dtype':<9} {'head_dim':>8} {'max_abs':>11} {'tol':>9}  result")

    failures = []
    worst = {}
    for label, B, H, S, causal, padded in SHAPES:
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            for d in head_dims:
                # Keep the big cases off the ragged/long shapes: a [B,H,S,2048]
                # tensor in fp64 is the only thing here that could OOM an 8 GB
                # card, and it tests nothing the small shapes do not.
                if d >= 512 and (S > 128 or dtype == torch.float64):
                    continue
                q, k, v, mask, is_causal = build_case(
                    B, H, S, d, causal, padded, "cuda", dtype, seed=d
                )
                scale = d ** -0.5
                try:
                    got = run(kernels, q, k, v, mask, is_causal, scale)
                except RuntimeError as e:
                    print(f"{label:<10} {str(dtype)[6:]:<9} {d:>8} {'-':>11} {'-':>9}  "
                          f"RAISED: {str(e).splitlines()[0][:60]}")
                    failures.append((label, dtype, d, "raised"))
                    continue

                ref = reference_attention_f64(q, k, v, mask, is_causal, scale)
                err = (got.double() - ref).abs().max().item()
                tol = TOL[dtype]
                ok = err <= tol and torch.isfinite(got).all().item()
                worst[dtype] = max(worst.get(dtype, 0.0), err)
                if not ok:
                    failures.append((label, dtype, d, err))
                    print(f"{label:<10} {str(dtype)[6:]:<9} {d:>8} {err:>11.3e} "
                          f"{tol:>9.1e}  FAIL")

    # Strided q/k/v: the fused QKV projection hands the kernel non-contiguous
    # views. Same values at different addresses, so the answer must be bitwise
    # identical -- not merely within tolerance.
    print("\nstrided views (must be bitwise identical to the contiguous run):")
    for d in (48, 96, 192, 320):
        q, k, v, mask, is_causal = build_case(2, 3, 64, d, True, True, "cuda",
                                              torch.float32, seed=d)
        base = run(kernels, q, k, v, mask, is_causal, d ** -0.5)
        qs, ks, vs = as_packed_views(q, k, v)
        strided = run(kernels, qs, ks, vs, mask, is_causal, d ** -0.5)
        same = torch.equal(base, strided)
        print(f"  head_dim {d:>4}: {'identical' if same else 'DIFFERS'}")
        if not same:
            failures.append(("strided", torch.float32, d, "not bitwise equal"))

    # Both output layouts, since the epilogue writes them directly.
    print("\nout_layout 1 ([B,S,H*head_dim], what out_proj consumes):")
    for d in (48, 96, 320):
        q, k, v, mask, is_causal = build_case(2, 3, 64, d, False, False, "cuda",
                                              torch.float32, seed=d)
        got = run(kernels, q, k, v, mask, is_causal, d ** -0.5, layout=1)
        ref = reference_attention_f64(q, k, v, mask, is_causal, d ** -0.5, layout=1)
        err = (got.double() - ref).abs().max().item()
        ok = got.shape == ref.shape and err <= TOL[torch.float32]
        print(f"  head_dim {d:>4}: shape {tuple(got.shape)} max_abs {err:.3e} "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(("layout1", torch.float32, d, err))

    # The declared ceiling. 2048 is the widest a row's threads fit in a warp;
    # 4096 must decline loudly rather than launch something wrong.
    print("\nceiling:")
    for d, expect_ok in ((2048, True), (4096, False)):
        q, k, v, mask, is_causal = build_case(1, 1, 16, d, False, False, "cuda",
                                              torch.float32, seed=1)
        try:
            got = run(kernels, q, k, v, mask, is_causal, d ** -0.5)
            ref = reference_attention_f64(q, k, v, mask, is_causal, d ** -0.5)
            err = (got.double() - ref).abs().max().item()
            got_ok, note = True, f"max_abs {err:.3e}"
            if err > TOL[torch.float32]:
                got_ok = False
        except RuntimeError as e:
            got_ok, note = False, "raised: " + str(e).splitlines()[0][:50]
        status = "ok" if got_ok == expect_ok else "FAIL"
        print(f"  head_dim {d:>4}: expected {'a result' if expect_ok else 'a raise':<10} "
              f"-> {note}  {status}")
        if got_ok != expect_ok:
            failures.append(("ceiling", torch.float32, d, note))

    print("\nworst max_abs by dtype:")
    for dtype, err in sorted(worst.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(dtype)[6:]:<10} {err:.3e}  (tol {TOL[dtype]:.1e})")

    if failures:
        print(f"\n{len(failures)} FAILURES")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
