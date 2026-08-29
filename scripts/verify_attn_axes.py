"""
Check the impl x precision matrix -- that every pair either runs or refuses,
exactly as declared, in all three places it is written down.

The two axes are independent now: --attn-impl picks the kernel, --attn-precision
picks the arithmetic. Not every pair exists, and which ones do is stated three
times:

  csrc/kernel_common.cuh        the table in the AttnPrecision comment
  csrc/attention_dispatch.cuh   impl_supports(), which actually enforces it
  dashboard/knobs.py            PRECISION_SUPPORT, which blocks a run early

Three copies can disagree, and the failure is silent in both directions: a
dashboard that blocks a runnable pair is as wrong as one that queues a run the
kernel will refuse. So this asks the kernel and the dashboard about every pair
and fails on any disagreement, rather than checking either against a list
written here -- a fourth copy would just be one more thing to drift.

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_attn_axes.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernel_ext  # noqa: E402  -- before torch, see verify_kernel.py

import torch  # noqa: E402

from dashboard import knobs  # noqa: E402
from optimized import config  # noqa: E402

from scripts.verify_kernel import (  # noqa: E402
    build_case,
    reference_attention_f64,
)

# Read the codes off config rather than repeating them, so a renamed or
# renumbered value shows up here as a KeyError instead of a wrong answer.
IMPLS = config._IMPL_CODE
PRECISIONS = config._PRECISION_CODE

# Budgets by precision. fp32 is the scalar kernel's, three orders tighter than
# the tensor-core paths; bf16's 8 significand bits earn an order more room.
TOL = {"auto": 3e-3, "fp32": 5e-6, "tf32": 3e-3, "fp16": 3e-3, "bf16": 3e-2}

# head_dim cases the coverage rules should decide, and what they should decide.
# 96 is the interesting one: it has no tuned instantiation anywhere, so wmma
# refuses it while scalar's generic kernel takes it -- which is also why auto
# takes it.
COVERAGE_CASES = (
    ("auto", 384, 4, True, "generic scalar catches it"),
    ("scalar", 384, 4, True, "generic scalar, forced"),
    ("wmma", 384, 4, False, "no tuned head_dim 96"),
    ("tile", 384, 4, False, "no tuned head_dim 96"),
    ("scalar", 4096, 2, True, "head_dim 2048, exactly the ceiling"),
    ("auto", 8192, 2, False, "head_dim 4096, past every kernel"),
)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print("extension did not load:", kernel_ext.load_error())
        return 1

    B, H, S, D = 2, 4, 128, 64
    q, k, v, mask, causal = build_case(B, H, S, D, True, True, "cuda",
                                       torch.float32, seed=3)
    scale = D ** -0.5
    ref = reference_attention_f64(q, k, v, mask, causal, scale)

    failures = []
    print("Every pair, against the kernel and against the dashboard's copy:\n")
    print(f"{'impl':<8} {'precision':<10} {'kernel':<9} {'dashboard':<10} "
          f"{'max_abs':>10}  verdict")

    for iname, icode in IMPLS.items():
        for pname, pcode in PRECISIONS.items():
            try:
                out = kernels.fused_attention_forward(
                    q, k, v, mask, causal, scale, icode, 0, pcode)
                err = (out.double() - ref).abs().max().item()
                # auto + fp32 lands on the scalar kernel, so it is held to
                # scalar's budget rather than to auto's looser one.
                tol = TOL["fp32"] if pname == "fp32" else TOL[pname]
                kernel_ok = err <= tol and torch.isfinite(out).all().item()
                note = f"{err:10.2e}" + ("" if kernel_ok else f" >{tol:.0e}")
            except RuntimeError:
                kernel_ok, note = False, f"{'refused':>10}"

            form = {"batch_size": B, "seq_len": S, "d_model": H * D,
                    "heads": H, "layers": 2,
                    "attn_impl": iname, "attn_precision": pname}
            issues = knobs.preflight(form, tile_available=True,
                                     device_total_bytes=8 * 1024 ** 3)
            dash_ok = not any(i["level"] == "error" for i in issues)

            agree = kernel_ok == dash_ok
            if not agree:
                failures.append(f"{iname}+{pname}: kernel "
                                f"{'runs' if kernel_ok else 'refuses'} but "
                                f"dashboard {'allows' if dash_ok else 'blocks'}")
            print(f"{iname:<8} {pname:<10} "
                  f"{('runs' if kernel_ok else 'refuses'):<9} "
                  f"{('allows' if dash_ok else 'blocks'):<10} {note}  "
                  f"{'ok' if agree else 'DISAGREE'}")

    print("\nhead_dim coverage the dashboard should decide before launching:")
    for impl, d_model, heads, want_allowed, why in COVERAGE_CASES:
        form = {"batch_size": 1, "seq_len": 64, "d_model": d_model,
                "heads": heads, "layers": 1, "attn_impl": impl,
                "attn_precision": "auto"}
        issues = knobs.preflight(form, True, 8 * 1024 ** 3)
        allowed = not any(i["level"] == "error" for i in issues)
        ok = allowed == want_allowed
        if not ok:
            failures.append(f"{impl} head_dim {d_model // heads}: "
                            f"{'allowed' if allowed else 'blocked'}, wanted the "
                            f"opposite")
        print(f"  {impl:<7} head_dim {d_model // heads:<5} "
              f"{'allowed' if allowed else 'blocked':<8} {'ok' if ok else 'WRONG':<6} "
              f"{why}")

    # A refusal has to say what IS available, or it sends the reader to the
    # source to find out.
    print("\nrefusal messages name the alternative:")
    for impl, prec in (("scalar", "fp16"), ("wmma", "fp32")):
        try:
            kernels.fused_attention_forward(
                q, k, v, mask, causal, scale,
                IMPLS[impl], 0, PRECISIONS[prec])
            failures.append(f"{impl}+{prec} was expected to raise")
        except RuntimeError as exc:
            text = str(exc).splitlines()[0]
            if "supports" not in text:
                failures.append(f"{impl}+{prec} raised without naming what it "
                                f"does support")
            print(f"  {impl}+{prec}: {text[:110]}")

    if failures:
        print(f"\n{len(failures)} FAILURES")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nkernel, dispatcher and dashboard all agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
