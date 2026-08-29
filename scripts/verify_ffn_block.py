"""Verify the fused post-attention block against the unfused chain it replaces.

Two references, because they answer different questions:

  * `torch`  -- the arithmetic the harness grades against, in fp64 for the
    reference so its own rounding does not count against the kernel.
  * `chain`  -- the existing fused_add_layernorm + linear_gelu + mm + add/LN
    sequence. This is the one that matters for a swap: the fused kernel has to
    agree with what it is replacing, not merely be close to fp64.

Judged against the harness budget (atol 2e-3 / rtol 2e-2), not against zero:
both paths run tf32 GEMMs and sum in different orders, so exact equality is not
the target and demanding it would reject a correct kernel.

    python scripts/verify_ffn_block.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernel_ext  # noqa: E402  (must precede torch; see preload_tile_compiler)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ATOL = 2e-3
RTOL = 2e-2

# (label, rows, d_model, ffn_dim) -- the widths the grading shapes actually use,
# plus a ragged row count to exercise the tail-row zeroing.
CASES = [
    ("shape 1/5/6 width", 8192, 128, 128),
    ("shape 7 width",     8192,  32,  32),
    ("shape 13 width",   65536, 128, 128),
    ("ragged rows",         17, 128, 128),
    ("single tile",         16, 128, 128),
    ("uneven tail",       1000, 128, 128),
    ("d64 f64",           4096,  64,  64),
]


def reference(x, attn_out, n1w, n1b, wi, bi, wo, bo, n2w, n2b, eps):
    """fp64 reference: what the harness grades against."""
    xd = x.double()
    x1 = xd + attn_out.double()
    n1 = F.layer_norm(x1, (x1.shape[-1],), n1w.double(), n1b.double(), eps)
    h = F.gelu(F.linear(n1, wi.double(), bi.double()), approximate="none")
    y = F.linear(h, wo.double(), bo.double())
    x2 = x1 + y
    n2 = F.layer_norm(x2, (x2.shape[-1],), n2w.double(), n2b.double(), eps)
    return x2, n2


def chain(ext, x, attn_out, n1w, n1b, wi, bi, wo, bo, n2w, n2b, eps):
    """The unfused kernels this replaces, in fp32."""
    x1, n1 = ext.fused_add_layernorm(x, attn_out, n1w, n1b, eps)
    h = F.gelu(F.linear(n1, wi, bi), approximate="none")
    y = F.linear(h, wo, bo)
    return ext.fused_add_layernorm(x1, y, n2w, n2b, eps)


def report(name, got, want, tag):
    err = (got.double() - want).abs()
    rel = err / want.abs().clamp_min(1e-12)
    ok = ((err <= ATOL) | (rel <= RTOL)).all().item()
    bad = int((~((err <= ATOL) | (rel <= RTOL))).sum().item())
    print(f"    {tag:<6} {name:<8} max_abs={err.max().item():.3e} "
          f"max_rel={rel.max().item():.3e} failed={bad}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ext = kernel_ext.get_kernels()
    if ext is None or not hasattr(ext, "fused_ffn_block"):
        print("fused_ffn_block is not in the extension; build it first")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    eps = 1e-5
    all_ok = True

    for label, rows, D, Fd in CASES:
        torch.manual_seed(1234)
        x = torch.randn(rows, D, device=dev)
        attn_out = torch.randn(rows, D, device=dev)
        n1w = torch.randn(D, device=dev) * 0.1 + 1.0
        n1b = torch.randn(D, device=dev) * 0.1
        wi = torch.randn(Fd, D, device=dev) * (D ** -0.5)
        bi = torch.randn(Fd, device=dev) * 0.1
        wo = torch.randn(D, Fd, device=dev) * (Fd ** -0.5)
        bo = torch.randn(D, device=dev) * 0.1
        n2w = torch.randn(D, device=dev) * 0.1 + 1.0
        n2b = torch.randn(D, device=dev) * 0.1

        out = ext.fused_ffn_block(x, attn_out, n1w, n1b, wi, bi, wo, bo,
                                  n2w, n2b, eps)
        if not out:
            print(f"  {label}: rows={rows} D={D} F={Fd} -> DECLINED (not covered)")
            continue
        x2_f, n2_f = out

        ref_x2, ref_n2 = reference(x, attn_out, n1w, n1b, wi, bi, wo, bo,
                                   n2w, n2b, eps)
        ch_x2, ch_n2 = chain(ext, x, attn_out, n1w, n1b, wi, bi, wo, bo,
                             n2w, n2b, eps)

        print(f"  {label}: rows={rows} D={D} F={Fd}")
        all_ok &= report("x2", x2_f, ref_x2, "fp64")
        all_ok &= report("normed", n2_f, ref_n2, "fp64")
        all_ok &= report("x2", x2_f, ch_x2.double(), "chain")
        all_ok &= report("normed", n2_f, ch_n2.double(), "chain")

    print()
    print("every case matches within the harness budget" if all_ok
          else "MISMATCH -- see the FAIL rows above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
