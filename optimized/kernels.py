"""Dispatch into the custom CUDA extension, with an SDPA fallback.

Nothing here knows about CUDA graphs or about the model; it is the boundary
between torch ops and csrc/.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from . import config

_fallback_warned = False

# Last value pushed into the extension's wmma_set_fp16 knob. The knob is a
# process-global in the extension rather than a per-call argument, so it is
# pushed on change only -- six attention calls per forward pass would otherwise
# be six pybind round trips to set a flag that almost never moves.
_fp16_pushed = None


def _sync_attention_precision(kernels) -> None:
    global _fp16_pushed
    want = config.ATTENTION_FP16 == "auto"
    if _fp16_pushed != want:
        kernels.wmma_set_fp16(want)
        _fp16_pushed = want


def _custom_kernels():
    """The extension module, or None when unavailable or switched off.

    ATTENTION_BACKEND == "sdpa" means "no custom kernels at all", not just no
    custom attention -- otherwise --attn-backend sdpa would still be timing
    hand-written code and the comparison would mean nothing.
    """
    if config.ATTENTION_BACKEND == "sdpa":
        return None
    import kernel_ext

    return kernel_ext.get_kernels()


def _add_layernorm(
    x: torch.Tensor,
    sub: torch.Tensor,
    norm: "MyLayerNorm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(x + sub, norm(x + sub)), fused into one kernel where available.

    Unfused, the intermediate x + sub goes to global memory only to be read
    straight back. One kernel holds it on chip and returns both, since the
    caller needs the un-normalised sum for its own skip connection.
    """
    kernels = _custom_kernels()
    if kernels is not None:
        return kernels.fused_add_layernorm(x, sub, norm.weight, norm.bias, norm.eps)

    total = x + sub
    return total, norm(total)


def _linear_gelu(x: torch.Tensor, lin: "MyLinear") -> torch.Tensor:
    """GELU(lin(x)), fused into one kernel on the shapes where that is faster.

    Unfused, the GELU pass reads the whole [M, ffn_dim] GEMM result back out of
    global memory, applies one cheap function and writes it again. Folding it
    into the accumulator makes it free -- a third of the pair's cost at this
    model's shapes, where ffn_dim == d_model and the GEMM is small.

    The extension decides which shapes it can serve: `linear_gelu` returns an
    undefined tensor -- None here -- rather than raising, so the fallback below
    covers anything it declines. With fp16 fragments it is faster than cuBLAS at
    every shape measured, so nothing is declined on speed grounds any more; see
    pick_gemm_tile() for the table.
    """
    if config.LINEAR_GELU != "off" and lin.bias is not None:
        kernels = _custom_kernels()
        if kernels is not None:
            out = kernels.linear_gelu(
                x, lin.weight, lin.bias, -1,
                config._GEMM_MATH_CODE[config.LINEAR_GELU],
            )
            if out is not None:
                return out

    return F.gelu(lin(x), approximate="none")


def _attention_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
    causal_mask: Optional[Callable[[], torch.Tensor]] = None,
) -> torch.Tensor:
    """Route attention to the custom CUDA kernel or to SDPA.

    Always returns [B, S, H*head_dim], the layout out_proj consumes, whichever
    backend served the call: the custom kernels write it from their epilogue,
    SDPA cannot, so the transpose/reshape happens here. One contract, so the
    caller never has to ask which backend it got.

    `attn_mask` and `is_causal` may both be set, which is what lets a padded
    causal call stay a [B, 1, 1, S] key mask plus a flag rather than a folded
    [B, 1, S, S] triangle -- the kernels keep their causal early exit that way,
    worth 1.18x-1.85x on the op. SDPA cannot take the pair, so `causal_mask`
    optionally supplies a cached [S, S] triangle to fold with; it is called only
    on the branch that needs it, and that branch builds its own when there is
    none.
    """
    global _fallback_warned

    if config.ATTENTION_BACKEND != "sdpa":
        import kernel_ext

        kernels = kernel_ext.get_kernels()
        if kernels is not None:
            _sync_attention_precision(kernels)
            return kernels.fused_attention_forward(
                q, k, v, attn_mask, is_causal, scale,
                config._IMPL_CODE[config.ATTENTION_IMPL],
                config._OUT_LAYOUT_BSHD
            )
        if config.ATTENTION_BACKEND == "custom":
            raise RuntimeError(
                'ATTENTION_BACKEND is "custom" but the CUDA extension failed to '
                "load. Build it with scripts/build_ext.bat. "
                f"Cause: {kernel_ext.load_error()}"
            )
        if not _fallback_warned:
            _fallback_warned = True
            print(
                f"[info] custom CUDA kernel unavailable, using SDPA instead "
                f"(results are still correct). Reason: {kernel_ext.load_error()}\n"
                f"[info] to use the kernel:  cmd.exe /c scripts\\devenv.bat "
                f"python torch_transformer_benchmark.py\n"
                f'[info] to silence this:    set ATTENTION_BACKEND = "sdpa" '
                f"in optimized/config.py"
            )

    if is_causal and attn_mask is not None:
        # SDPA rejects the combination the kernels accept. Fold, and pay for the
        # [B, 1, S, S] the custom path no longer builds. `causal_mask` is the
        # caller's cache for the triangle -- seq_len is fixed for a model
        # instance's lifetime, so rebuilding it per layer would be waste -- but
        # it is a cache, not a contract: without one, build the triangle here
        # rather than refuse. This is the branch that has to stay correct for
        # any caller, not the one that has to be fast.
        if causal_mask is not None:
            allowed = causal_mask()
        else:
            seq_len = q.shape[2]
            allowed = torch.ones(
                seq_len, seq_len, device=q.device, dtype=torch.bool
            ).tril()
        attn_mask = attn_mask & allowed
        is_causal = False

    context = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
    # [B, H, S, D] -> [B, S, H*D]. Cannot be a view across the transpose, so
    # this is the strided repack the custom kernels skip. flatten(2) rather than
    # reshape(b, s, h*d) keeps the shape arithmetic on the C++ side, which is
    # not free in a model this close to launch-bound.
    return context.transpose(1, 2).flatten(2)
