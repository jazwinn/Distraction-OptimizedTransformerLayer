"""Dispatch into the custom CUDA extension.

Nothing here knows about CUDA graphs or about the model; it is the boundary
between torch ops and csrc/.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import config


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
    """The extension module, or None when the build is unavailable.

    None is now only ever "the extension would not load" -- the switch that
    used to turn the custom kernels off wholesale went with the SDPA backend.
    Attention treats that as fatal (see _attention_dispatch); the elementwise
    helpers below can still answer in torch, and do.
    """
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
) -> torch.Tensor:
    """Route attention to the custom CUDA kernel. There is no other route.

    Always returns [B, S, H*head_dim], the layout out_proj consumes: the
    kernels write it from their epilogue, so no transpose or reshape happens
    here at all.

    `attn_mask` and `is_causal` may both be set, which is what lets a padded
    causal call stay a [B, 1, 1, S] key mask plus a flag rather than a folded
    [B, 1, S, S] triangle -- the kernels keep their causal early exit that way,
    worth 1.18x-1.85x on the op.

    This used to fall back to F.scaled_dot_product_attention when the extension
    would not load, and `causal_mask` existed to hand that path a cached [S, S]
    triangle, because SDPA rejects the mask/flag pair the kernels accept. Both
    are gone: a prebuilt attention is not an acceptable substitute here, so a
    missing extension is an error rather than a quietly different answer.
    """
    import kernel_ext

    kernels = kernel_ext.get_kernels()
    if kernels is None:
        raise RuntimeError(
            "the CUDA extension failed to load, and there is no fallback: this "
            "project implements attention itself and will not substitute a "
            "prebuilt one. Build it with scripts/build_ext.bat. "
            f"Cause: {kernel_ext.load_error()}"
        )
    _sync_attention_precision(kernels)
    return kernels.fused_attention_forward(
        q, k, v, attn_mask, is_causal, scale,
        config._IMPL_CODE[config.ATTENTION_IMPL],
        config._OUT_LAYOUT_BSHD,
    )
