"""Dispatch into the custom CUDA extension, with SDPA and ATen fallbacks.

Nothing here knows about CUDA graphs or about the model; it is the boundary
between torch ops and csrc/.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import config

_fallback_warned = False


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


def _attention_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """Route attention to the custom CUDA kernel or to SDPA.

    Always returns [B, S, H*head_dim], the layout out_proj consumes, whichever
    backend served the call: the custom kernels write it from their epilogue,
    SDPA cannot, so the transpose/reshape happens here. One contract, so the
    caller never has to ask which backend it got.
    """
    global _fallback_warned

    if config.ATTENTION_BACKEND != "sdpa":
        import kernel_ext

        kernels = kernel_ext.get_kernels()
        if kernels is not None:
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

    context = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
    # [B, H, S, D] -> [B, S, H*D]. Cannot be a view across the transpose, so
    # this is the strided repack the custom kernels skip. flatten(2) rather than
    # reshape(b, s, h*d) keeps the shape arithmetic on the C++ side, which is
    # not free in a model this close to launch-bound.
    return context.transpose(1, 2).flatten(2)
