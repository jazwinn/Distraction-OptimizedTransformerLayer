"""Dispatch into the custom CUDA extension.

Nothing here knows about CUDA graphs or about the model; it is the boundary
between torch ops and csrc/.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import config


# Last value pushed into the extension's cp.async knob, same push-on-change
# contract as _fp16_pushed below.
_cp_async_pushed = None

# Last value pushed into the extension's wmma_set_fp16 knob. The knob is a
# process-global in the extension rather than a per-call argument, so it is
# pushed on change only -- six attention calls per forward pass would otherwise
# be six pybind round trips to set a flag that almost never moves.
_fp16_pushed = None


def _sync_attention_precision(kernels) -> None:
    """Keep the wmma kernel's process-wide fp16 knob in step with the config.

    The precision is passed per call now, so this only matters for
    ATTENTION_PRECISION == "auto", which is where the kernel falls back to this
    knob (and to the WMMA_FP16 environment variable the A/B scripts drive). An
    explicit precision goes straight through the call and ignores it.
    """
    global _fp16_pushed
    want = config.ATTENTION_PRECISION in ("auto", "fp16")
    if _fp16_pushed != want:
        kernels.wmma_set_fp16(want)
        _fp16_pushed = want


def _sync_cp_async(kernels) -> None:
    """Keep the kernel's cp.async knob in step with the config.

    Pushed on change rather than per call, for the same reason as the fp16
    knob: it is a process-global in the extension and a forward pass makes
    several attention calls that would otherwise each pay a pybind round trip
    to set a flag that almost never moves.
    """
    global _cp_async_pushed
    want = config._CP_ASYNC_CODE[config.CP_ASYNC]
    if _cp_async_pushed != want and hasattr(kernels, "wmma_set_cp_async"):
        kernels.wmma_set_cp_async(want)
        _cp_async_pushed = want


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
    normed_half: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(x + sub, norm(x + sub)), fused into one kernel where available.

    Unfused, the intermediate x + sub goes to global memory only to be read
    straight back. One kernel holds it on chip and returns both, since the
    caller needs the un-normalised sum for its own skip connection.
    """
    kernels = _custom_kernels()
    if kernels is not None:
        return kernels.fused_add_layernorm(x, sub, norm.weight, norm.bias,
                                           norm.eps, normed_half)

    total = x + sub
    out = norm(total)
    return total, (out.half() if normed_half else out)


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


def _layernorm(x: torch.Tensor, norm: "MyLayerNorm",
               out_half: bool = False) -> torch.Tensor:
    """LayerNorm(x) * w + b, on the custom kernel where it is available.

    The model's other LayerNorms all consume a residual add and fuse into it via
    _add_layernorm; this is the one that has nothing before it, so it needs the
    plain form. It was the last op in the forward pass still served by ATen, and
    ATen ran it at 198 GB/s on `[64,128,128]` against the 350 GB/s the same
    warp-per-row kernel reaches on twice the traffic.

    Falls back to F.layer_norm when the extension is unavailable, or when it
    predates the `fused_layernorm` entry -- the elementwise helpers here answer
    in torch rather than treating a missing extension as fatal, unlike
    attention.
    """
    if config.LAYERNORM != "off":
        kernels = _custom_kernels()
        if kernels is not None and hasattr(kernels, "fused_layernorm"):
            return kernels.fused_layernorm(x, norm.weight, norm.bias, norm.eps,
                                           out_half)

    out = F.layer_norm(x, (x.shape[-1],), norm.weight, norm.bias, norm.eps)
    return out.half() if out_half else out


def _attention_wants_fp16() -> bool:
    """Will the attention call really contract q/k/v in fp16?

    Only then is handing it fp16 tensors free. Three ways it would not:

      * an explicit fp32 or tf32 ATTENTION_PRECISION -- tf32 fragments load
        float, and fp32 rules the wmma kernel out entirely
      * ATTENTION_IMPL "scalar", whose whole point is fp32 accumulation, or
        "tile", which takes float32 pointers
      * bf16, which is a different narrowing and not this one

    "auto" resolves to fp16 for wmma, which is the kernel "auto" dispatch picks
    for every grading shape. Kept as a predicate rather than folded into the
    call site because getting it wrong is silent: the kernel would widen the
    fp16 back and the answer would still be right, just slower and no longer
    bit-comparable.
    """
    if config.QKV_FP16 == "off":
        return False
    if config.ATTENTION_IMPL not in ("auto", "wmma"):
        return False
    return config.ATTENTION_PRECISION in ("auto", "fp16")


def _normed_wants_fp16(rows: int, width: int) -> bool:
    """Should a LayerNorm whose consumer is the QKV GEMM emit fp16?

    Nearly the conditions that make QKV_FP16 free, because it is the same GEMM
    reading the result: the attention path must contract in fp16, the row count
    must be high enough for halved traffic to beat the fp16 read path's extra
    ALU, and low enough that the custom GEMM still serves the shape at all --
    above _LINEAR_BIAS_MAX_ROWS it declines and F.linear would have to widen
    the fp16 back again.

    One condition QKV_FP16 does not need: a bound on `width`, the GEMM's K.
    There the fp16 was the output; here it is the A operand, read once per
    k-iteration, so the cost scales with K while the saving does not. Shape 8
    lost 15% before this bound existed.
    """
    if config.NORMED_FP16 == "off":
        return False
    if not _attention_wants_fp16():
        return False
    # width is the GEMM's K, and it decides how many times the k-loop pays the
    # fp16 widening. See config._NORMED_FP16_MAX_K -- shape 8 lost 15% without
    # this bound.
    if width > config._NORMED_FP16_MAX_K:
        return False
    return config._QKV_FP16_MIN_ROWS <= rows <= config._LINEAR_BIAS_MAX_ROWS


def _gelu_input_wants_fp16(rows: int, width: int) -> bool:
    """Should the norm2 LayerNorm emit fp16 for linear_gelu to consume?

    Same shape reasoning as _normed_wants_fp16 -- it is the same GEMM machinery
    reading A -- but two differences:

      * it does not care what attention does; this edge never reaches attention
      * it DOES care that the fragments are fp16. Fp16Math narrows A on the way
        into shared memory, so pre-narrowing moves a rounding and changes no
        value; Tf32Math stages A as float and pre-narrowing would genuinely
        lose the mantissa. LINEAR_GELU == "tf32" is a measurement mode, and
        under it this must stay fp32.
    """
    if config.NORMED_FP16 == "off" or config.LINEAR_GELU != "auto":
        return False
    if width > config._NORMED_FP16_MAX_K:
        return False
    return config._QKV_FP16_MIN_ROWS <= rows <= config._LINEAR_BIAS_MAX_ROWS


def _linear_bias(x: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor],
                 out_half: bool = False) -> torch.Tensor:
    """x @ weight^T + bias, on the custom GEMM where it is faster than cuBLAS.

    The same kernel as _linear_gelu with the activation compiled out. It exists
    because the projections with no activation -- QKV, out_proj, ffn_out -- were
    the last cuBLAS calls in the forward pass, and cuBLAS serves them in TF32,
    or at the small-K grading shapes in *SIMT fp32*: shapes 7 and 12 land on
    cutlass_80_simt_sgemm_128x256, which is not using tensor cores at all. fp16
    fragments carry the same 10-bit mantissa as TF32 at twice the rate.

    Measured on the QKV shapes against F.linear, best-of interleaved with a
    self-control (see scripts/ab_common.py):

        M       K     N      cuBLAS    custom    ratio
        128     128   384    32.8us    11.8us    2.769x   shape 2
        2048    128   384    35.5us    20.0us    1.774x   shapes 4, 12
        8192    128   384    94.9us    61.4us    1.547x   shapes 1, 5, 9-11
        8192    1024  3072   2745us    2572us    1.067x   shape 8
        65536   128   384    445us     417us     1.067x   shape 13

    The win shrinks with M because cuBLAS reaches its own roofline once the
    grid is many waves deep, and past _LINEAR_BIAS_MAX_ROWS it turns into a
    loss -- see that constant for the crossover and for why shape 6 loses far
    more end to end than its op-level ratio predicts.

    A `None` bias falls through to F.linear: the kernel's epilogue always adds
    one, and every Linear in this model has one, so a biasless variant would be
    dead code.
    """
    rows = x.numel() // x.shape[-1] if x.dim() >= 1 else 0
    if (config.LINEAR_BIAS != "off" and bias is not None
            and rows <= config._LINEAR_BIAS_MAX_ROWS):
        kernels = _custom_kernels()
        if kernels is not None and hasattr(kernels, "linear_bias"):
            out = kernels.linear_bias(
                x, weight, bias, -1, config._GEMM_MATH_CODE[config.LINEAR_BIAS],
                out_half,
            )
            if out is not None:
                return out

    # The fallback cannot honour out_half, and must not silently produce an fp16
    # tensor the caller would hand to a consumer expecting fp32 -- or an fp32 one
    # where fp16 was promised. F.linear follows x's dtype, so cast the result to
    # what was asked for. Only reached when the kernel declines (shape 6's row
    # gate, or a build that predates linear_bias).
    out = F.linear(x, weight, bias)
    want = torch.float16 if out_half else torch.float32
    return out if out.dtype == want else out.to(want)


def _ffn_block(x, attn_out, norm1, ffn_in, ffn_out, norm2):
    """The whole post-attention block in one kernel, or None to run it unfused.

    None rather than an exception on every decline: this is a speed preference,
    and the caller has a correct path either way. The width gate is here rather
    than in the kernel because the kernel covers up to d_model 128 and only
    *wins* up to 64 -- see config._FFN_BLOCK_MAX_D for the measured crossover.

    Not reached on the padded path. Padded rows are zeroed between the add and
    the norm, and the reference normalises the zeroed rows rather than the raw
    sum, so that pair genuinely cannot fuse.
    """
    if config.FFN_BLOCK == "off":
        return None
    if config.FFN_BLOCK != "force" and x.shape[-1] > config._FFN_BLOCK_MAX_D:
        return None
    kernels = _custom_kernels()
    if kernels is None or not hasattr(kernels, "fused_ffn_block"):
        return None
    out = kernels.fused_ffn_block(
        x, attn_out, norm1.weight, norm1.bias,
        ffn_in.weight, ffn_in.bias, ffn_out.weight, ffn_out.bias,
        norm2.weight, norm2.bias, norm1.eps,
    )
    # Empty list is the kernel declining a shape it has no instantiation for.
    return (out[0], out[1]) if out else None


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
    _sync_cp_async(kernels)
    return kernels.fused_attention_forward(
        q, k, v, attn_mask, is_causal, scale,
        config._IMPL_CODE[config.ATTENTION_IMPL],
        config._OUT_LAYOUT_BSHD,
        config._PRECISION_CODE[config.ATTENTION_PRECISION],
    )
