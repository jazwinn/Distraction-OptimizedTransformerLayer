"""Drop-in replacements for the baseline's submodules.

Each keeps the baseline's parameter and submodule names, which is what lets
copy_model_weights() load one state_dict into either model with strict=True.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import (_add_layernorm, _attention_dispatch, _ffn_block,
                      _linear_gelu)
from .util import _version_or_none


class MyLinear(nn.Module):
    """Same parameter names/shapes as nn.Linear -> free strict weight loading."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ====================== your codes here ======================
        return F.linear(x, self.weight, self.bias)
        # ============================================================


class MyLayerNorm(nn.Module):
    """Same parameter names/shapes as nn.LayerNorm -> free strict weight loading."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ====================== your codes here ======================
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        # ============================================================


class MySelfAttention(nn.Module):
    """Same submodule names as BaselineSelfAttention -> free strict weight loading."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = MyLinear(d_model, d_model)
        self.k_proj = MyLinear(d_model, d_model)
        self.v_proj = MyLinear(d_model, d_model)
        self.out_proj = MyLinear(d_model, d_model)

        # Lazily-built causal mask cache, for the SDPA fallback only -- it is
        # the one consumer that cannot take is_causal and a mask together and
        # so has to fold them. The custom kernels never touch it. Plain
        # attributes, not buffers, so they stay out of state_dict() and strict
        # weight copying keeps working.

        # Fused QKV projection cache. See _get_qkv_weight.
        self._qkv_key: Optional[Tuple] = None
        self._qkv_weight: Optional[torch.Tensor] = None
        self._qkv_bias: Optional[torch.Tensor] = None

    def _get_qkv_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q, K and V as one [3*d_model, d_model] weight, built on demand.

        Three separate [B*S, d]x[d, d] GEMMs leave ~32 output tiles for 46 SMs,
        so cuBLAS splits the contraction and launches a second kernel to add the
        partial sums back -- that reduction alone was 9.5% of GPU time in a
        profile. One [d, 3*d] GEMM produces ~96 tiles and fills the card in a
        single pass.

        Lazy rather than in __init__ because the weights are not there at
        construction time: they arrive via load_state_dict() and then move under
        .to(device, dtype). The key covers both.
        """
        w_q, w_k, w_v = self.q_proj.weight, self.k_proj.weight, self.v_proj.weight
        key = (w_q.device, w_q.dtype,
               _version_or_none(w_q), _version_or_none(w_k), _version_or_none(w_v))
        if self._qkv_key != key:
            self._qkv_weight = torch.cat([w_q, w_k, w_v], dim=0)
            self._qkv_bias = torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
            )
            self._qkv_key = key
        return self._qkv_weight, self._qkv_bias


    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        mask_is_trivial: bool = False,
    ) -> torch.Tensor:
        # ====================== your codes here ======================
        # Attention is one fused call rather than four ops, and never
        # materializes the full [B,H,S,S] score matrix the way the baseline's
        # manual version does.
        batch, seq_len, d_model = x.shape

        # One GEMM for all three projections instead of three narrow ones. The
        # weights are concatenated along dim 0, so the output columns are
        # [q | k | v] and 3 is the slowest-varying axis of the split below --
        # the view and permute only relabel strides, they move no data. The
        # three results are non-contiguous views, exactly as .transpose(1, 2)
        # left them, so nothing downstream sees a new layout.
        qkv_w, qkv_b = self._get_qkv_weight()
        qkv = F.linear(x, qkv_w, qkv_b)  # [B, S, 3*d_model]
        q, k, v = (
            qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)      # [3, B, H, S, head_dim]
            .unbind(0)
        )

        # An all-valid mask (the default --padding-ratio 0) still costs a real
        # tensor and blocks the faster is_causal-only path below, so it is
        # treated as "no mask". mask_is_trivial is computed once per forward
        # pass by the caller, not re-derived per layer.
        use_mask = valid_token_mask is not None and not mask_is_trivial

        # The bool mask means the OPPOSITE of masked_fill's: True = allowed to
        # attend, not blocked.
        #
        # Padding stays a [B, 1, 1, S] key mask and causal stays a flag, even
        # when both apply. Folding the triangle into the mask -- which SDPA
        # forces, since it rejects the pair -- costs a [B, 1, S, S] tensor per
        # layer and, worse, hides the triangle from the kernel: `is_causal` is
        # what stops its key loop at the block's own last row, and a mask it
        # cannot recognise as triangular makes it compute the whole upper half
        # and discard it. Measured on the attention op alone, folded against
        # this pair: 1.18x at seq 128, 1.69x at 512, 1.85x at 1024.
        attn_mask = valid_token_mask[:, None, None, :] if use_mask else None
        is_causal = causal

        # Already [B, S, d_model] -- see _attention_dispatch. No transpose or
        # reshape here any more; the kernel epilogue wrote this layout directly.
        # The cached [S, S] triangle that used to be threaded through here went
        # with the SDPA fallback: only that path could not take the mask and the
        # causal flag together, and it no longer exists.
        context = _attention_dispatch(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale,
        )
        output = self.out_proj(context)

        # An all-valid mask makes ~mask all-False, so this masked_fill writes
        # nothing -- but it is out-of-place, so it still clones the whole tensor
        # first. Skipping it when the mask is trivial is a no-op on the output.
        if valid_token_mask is not None and not mask_is_trivial:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
        # ============================================================


class MyTransformerBlock(nn.Module):
    """Same submodule names as BaselineTransformerBlock -> free strict weight loading."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, causal: bool) -> None:
        super().__init__()
        self.norm1 = MyLayerNorm(d_model)
        self.attention = MySelfAttention(d_model, num_heads)
        self.norm2 = MyLayerNorm(d_model)
        self.ffn_in = MyLinear(d_model, ffn_dim)
        self.ffn_out = MyLinear(ffn_dim, d_model)
        self.causal = causal  # stored once, not threaded through every forward call

    def forward(
        self,
        x: torch.Tensor,
        normed: torch.Tensor,
        next_norm: "MyLayerNorm",
        valid_token_mask: Optional[torch.Tensor] = None,
        mask_is_trivial: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (residual stream, next_norm applied to it).

        Both of this block's residual adds feed a LayerNorm, so both are fused
        -- but only one of those norms belongs to this block. `normed` is
        norm1(x), already computed by whoever produced x; `next_norm` is the
        LayerNorm that will consume this block's output, which lives in the
        *next* block (or is final_norm). A block cannot see either of those, so
        the caller supplies them. See UserOptimizedTransformer.forward.
        """
        # ====================== your codes here ======================
        attn_out = self.attention(normed, valid_token_mask, self.causal, mask_is_trivial)

        # Everything from here to the end of the block is row-local, so on the
        # unmasked path it collapses into one kernel. Tried first because when
        # it applies it replaces all four calls below, not just one.
        if valid_token_mask is None or mask_is_trivial:
            fused = _ffn_block(x, attn_out, self.norm2, self.ffn_in,
                               self.ffn_out, next_norm)
            if fused is not None:
                return fused

        x, normed = _add_layernorm(x, attn_out, self.norm2)

        ffn_out = self.ffn_out(_linear_gelu(normed, self.ffn_in))

        if valid_token_mask is not None and not mask_is_trivial:
            # Padded rows are zeroed *between* the add and the norm, so this
            # pair cannot fuse -- the reference normalises the zeroed rows, not
            # the raw sum, and matching that matters more than one kernel.
            x = (x + ffn_out).masked_fill(~valid_token_mask[..., None], 0)
            return x, next_norm(x)

        return _add_layernorm(x, ffn_out, next_norm)
        # ============================================================
