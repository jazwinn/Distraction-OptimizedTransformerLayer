"""The optimized transformer itself."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .graphs import (_GraphRunner, _capture_graph, _graph_eligible,
                     _graph_key, _graph_note_decline)
from .layers import MyLayerNorm, MyTransformerBlock
from .util import _version_or_none

if TYPE_CHECKING:
    # For the annotation below only. The model just reads attributes off
    # the config, so importing the harness at run time would be a cycle
    # for nothing.
    from torch_transformer_benchmark import TransformerConfig


class OptimizedTransformer(nn.Module):
    """The optimized model.

    Parameter names and submodule names match BaselineTransformer exactly, so
    copy_model_weights() loads one state_dict into either with strict=True.
    The harness mixes this into UserOptimizedTransformer, which is what keeps
    the two isinstance-compatible.

    BaselineTransformer.__init__ is deliberately not called -- every submodule
    here is a replacement for the baseline's, so there is nothing to inherit.
    """

    def __init__(self, config: TransformerConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layers = nn.ModuleList(
            [
                MyTransformerBlock(config.d_model, config.num_heads, config.ffn_dim, config.causal)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = MyLayerNorm(config.d_model)

        # Cache for _mask_is_trivial below. Plain attributes, so they stay out
        # of state_dict() and strict weight copying keeps working.
        self._triv_ref: Optional[weakref.ReferenceType] = None
        self._triv_version: Optional[int] = None
        self._triv: bool = False

        # Captured CUDA graphs, one per (shape, dtype, device, mask mode,
        # backend, impl), plain attributes for the same reason. _graph_denied
        # remembers keys whose capture *failed*, so a failure is paid for once.
        # Ineligibility is not recorded -- it is re-derived every call, since it
        # depends on the CUDA_GRAPH global that scripts flip mid-process.
        self._graphs: Dict[Tuple, _GraphRunner] = {}
        self._graph_denied: set = set()

    def _mask_is_trivial(self, mask: Optional[torch.Tensor]) -> bool:
        """True when every position is valid, so every mask op is a no-op.

        mask.all().item() is a device->host sync: it stalls the pipeline once
        per forward pass and makes CUDA graph capture illegal outright. The
        harness reuses one mask tensor for a whole run, so caching costs one
        sync during warmup and none in the measured region.

        Keyed on a weakref, not data_ptr(): the caching allocator recycles
        addresses, so a freed mask and an unrelated new one can share a pointer
        and the cache would serve a stale answer. A live weakref proves the
        object was never freed; the version is a best-effort mutation check on
        top of that.
        """
        if mask is None:
            return False

        version = _version_or_none(mask)
        cached = self._triv_ref() if self._triv_ref is not None else None
        if cached is mask and self._triv_version == version:
            return self._triv

        self._triv = bool(mask.all().item())
        self._triv_ref = weakref.ref(mask)
        self._triv_version = version
        return self._triv

    def _forward_eager(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        use_mask: bool,
    ) -> torch.Tensor:
        """The whole model, with every host-side decision already made.

        Split out of forward() so it can be captured into a CUDA graph, which
        makes this contract narrower than an ordinary method's:

          * No device->host reads -- no .item(), .cpu(), .tolist(), no `if` on a
            device tensor, no print. Any one makes capture illegal, and
            _mask_is_trivial's mask.all().item() is why the mask decision is
            taken by the caller and handed down as `use_mask`.
          * No lazy allocation of anything that outlives the call: a cache first
            built in here would land in the graph's private pool and then be
            read by the *eager* path too. _capture_graph's warmup iterations
            force those allocations to happen outside first.

        A violation does not pass silently -- _capture_graph replays against an
        eager reference and refuses a graph whose output differs.

        `use_mask` rather than mask_is_trivial because every mask-dependent
        branch below (and in MySelfAttention and MyTransformerBlock) is spelled
        `valid_token_mask is not None and not mask_is_trivial`. Passing the
        reduced predicate makes the graph cache key *be* the flag that selects
        the body, so the two cannot drift apart.
        """
        # Every LayerNorm in the model consumes the output of a residual add, so
        # every one of them fuses into that add -- except the very first, which
        # has no add before it. That one runs on its own here; the rest are
        # handed to the block whose add produces their input.
        normed = self.layers[0].norm1(x)
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            next_norm = self.final_norm if index == last else self.layers[index + 1].norm1
            x, normed = layer(
                x, normed, next_norm, valid_token_mask, not use_mask
            )

        # The last block folded final_norm into its trailing add, so `normed` is
        # already final_norm(x) and there is no separate call to make here.
        if use_mask:
            normed = normed.masked_fill(~valid_token_mask[..., None], 0)
        return normed

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ====================== your codes here ======================
        # Once per forward pass rather than once per layer (6x redundant
        # GPU->CPU syncs otherwise), and memoized across calls on top of that,
        # so the steady state has no sync at all. This is the only device->host
        # sync in the forward pass, and it stays outside the capturable region.
        mask_is_trivial = self._mask_is_trivial(valid_token_mask)
        use_mask = valid_token_mask is not None and not mask_is_trivial

        # Steady state on the graph path is a tuple build, a dict hit and a
        # method call. The lookup is inlined rather than behind a helper so the
        # hot path is a lookup, not a call that does a lookup.
        key = _graph_key(x, use_mask)
        runner = self._graphs.get(key)
        if runner is None:
            runner = self._maybe_capture(key, x, valid_token_mask, use_mask)
        if runner is not None:
            return runner.replay(x, valid_token_mask)

        return self._forward_eager(x, valid_token_mask, use_mask)
        # ============================================================

    def _maybe_capture(
        self,
        key: Tuple,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        use_mask: bool,
    ) -> Optional[_GraphRunner]:
        """Capture a graph for this key, or decide not to.

        Three reasons to end up eager, kept separate on purpose -- collapsing
        them would make a permanent decline look transient, or make a
        mid-process CUDA_GRAPH change silently do nothing:

          * the key failed capture before          -> _graph_denied, permanent
          * the shape/mode is not eligible now     -> re-derived every call
          * capture declined or failed this time   -> recorded as a denial
        """
        if key in self._graph_denied:
            return None
        if not _graph_eligible(x, len(self._graphs)):
            _graph_note_decline(x)
            return None

        runner = _capture_graph(self, x, valid_token_mask, use_mask)
        if runner is None:
            self._graph_denied.add(key)
            return None

        self._graphs[key] = runner
        return runner
