"""The optimized transformer itself."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import config
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

        # Rows per chunk that last fit. Plain attributes for the same reason as
        # the caches above -- out of state_dict(), so strict weight copying keeps
        # working. Caching pays the failed whole-batch attempt once per shape and
        # keeps the chunk size, which moves the numerics, stable across calls.
        self._chunk_rows: Dict[Tuple, int] = {}
        self._device_budget: Optional[int] = None

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
                x, normed, next_norm, valid_token_mask, not use_mask,
                # The last block's next_norm is final_norm, whose output this
                # function returns -- so that one may not be narrowed.
                index == last,
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

        if config.MICROBATCH_FALLBACK and x.is_cuda and x.shape[0] > 1:
            return self._forward_chunk_on_oom(x, valid_token_mask, use_mask)
        return self._forward_eager(x, valid_token_mask, use_mask)
        # ============================================================

    def _rows_within_budget(self, x: torch.Tensor) -> int:
        """Rows whose predicted peak fits the budget; x.shape[0] if all do. One
        multiply and a compare -- the device total is read once and cached.
        """
        rows = x.shape[0]
        per_row = (
            x.shape[1] * x.shape[2] * x.element_size() * config._MICROBATCH_PEAK_FACTOR
        )
        if per_row <= 0:
            return rows

        budget = self._device_budget
        if budget is None:
            _, total = torch.cuda.mem_get_info(x.device)
            budget = int(total * config._MICROBATCH_BUDGET_FRACTION)
            self._device_budget = budget

        if per_row * rows <= budget:
            return rows
        return max(config._MICROBATCH_MIN_ROWS, min(rows, budget // per_row))

    def _forward_chunk_on_oom(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        use_mask: bool,
    ) -> torch.Tensor:
        """_forward_eager, retried over batch chunks if it runs out of memory.

        Eager path only, deliberately: an OOM *during* graph capture leaves the
        capture broken rather than merely failed. `use_mask` is threaded through
        rather than re-derived, so per-chunk mask slices cannot miss
        _mask_is_trivial's weakref cache and cost a sync each.
        """
        rows = x.shape[0]
        key = (rows, x.shape[1], x.shape[2], x.dtype, use_mask)
        n = self._chunk_rows.get(key)
        if n is None:
            # A guess the ladder below still corrects, but the only thing that
            # catches a WDDM spill, which crawls instead of raising.
            n = self._rows_within_budget(x)

        if n >= rows:
            try:
                return self._forward_eager(x, valid_token_mask, use_mask)
            except torch.OutOfMemoryError:
                # Without this the retry inherits the fragmentation that caused
                # the OOM and fails at the same size.
                torch.cuda.empty_cache()
                n = rows // 2

        while n >= config._MICROBATCH_MIN_ROWS:
            out = None
            try:
                # In place: torch.cat would hold the pieces and the whole at
                # once, raising the peak just when it must come down.
                out = torch.empty_like(x)
                for lo in range(0, rows, n):
                    hi = min(lo + n, rows)
                    chunk_mask = (
                        valid_token_mask[lo:hi] if valid_token_mask is not None else None
                    )
                    out[lo:hi] = self._forward_eager(x[lo:hi], chunk_mask, use_mask)
                self._chunk_rows[key] = n
                return out
            except torch.OutOfMemoryError:
                del out
                torch.cuda.empty_cache()
                if n <= config._MICROBATCH_MIN_ROWS:
                    raise
                n //= 2

        # Nothing was attempted: let the caller see a real OOM, not an invented one.
        return self._forward_eager(x, valid_token_mask, use_mask)

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
