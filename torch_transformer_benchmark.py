      
#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import atexit
import copy
import math
import os
import statistics
import time
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


# Attention backend. --attn-backend overrides this for a single run.
#
#   "auto"     custom CUDA kernel when it builds and loads, else SDPA
#   "sdpa"     always F.scaled_dot_product_attention. No build required.
#   "custom"   require the custom kernel, so a broken build fails loudly
#              instead of quietly benchmarking the fallback and looking slow
ATTENTION_BACKEND = "auto"

# Which kernel inside the extension handles attention; only meaningful when the
# custom backend is in play. --attn-impl overrides this for a single run.
#
#   "auto"       tensor-core kernel where it applies, scalar kernel otherwise
#   "scalar"     force the scalar kernel (no tensor cores, no TF32 rounding)
#   "wmma"       force the tensor-core kernel; raises on shapes it misses
#   "tile"       force the cuTile kernel, fp32 operands: exact, CUDA cores.
#                float32 and head_dim in {8,16,32,64}, and needs a build that
#                found CUDA 13.3+. Never picked by "auto".
#   "tile-tf32"  the cuTile kernel with its GEMMs narrowed to tf32, which is
#                what puts them on the tensor cores. Same arithmetic cuBLAS
#                gives the baseline under allow_tf32 (~1e-3), so this is the
#                tensor-core mode to reach for first.
#   "tile-bf16"  as above, narrowed to bfloat16 -- 8 mantissa bits, ~4e-3.
#                Expect it to fail the accuracy gate where "tile" passes.
ATTENTION_IMPL = "auto"

_IMPL_CODE = {"auto": 0, "scalar": 1, "wmma": 2, "tile": 3, "tile-bf16": 4,
              "tile-tf32": 5}

# Ask the kernel for [B, S, H*head_dim] rather than [B, H, S, head_dim].
# out_proj wants the flattened layout, and the transpose+reshape that used to
# produce it could not be a view -- it repacked the whole tensor once per layer.
# The kernel epilogue reaches the same addresses for free.
_OUT_LAYOUT_BSHD = 1

# CUDA graph capture. --cuda-graph overrides this for a single run.
#
#   "off"     never capture; every forward pass launches its ~79 kernels
#   "auto"    capture when launch overhead dominates and the graph's pinned
#             activation volume batch*seq*d_model is at or under
#             _GRAPH_MAX_ACTIVATION
#   "always"  capture regardless of size, to measure where the crossover
#             actually is. Pins the whole working set; not for benchmark runs.
#
# Launches are asynchronous, so the CPU runs ahead queueing kernel n+1 while the
# GPU works on n: when the average kernel outlasts the time to issue one, launch
# cost is invisible and a graph buys nothing, and when it does not, the GPU
# starves. At batch 1 seq 32 the mean kernel is 6.0 us and the GPU idles 80% of
# the wall clock (4.2x from replay); at batch 8 seq 2048 it is 1018 us, idles
# 0.9%, and a graph is worth nothing. Kernel *count* does not predict this --
# both shapes issue ~79-91 -- so the gate is on activation volume, not on
# launches; see _GRAPH_MAX_ACTIVATION, including how to re-derive it elsewhere.
#
# Replay is bit-identical to eager, not merely close -- same kernels, same
# order, same addresses. _capture_graph verifies that rather than assuming it.
#
# A capture freezes more than it looks like: the kernel chosen for each op, the
# cuBLAS algorithm, allow_tf32/matmul precision, and the extension's own runtime
# knobs (tile_set_split_kv). Only ATTENTION_BACKEND and ATTENTION_IMPL are in
# the cache key, so do not change the rest after the first forward pass -- a
# captured model will quietly ignore you.
CUDA_GRAPH = "auto"

# Elements in one activation tensor -- batch * seq * d_model -- at or below which
# "auto" captures. Set from scripts/ab_graph.py, which measured all three axes
# rather than assuming the obvious one mattered:
#
#   batch*seq*d_model    measured ratio (best of 5, control +/-1.2%)
#   ----------------     -------------------------------------------
#      16384 (b1 s32)        4.23x
#      65536                 2.41x - 3.26x
#     262144                 1.02x - 1.32x
#     524288                 1.029x, 1.038x    <- this gate
#    1048576 and up          1.01x or less, i.e. inside the noise
#
# ---------------------------------------------------------------------------
# THIS NUMBER WAS MEASURED ON AN RTX 3070 (8 GiB, SM 8.6). On other hardware,
# re-derive it:
#
#     cmd.exe /c scripts\devenv.bat python scripts\ab_graph.py
#
# and read off where the ratio column drops into the control column. The
# crossover is the point where the GPU stops starving, which depends on the
# card's throughput relative to how fast the host can feed it -- so a faster GPU
# starves at larger shapes and wants a LARGER value here, and a busier host wants
# a larger one too.
#
# Getting it wrong is cheap in both directions. Too low costs some latency on
# shapes that would have benefited; too high costs some pinned memory on shapes
# that do not. Neither can produce a wrong answer, because replay is bit-identical
# to eager whatever this is set to.
# ---------------------------------------------------------------------------
#
# Why activation volume and not tokens, which is the obvious choice: tokens
# mispredict badly. At 512 tokens, d_model 256 gave 2.708x and d_model 512 gave
# 1.036x -- the same token count, a 2.6x difference in payoff. Work per kernel
# scales with the activation tensor rather than with its rows, and a pure token
# gate of 1024 would have declined batch 8 seq 256 at d_model 256, measured at
# 1.038x.
#
# num_layers does not belong here at all. At fixed activation volume, 3/6/12/24
# layers gave 1.031x/1.037x/1.040x/1.040x -- eager and replay scale with depth
# together.
#
# Nothing swept was ever *slower* than eager (worst case 0.998x, inside the
# control's spread), so this gate is not protecting against a slowdown. It caps
# the pinned pool, which at this threshold measured about 84 MiB.
_GRAPH_MAX_ACTIVATION = 1 << 19    # 524288

# Safety net, deliberately not the gate: a captured pool larger than this share
# of the card is released rather than held for the whole run. At the threshold
# above it should never fire -- 84 MiB against 2 GiB on an 8 GiB card -- and that
# is the point. It is there so that a much larger _GRAPH_MAX_ACTIVATION set on
# unfamiliar hardware, or a pool that turns out bigger than it was here, degrades
# to eager instead of quietly eating the card.
_GRAPH_POOL_SAFETY_FRACTION = 0.25

# Each captured graph holds its own private memory pool for the whole run, so
# this caps memory, not time. One shape and one mask mode is the normal case; 4
# leaves room for a mask-mode flip without letting a caller capture dozens.
_GRAPH_MAX_ENTRIES = 4

# Iterations before capture, the count torch's own make_graphed_callables uses.
# One would populate every Python-level cache; three is cheap insurance against
# anything initializing on a second or third touch (cuBLAS heuristic caches,
# per-kernel lazy module loading).
_GRAPH_WARMUP_ITERS = 3

# Refuse to capture with less than this much device memory free. Cheaper than
# discovering it by OOM, though OOM is handled too.
_GRAPH_MIN_FREE_BYTES = 512 << 20

# Capture feeds the model its own data rather than the caller's, from a local
# generator so it cannot perturb the global RNG stream main() seeds.
_GRAPH_SEED = 20240827

_fallback_warned = False
_graph_warned = False
_graph_noted = False
_graph_declined_noted = False


def _custom_kernels():
    """The extension module, or None when unavailable or switched off.

    ATTENTION_BACKEND == "sdpa" means "no custom kernels at all", not just no
    custom attention -- otherwise --attn-backend sdpa would still be timing
    hand-written code and the comparison would mean nothing.
    """
    if ATTENTION_BACKEND == "sdpa":
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

    if ATTENTION_BACKEND != "sdpa":
        import kernel_ext

        kernels = kernel_ext.get_kernels()
        if kernels is not None:
            return kernels.fused_attention_forward(
                q, k, v, attn_mask, is_causal, scale,
                _IMPL_CODE[ATTENTION_IMPL], _OUT_LAYOUT_BSHD
            )
        if ATTENTION_BACKEND == "custom":
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
                f"python {os.path.basename(__file__)}\n"
                f'[info] to silence this:    set ATTENTION_BACKEND = "sdpa" '
                f"at the top of this file"
            )

    context = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
    # [B, H, S, D] -> [B, S, H*D]. Cannot be a view across the transpose, so
    # this is the strided repack the custom kernels skip. flatten(2) rather than
    # reshape(b, s, h*d) keeps the shape arithmetic on the C++ side, which is
    # not free in a model this close to launch-bound.
    return context.transpose(1, 2).flatten(2)


class _GraphRunner:
    """One captured CUDA graph, plus the buffers whose addresses it baked in.

    A graph records kernel launches against *addresses*, not against tensors.
    Everything this object holds is therefore load-bearing: drop any of it and
    the graph replays into freed memory.
    """

    __slots__ = ("graph", "stream", "static_x", "static_mask", "static_out",
                 "needs_mask", "replays", "pinned", "__weakref__")

    def __init__(self, graph, stream, static_x, static_mask, static_out,
                 needs_mask, pinned):
        self.graph = graph            # keeps the exec graph and its private pool alive
        self.stream = stream          # see the cuBLAS note in _capture_graph
        self.static_x = static_x
        self.static_mask = static_mask
        self.static_out = static_out  # lives in the private pool; holding it is mandatory
        self.needs_mask = needs_mask
        self.pinned = pinned          # see the note on `pinned` below
        self.replays = 0

    def replay(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Run the graph on new inputs. Bit-identical to _forward_eager.

        Three launches at most -- two copies and the graph -- against the ~79
        the eager path issues. A graph captured with use_mask=False holds no
        reference to static_mask at all, so there is no mask copy to make.

        Deliberately not done here: comparing data_ptr() to skip a copy whose
        source is unchanged. It would save ~2 us of a 488 us call, but
        _version_or_none returns None for inference tensors and the
        accuracy-phase x *is* one, so the only guard against in-place mutation
        of a reused buffer degrades to identity-only in exactly the phase where
        correctness is being checked.
        """
        self.static_x.copy_(x)
        if self.needs_mask:
            self.static_mask.copy_(mask)
        self.graph.replay()
        self.replays += 1
        # Cloned, not returned directly: the next replay overwrites static_out
        # in place, so handing it out would make correctness depend on the
        # caller never holding a result across a call. It also matches eager's
        # return type -- clone() inside the caller's inference_mode yields an
        # inference tensor, where static_out carries autograd metadata.
        return self.static_out.clone()

    def release(self) -> None:
        """Give the private memory pool back.

        All three steps are required: the pool survives until the CUDAGraph is
        reset *and* every tensor allocated from it is unreferenced *and*
        empty_cache() runs. Dropping the graph alone does nothing.
        """
        try:
            self.graph.reset()
        except Exception:  # noqa: BLE001 - reset on a half-captured graph may throw
            pass
        self.graph = None
        self.stream = None
        self.static_x = None
        self.static_mask = None
        self.static_out = None
        self.pinned = None
        torch.cuda.empty_cache()


def _graph_key(x: torch.Tensor, use_mask: bool) -> Tuple:
    """What makes two calls interchangeable to one captured graph.

    ATTENTION_BACKEND/ATTENTION_IMPL are in the key because a capture freezes
    whichever kernel they selected; without them, an in-process A/B script
    flipping the global would silently do nothing on a captured model.

    Strides are deliberately absent: inputs are copied into a freshly allocated
    contiguous buffer and copy_ handles any source layout, so one graph is valid
    for every layout of x.
    """
    return (x.shape, x.dtype, x.device.index, use_mask,
            ATTENTION_BACKEND, ATTENTION_IMPL)


def _graph_pool_cap_bytes(device: torch.device) -> int:
    """Largest private pool worth holding for a whole run on this device."""
    total = torch.cuda.get_device_properties(device).total_memory
    return int(total * _GRAPH_POOL_SAFETY_FRACTION)


def _graph_eligible(x: torch.Tensor, entries: int) -> bool:
    """Cheap per-call gate. Pure Python comparisons, no GPU calls, no syncs.

    Deliberately not memoized: the verdict depends on the CUDA_GRAPH global,
    which scripts flip mid-process, so caching a False from an "off" pass would
    make a later "always" pass do nothing. Only capture *failures* are
    remembered, in the model's _graph_denied set.
    """
    if CUDA_GRAPH == "off":
        return False
    if x.device.type != "cuda" or x.dim() != 3:
        return False
    # torch.compile's reduce-overhead mode already captures graphs; nesting a
    # capture inside Dynamo's, or inside anyone else's, is not valid.
    if torch.compiler.is_compiling():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if entries >= _GRAPH_MAX_ENTRIES:
        return False
    if CUDA_GRAPH == "always":
        return True
    return x.shape[0] * x.shape[1] * x.shape[2] <= _GRAPH_MAX_ACTIVATION


def _graph_note_decline(x: torch.Tensor) -> None:
    """Say once why "auto" is not capturing, so the eager path is not a mystery.

    The eligibility check is silent by design -- it runs on every call -- but a
    run that quietly declines looks identical to a run where the feature is
    broken. One line, once, is the difference between "graphs are off for this
    shape because the pool would not fit" and a user wondering why the speedup
    they read about did not appear.
    """
    global _graph_declined_noted
    if _graph_declined_noted or CUDA_GRAPH == "off":
        return
    if x.device.type != "cuda" or torch.compiler.is_compiling():
        return
    _graph_declined_noted = True
    activation = x.shape[0] * x.shape[1] * x.shape[2]
    if activation > _GRAPH_MAX_ACTIVATION:
        print(f"[info] CUDA graph declined for shape {tuple(x.shape)}: "
              f"batch*seq*d_model is {activation}, over the "
              f"{_GRAPH_MAX_ACTIVATION} above which replay measured no gain on "
              f"this hardware. Running eagerly; results are unaffected.\n"
              f"[info] to capture anyway:  --cuda-graph always\n"
              f"[info] to re-measure the threshold for this machine:  "
              f"python scripts/ab_graph.py")
    else:
        print(f"[info] CUDA graph declined for shape {tuple(x.shape)}, running "
              f"eagerly (results are unaffected).")


def _capture_graph(
    model: "UserOptimizedTransformer",
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    use_mask: bool,
) -> Optional[_GraphRunner]:
    """Capture model._forward_eager, or return None and leave the run eager.

    Returns None for every recoverable reason: too little memory, a build
    failure, an illegal op in the body, OOM, or a replay that does not match
    eager exactly. The one thing it will not do is return None after a failure
    that left the CUDA context unusable -- see the health probe at the bottom.
    """
    global _graph_noted, _graph_warned

    dev = x.device
    g = None
    static_x = static_mask = static_out = ref = None

    try:
        # -- Step 0: everything that lazily initializes, forced to happen here.
        #
        # Not redundant with the warmup below: doing it before any stream is
        # touched means a *build* failure is a plain None return rather than an
        # exception thrown out of a capture region. get_kernels() on its first
        # call runs ninja and dlopens the result.
        _custom_kernels()
        torch.cuda.get_device_properties(dev)   # torch's cudaDeviceProp cache

        free_bytes, _ = torch.cuda.mem_get_info(dev)
        if free_bytes < _GRAPH_MIN_FREE_BYTES:
            return None                         # quiet decline; not a failure

        # Drain the allocator's cache *before* the baseline reading: entering
        # torch.cuda.graph() calls empty_cache() itself, so without this the
        # reported pool size is (pool - whatever cache got freed), which can
        # come out negative and mean nothing.
        torch.cuda.empty_cache()
        reserved_before = torch.cuda.memory_reserved(dev)

        # -- Step 1: static buffers, as *normal* tensors.
        #
        # inference_mode(False) so they are not inference tensors -- one cannot
        # be updated in place from outside inference mode, which would make them
        # unrefillable. torch.empty(x.shape, ...) rather than empty_like(x)
        # because x may be an inference tensor and passing it as an *operand*
        # out here is the one grey area; reading .shape/.dtype is metadata, so
        # this routine does not care what mode the caller was in.
        with torch.inference_mode(False):
            static_x = torch.empty(x.shape, dtype=x.dtype, device=dev)
            static_mask = (
                torch.empty(mask.shape, dtype=torch.bool, device=dev)
                if use_mask else None
            )

            # -- Step 2: seed with our own data, never the caller's.
            #
            # Removes the last reason to touch x or mask, and gives the
            # self-check below a known-nontrivial mask rather than whatever the
            # caller brought (a low padding ratio can produce rows with no False
            # in them at all). A local generator, so this cannot consume the
            # global RNG stream main() seeds.
            gen = torch.Generator(device=dev)
            gen.manual_seed(_GRAPH_SEED)
            static_x.normal_(generator=gen)
            if use_mask:
                static_mask.fill_(True)
                static_mask[:, static_mask.shape[1] // 2:] = False

            with torch.no_grad():
                # no_grad, not just inference_mode(False): parameters have
                # requires_grad=True and the model is only .eval()ed, so
                # escaping inference mode without this would build an autograd
                # graph during capture and pin its saved tensors in the private
                # pool. A no-op if the caller was already in inference mode.

                # -- Step 3: warmup, on the stream we are about to capture on.
                #
                # This is what keeps lazily-built persistent state out of the
                # graph's private pool: _get_qkv_weight's torch.cat and
                # _get_causal_mask's ones().tril() get built here, by the
                # general allocator. Built inside the capture they would live in
                # the pool, and the *eager* fallback path would then be reading
                # graph-pool memory -- the most dangerous failure this feature
                # can have.
                #
                # Same stream as the capture because cuBLAS caches a workspace
                # per (handle, stream): warming this stream first puts that
                # workspace in the general pool.
                side = torch.cuda.Stream(device=dev)
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):
                    for _ in range(_GRAPH_WARMUP_ITERS):
                        model._forward_eager(static_x, static_mask, use_mask)
                torch.cuda.current_stream(dev).wait_stream(side)
                torch.cuda.synchronize(dev)

                # Snapshot after the warmup: the addresses the graph is about
                # to bake in. Compared again after capture, so "a cache was
                # rebuilt inside the capture" is caught rather than hoped
                # against.
                ptrs_before = _graph_cache_ptrs(model)

                # -- Step 4: the eager reference for the self-check.
                # Cloned because the uncloned result is a general-pool tensor
                # whose block gets reused during capture.
                ref = model._forward_eager(static_x, static_mask, use_mask).clone()

                # -- Step 5: capture.
                #
                # No pool=, so each key gets its own: sharing one would save
                # memory and introduce a cross-graph aliasing hazard, in a
                # feature whose whole justification is bit-exactness.
                # _GRAPH_MAX_ENTRIES is the memory answer instead.
                #
                # capture_error_mode stays at its "global" default; "relaxed"
                # would let an illegal op through and produce a silently wrong
                # graph.
                #
                # Entering this context calls torch.cuda.synchronize() and
                # empty_cache(), which drops the *baseline* model's cached
                # blocks too -- harmless during the accuracy phase, since
                # warmup_model reruns before anything is timed, but it is why
                # capture must never be deferred into the benchmark phase.
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=side):
                    static_out = model._forward_eager(static_x, static_mask, use_mask)

                # -- Step 6: prove the graph computes what eager computes.
                g.replay()
                torch.cuda.synchronize(dev)
                delta = (static_out.float() - ref.float()).abs().max().item()
                ptrs_after = _graph_cache_ptrs(model)

        if ptrs_after != ptrs_before:
            _graph_release(g)
            if not _graph_warned:
                _graph_warned = True
                print("[info] CUDA graph capture rebuilt a weight cache inside the "
                      "capture region, so it was discarded and this run stays eager "
                      "(results are still correct). This is a bug in the warmup, not "
                      "in your inputs.")
            return None

        if delta != 0.0:
            # A replay that differs from eager is the one failure mode a
            # benchmark would happily report as a *win*, so it is fatal to the
            # graph rather than tolerated. Nothing should produce a nonzero
            # here, so the number is worth printing.
            _graph_release(g)
            if not _graph_warned:
                _graph_warned = True
                print(f"[info] CUDA graph replay differs from eager by "
                      f"{delta:.3e}, so the graph was discarded and this run "
                      f"stays eager (results are still correct). Expected "
                      f"exactly 0; a nonzero value means the captured kernels "
                      f"are not the eager kernels.")
            return None

        # Safety net on the real pool size; see _GRAPH_POOL_SAFETY_FRACTION. It
        # should never fire at the default threshold. "always" is exempt, since
        # its entire purpose is measuring the shapes auto declines.
        pool_bytes = torch.cuda.memory_reserved(dev) - reserved_before
        budget = _graph_pool_cap_bytes(dev)
        if CUDA_GRAPH != "always" and pool_bytes > budget:
            _graph_release(g)
            if not _graph_warned:
                _graph_warned = True
                print(f"[info] CUDA graph for shape {tuple(x.shape)} reserved "
                      f"{pool_bytes / (1 << 20):.0f} MiB of pinned pool, over the "
                      f"{budget / (1 << 20):.0f} MiB cap "
                      f"({_GRAPH_POOL_SAFETY_FRACTION:.0%} of the card), so it was "
                      f"released and this shape runs eagerly. If this fires, "
                      f"_GRAPH_MAX_ACTIVATION is too high for this card.")
            return None

        runner = _GraphRunner(
            graph=g, stream=side, static_x=static_x, static_mask=static_mask,
            static_out=static_out, needs_mask=use_mask,
            # Strong references to every per-layer cache whose address the
            # graph baked in. If _get_qkv_weight rebuilt one, the old buffer
            # would be freed and the graph would read whatever the allocator
            # handed out next -- silent garbage. Holding them downgrades that to
            # reading the old, still-live values, which are the same values
            # since weights never move after copy_model_weights. Cheaper than
            # re-checking pointers on every replay.
            pinned=_graph_pinned_refs(model),
        )
        _graph_register(runner)

        if not _graph_noted:
            _graph_noted = True
            pool_mib = (torch.cuda.memory_reserved(dev) - reserved_before) / (1 << 20)
            print(f"[info] CUDA graph captured: shape={tuple(x.shape)} "
                  f"{str(x.dtype).replace('torch.', '')} mask="
                  f"{'on' if use_mask else 'off'} backend={ATTENTION_BACKEND} "
                  f"impl={ATTENTION_IMPL}, pool +{pool_mib:.1f} MiB, "
                  f"replay matches eager exactly")
        return runner

    except Exception as exc:  # noqa: BLE001
        # Not BaseException: capture takes a second or two, and swallowing
        # KeyboardInterrupt around it is just obstructive.
        _graph_release(g)
        del static_x, static_mask, static_out, ref

        # torch.cuda.graph.__exit__ calls capture_end() unconditionally, so an
        # exception from the body propagates *through* it and can leave the
        # stream capturing. Every subsequent CUDA op then fails, including the
        # baseline's, and a run that "recovered" would print garbage for both
        # sides. Check rather than assume.
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("stream is still capturing")
            torch.cuda.synchronize(dev)
            torch.empty(1, device=dev).fill_(1.0)
            torch.cuda.synchronize(dev)
        except Exception as fatal:  # noqa: BLE001
            raise RuntimeError(
                "CUDA graph capture failed and left the CUDA context unusable, so "
                "every number printed after this point would be meaningless. "
                f"Re-run with --cuda-graph off. Original failure: {exc}"
            ) from fatal

        torch.cuda.empty_cache()
        if not _graph_warned:
            _graph_warned = True
            print(f"[info] CUDA graph capture failed for shape {tuple(x.shape)}, "
                  f"running eagerly instead (results are still correct). "
                  f"Reason: {type(exc).__name__}: {exc}\n"
                  f'[info] to silence this:    set CUDA_GRAPH = "off" at the top '
                  f"of this file")
        return None


def _graph_cache_ptrs(model: "UserOptimizedTransformer") -> Tuple:
    """Addresses of the lazily-built per-layer caches a graph bakes in."""
    out = []
    for layer in model.layers:
        attn = layer.attention
        for t in (attn._qkv_weight, attn._qkv_bias, attn._causal_mask):
            out.append(None if t is None else t.data_ptr())
    return tuple(out)


def _graph_pinned_refs(model: "UserOptimizedTransformer") -> Tuple:
    """Strong references keeping those caches alive for the graph's lifetime."""
    out = []
    for layer in model.layers:
        attn = layer.attention
        out.extend((attn._qkv_weight, attn._qkv_bias, attn._causal_mask))
    return tuple(out)


def _graph_release(g) -> None:
    if g is not None:
        try:
            g.reset()
        except Exception:  # noqa: BLE001
            pass


# Every live graph, weakly held, so they can be torn down while CUDA is still
# up. A CUDAGraph destroyed during interpreter finalization -- after CUDA has
# de-initialized -- takes the process out with an access violation (0xC0000005
# on Windows) *after* main() has returned its exit code, so it looks like a
# passing run with a mysterious exit code. The harness keeps one graph and
# survives; a script holding several to shutdown does not.
_graph_live: "weakref.WeakSet" = weakref.WeakSet()
_graph_atexit_registered = False


def _graph_register(runner: _GraphRunner) -> None:
    global _graph_atexit_registered
    _graph_live.add(runner)
    if not _graph_atexit_registered:
        _graph_atexit_registered = True
        atexit.register(_graph_release_all)


def _graph_release_all() -> None:
    """Drop every surviving graph. Registered with atexit, and safe to re-run."""
    for runner in list(_graph_live):
        try:
            runner.release()
        except Exception:  # noqa: BLE001 - shutdown path; nothing left to report to
            pass
    _graph_live.clear()


def _version_or_none(t: torch.Tensor) -> Optional[int]:
    """t._version, or None when the tensor does not track one.

    Inference tensors raise instead of reporting a version, and both cache keys
    below can be handed one -- the accuracy path builds its mask inside
    inference_mode(). None makes the version half of a key best-effort; the
    identity half still carries correctness.
    """
    try:
        return t._version
    except RuntimeError:
        return None


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

        # Lazily-built causal mask cache. seq_len is fixed for a model
        # instance's lifetime here, so rebuilding it 6x per forward pass was
        # waste. Plain attributes, not buffers, so they stay out of
        # state_dict() and strict weight copying keeps working.
        self._causal_mask_key: Optional[Tuple] = None
        self._causal_mask: Optional[torch.Tensor] = None

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

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if self._causal_mask_key != key:
            self._causal_mask = torch.ones(
                seq_len, seq_len, device=device, dtype=torch.bool
            ).tril()
            self._causal_mask_key = key
        return self._causal_mask

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

        attn_mask = None
        is_causal = False
        if use_mask:
            # SDPA's bool mask means the OPPOSITE of masked_fill's:
            # True = allowed to attend, not blocked.
            key_mask = valid_token_mask[:, None, None, :]
            if causal:
                causal_allowed = self._get_causal_mask(seq_len, x.device)
                attn_mask = key_mask & causal_allowed  # [B, 1, S, S]
            else:
                attn_mask = key_mask  # [B, 1, 1, S], broadcasts over queries
        else:
            # is_causal and attn_mask can't be combined -- SDPA rejects that --
            # so this path only applies once we know there's no padding to fold in.
            is_causal = causal

        # Already [B, S, d_model] -- see _attention_dispatch. No transpose or
        # reshape here any more; the kernel epilogue wrote this layout directly.
        context = _attention_dispatch(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
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
        x, normed = _add_layernorm(x, attn_out, self.norm2)

        ffn_out = self.ffn_out(F.gelu(self.ffn_in(normed), approximate="none"))

        if valid_token_mask is not None and not mask_is_trivial:
            # Padded rows are zeroed *between* the add and the norm, so this
            # pair cannot fuse -- the reference normalises the zeroed rows, not
            # the raw sum, and matching that matters more than one kernel.
            x = (x + ffn_out).masked_fill(~valid_token_mask[..., None], 0)
            return x, next_norm(x)

        return _add_layernorm(x, ffn_out, next_norm)
        # ============================================================


class UserOptimizedTransformer(BaselineTransformer):
    """
    Replace this class with the optimized implementation.

    Requirements:
      1. Keep the forward signature unchanged.
      2. Return a tensor with shape [batch_size, seq_len, d_model].
      3. Keep compatible parameter names, or customize copy_model_weights().
    """

    def __init__(self, config: TransformerConfig) -> None:
        nn.Module.__init__(self)  # skip BaselineTransformer.__init__ on purpose
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


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    parser.add_argument(
        "--attn-backend",
        choices=("auto", "sdpa", "custom"),
        default=None,
        help="override ATTENTION_BACKEND for this run only (default: use the "
             "value set at the top of this file)",
    )
    parser.add_argument(
        "--attn-impl",
        choices=("auto", "scalar", "wmma", "tile", "tile-bf16", "tile-tf32"),
        default=None,
        help="override ATTENTION_IMPL for this run only: which kernel inside "
             "the custom extension runs attention",
    )
    parser.add_argument(
        "--cuda-graph",
        choices=("off", "auto", "always"),
        default=None,
        help="override CUDA_GRAPH for this run only: capture the optimized "
             "model's forward pass into a CUDA graph and replay it. 'always' "
             "ignores the size gate, for measurement only",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")
    # Warnings rather than errors: the runtime gate already handles both, and
    # raising would make an otherwise reasonable command line fail for nothing.
    if args.cuda_graph in ("auto", "always") and device.type != "cuda":
        print("[warning] --cuda-graph has no effect on a non-CUDA device")
    activation = args.batch_size * args.seq_len * args.d_model
    if args.cuda_graph == "always" and activation > _GRAPH_MAX_ACTIVATION:
        print(f"[warning] --cuda-graph always at {activation} activation elements "
              f"is past the point where replay measured any gain "
              f"({_GRAPH_MAX_ACTIVATION}); it pins the whole working set in the "
              f"graph's private pool for no speedup")


def main() -> int:
    global ATTENTION_BACKEND, ATTENTION_IMPL, CUDA_GRAPH

    args = parse_args()
    if args.attn_backend is not None:
        ATTENTION_BACKEND = args.attn_backend
    if args.attn_impl is not None:
        ATTENTION_IMPL = args.attn_impl
    if args.cuda_graph is not None:
        CUDA_GRAPH = args.cuda_graph

    # --compile-user with reduce-overhead already puts the model behind
    # Inductor's own CUDA graphs, so hand-rolled capture on top is two
    # mechanisms for one job. Asymmetric on purpose: silently stand down when
    # graphs were only the file's default, so --compile-user keeps working
    # unchanged; refuse when both were asked for explicitly, because then the
    # command line is contradictory and picking one silently would hide it.
    if args.compile_user and CUDA_GRAPH != "off":
        if args.cuda_graph is None:
            CUDA_GRAPH = "off"
            print("[info] --compile-user: CUDA_GRAPH forced off, since "
                  "--compile-mode reduce-overhead already captures CUDA graphs")
        else:
            raise ValueError(
                "--compile-user and --cuda-graph are two implementations of the "
                "same optimization; reduce-overhead already captures graphs. "
                "Pick one."
            )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

    