"""CUDA graph capture for the optimized model.

The whole forward pass is recorded once per (shape, dtype, device, mask mode,
backend, impl) and replayed afterwards, which removes ~79 kernel launches per
call. Replay is verified bit-identical to eager before a graph is installed.

Everything in here is a performance choice that must never change a result: on
any doubt at all, capture declines and the caller stays eager.
"""

from __future__ import annotations

import atexit
import weakref
from typing import Optional, Tuple

import torch

from . import config
from .kernels import _custom_kernels

_graph_warned = False
_graph_noted = False
_graph_declined_noted = False


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

    ATTENTION_BACKEND/ATTENTION_IMPL/LINEAR_GELU are in the key because a
    capture freezes whichever kernel they selected; without them, an in-process
    A/B script flipping the global would silently do nothing on a captured
    model.

    Strides are deliberately absent: inputs are copied into a freshly allocated
    contiguous buffer and copy_ handles any source layout, so one graph is valid
    for every layout of x.
    """
    return (x.shape, x.dtype, x.device.index, use_mask,
            config.ATTENTION_BACKEND, config.ATTENTION_IMPL, config.LINEAR_GELU)


def _graph_pool_cap_bytes(device: torch.device) -> int:
    """Largest private pool worth holding for a whole run on this device."""
    total = torch.cuda.get_device_properties(device).total_memory
    return int(total * config._GRAPH_POOL_SAFETY_FRACTION)


def _graph_eligible(x: torch.Tensor, entries: int) -> bool:
    """Cheap per-call gate. Pure Python comparisons, no GPU calls, no syncs.

    Deliberately not memoized: the verdict depends on the CUDA_GRAPH global,
    which scripts flip mid-process, so caching a False from an "off" pass would
    make a later "always" pass do nothing. Only capture *failures* are
    remembered, in the model's _graph_denied set.
    """
    if config.CUDA_GRAPH == "off":
        return False
    if x.device.type != "cuda" or x.dim() != 3:
        return False
    # torch.compile's reduce-overhead mode already captures graphs; nesting a
    # capture inside Dynamo's, or inside anyone else's, is not valid.
    if torch.compiler.is_compiling():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if entries >= config._GRAPH_MAX_ENTRIES:
        return False
    if config.CUDA_GRAPH == "always":
        return True
    return x.shape[0] * x.shape[1] * x.shape[2] <= config._GRAPH_MAX_ACTIVATION


def _graph_note_decline(x: torch.Tensor) -> None:
    """Say once why "auto" is not capturing, so the eager path is not a mystery.

    The eligibility check is silent by design -- it runs on every call -- but a
    run that quietly declines looks identical to a run where the feature is
    broken. One line, once, is the difference between "graphs are off for this
    shape because the pool would not fit" and a user wondering why the speedup
    they read about did not appear.
    """
    global _graph_declined_noted
    if _graph_declined_noted or config.CUDA_GRAPH == "off":
        return
    if x.device.type != "cuda" or torch.compiler.is_compiling():
        return
    _graph_declined_noted = True
    activation = x.shape[0] * x.shape[1] * x.shape[2]
    if activation > config._GRAPH_MAX_ACTIVATION:
        print(f"[info] CUDA graph declined for shape {tuple(x.shape)}: "
              f"batch*seq*d_model is {activation}, over the "
              f"{config._GRAPH_MAX_ACTIVATION} above which replay measured no gain on "
              f"this hardware. Running eagerly; results are unaffected.\n"
              f"[info] to capture anyway:  --cuda-graph always\n"
              f"[info] to re-measure the threshold for this machine:  "
              f"python scripts/ab_graph.py")
    else:
        print(f"[info] CUDA graph declined for shape {tuple(x.shape)}, running "
              f"eagerly (results are unaffected).")


def _capture_graph(
    model: "OptimizedTransformer",
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
        if free_bytes < config._GRAPH_MIN_FREE_BYTES:
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
            gen.manual_seed(config._GRAPH_SEED)
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
                    for _ in range(config._GRAPH_WARMUP_ITERS):
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
        if config.CUDA_GRAPH != "always" and pool_bytes > budget:
            _graph_release(g)
            if not _graph_warned:
                _graph_warned = True
                print(f"[info] CUDA graph for shape {tuple(x.shape)} reserved "
                      f"{pool_bytes / (1 << 20):.0f} MiB of pinned pool, over the "
                      f"{budget / (1 << 20):.0f} MiB cap "
                      f"({config._GRAPH_POOL_SAFETY_FRACTION:.0%} of the card), so it "
                      f"was released and this shape runs eagerly. If this "
                      f"fires, "
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
                  f"{'on' if use_mask else 'off'} backend={config.ATTENTION_BACKEND} "
                  f"impl={config.ATTENTION_IMPL}, pool +{pool_mib:.1f} MiB, "
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


def _graph_cache_ptrs(model: "OptimizedTransformer") -> Tuple:
    """Addresses of the lazily-built per-layer caches a graph bakes in."""
    out = []
    for layer in model.layers:
        attn = layer.attention
        for t in (attn._qkv_weight, attn._qkv_bias, attn._causal_mask):
            out.append(None if t is None else t.data_ptr())
    return tuple(out)


def _graph_pinned_refs(model: "OptimizedTransformer") -> Tuple:
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
