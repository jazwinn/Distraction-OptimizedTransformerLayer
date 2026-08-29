"""The optimization knobs that have no command-line flag, and the gating rules.

Two things live here.

**The environment-variable knobs.** Several optimizations are switched by an
environment variable read once inside the extension, with no argparse flag
anywhere -- split-KV, the wmma causal block reversal, and the four LayerNorm
warp knobs. Today the only way to reach them is `set TILE_SPLIT_KV=0` before
the run. Because this dashboard spawns a child process per run, it can pass
them in that child's `env` and they become ordinary checkboxes, with no change
to any file in csrc/.

Each entry records where the variable is read, so the claim is checkable:

    WMMA_FP16               csrc/attention_wmma.cuh      wmma_fp16_flag()
    WMMA_SOFTMAX_MODE       csrc/attention_wmma.cuh      softmax_mode_flag()
    WMMA_SPLIT_KV           csrc/attention_wmma.cuh      split_kv_flag()
    WMMA_MASK_CLASSIFY      csrc/attention_wmma.cuh      mask_classify_flag()
    WMMA_DIRECT_O           csrc/attention_wmma.cuh      direct_o_flag()
    WMMA_ACC_FORMULA        csrc/attention_wmma.cuh      acc_formula_flag()
    WMMA_SPLIT_COUNT        csrc/attention_wmma.cuh      split_count_override()
    WMMA_CAUSAL_REVERSE     csrc/attention_wmma.cuh      causal_reverse_flag()
    TILE_SPLIT_KV           csrc/tile_attention.cu       split_flag()
    LAYERNORM_FUSED_REDUCE  csrc/add_layernorm.cuh
    LAYERNORM_WARP_WIDTH    csrc/add_layernorm.cuh
    LAYERNORM_WARP_ROWS     csrc/add_layernorm.cuh
    LAYERNORM_BLOCK_THREADS csrc/add_layernorm.cuh

The parsing convention in those files is that a leading '0' means off, so the
values written here are "0" and "1" rather than "false"/"true".

**The gating rules.** Combinations that are accepted by the form but cannot
work, checked before anything is spawned. Each one costs 5-15 seconds of torch
startup to discover the hard way, and one of them -- the memory estimate -- has
historically hung this machine rather than failing cleanly, because Windows
WDDM spills an oversubscribed allocation into system RAM instead of raising.
That is the same hazard scripts/bench_shape14.py guards with --mem-fraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import argspec

# name, label, kind, default, help, and the file the extension reads it in.
# "bool" knobs send "1"/"0"; "int" knobs send the number as text. A knob left
# at its default is not put in the environment at all, so a run with everything
# default is byte-for-byte the command the harness would get on its own.
ENV_KNOBS: List[Dict[str, Any]] = [
    {
        "name": "TILE_SPLIT_KV",
        "label": "Split-KV (Flash-Decoding)",
        "kind": "bool",
        "default": True,
        "source": "csrc/tile_attention.cu",
        "help": "Split the KV loop across blocks and combine partial softmaxes. "
                "Helps short sequences, where one block per (batch, head) leaves "
                "most of the card idle. Tile kernels only.",
    },
    {
        "name": "WMMA_CAUSAL_REVERSE",
        "label": "wmma causal block reversal",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Issue causal blocks longest-first so the heaviest rows start "
                "earliest. Only fires when head_dim <= 64, the run is causal, "
                "and there are more blocks than fit resident at once.",
    },
    {
        "name": "WMMA_FP16",
        "label": "wmma fp16 fragments",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Contract fp32 q/k/v in fp16 fragments rather than tf32. Same "
                "10-bit mantissa, so the same accuracy, but faster tensor cores. "
                "--attn-precision fp16/tf32 sets the same thing per call; prefer "
                "the flag "
                "and leave this alone unless comparing the two paths.",
    },
    {
        "name": "WMMA_SOFTMAX_MODE",
        "label": "wmma softmax mode",
        "kind": "int",
        "default": 1,
        "source": "csrc/attention_wmma.cuh",
        "help": "0 = the original softmax (score * scale, then __expf, plus an "
                "explicit -inf test). 1 = the base-2 domain: one fused score "
                "multiply, a bare exp2f, and no -inf test, at identical "
                "accuracy -- the default, worth 1.04x on the op. 2 also folds "
                "scale*log2(e) into Q, which is a wash on speed and breaks the "
                "2e-3 atol at head_dim 64; kept only so that can be re-checked.",
    },
    {
        "name": "WMMA_MASK_CLASSIFY",
        "label": "wmma per-tile mask classification",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Classify each key tile once -- rows inside S, columns inside "
                "S, wholly below the causal diagonal -- and skip the "
                "per-element bounds, causal and mask tests on the interior "
                "ones. Bit-identical either way, since it only skips tests "
                "that would have passed. Worth 1.33x on the op and 1.05x end "
                "to end.",
    },
    {
        "name": "WMMA_ACC_FORMULA",
        "label": "wmma accumulator map from a closed form",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Compute which row each accumulator element holds, instead of "
                "rediscovering it with a shared-memory probe once per block. "
                "The probe still runs once per process to confirm the closed "
                "form reproduces it, and the kernel falls back to probing per "
                "block if it does not. Bit-identical either way.",
    },
    {
        "name": "WMMA_DIRECT_O",
        "label": "wmma O straight to global",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Store O from the accumulator fragments straight to global "
                "memory instead of staging the whole block tile through "
                "shared. Also shrinks that tile to one fragment per warp, "
                "which frees 8 KB at head_dim 64 and 128 and 16 KB at 256 and "
                "buys a fourth resident block per SM at 64 and 128. Bit- "
                "identical either way.",
    },
    {
        "name": "WMMA_SPLIT_KV",
        "label": "wmma split-KV (Flash-Decoding)",
        "kind": "bool",
        "default": True,
        "source": "csrc/attention_wmma.cuh",
        "help": "Split the key range across extra blocks when the attention "
                "grid is too small to fill the card. Gated hard: head_dim > 8, "
                "at most an eighth of the card in use, and at least 4 key "
                "tiles to divide. Worth 1.7x-2.7x on the op and 1.18x-1.39x "
                "end to end at batch 1 with head_dim 64; every other shape is "
                "declined and runs the single-pass kernel unchanged.",
    },
    {
        "name": "WMMA_SPLIT_COUNT",
        "label": "wmma split count override",
        "kind": "int",
        "default": 0,
        "source": "csrc/attention_wmma.cuh",
        "help": "Force a split count instead of the measured rule; 0 restores "
                "the rule. Counts the rule declines are usually slower -- this "
                "is here to re-measure that, not to tune with.",
    },
    {
        "name": "LAYERNORM_FUSED_REDUCE",
        "label": "LayerNorm fused reduction",
        "kind": "bool",
        "default": True,
        "source": "csrc/add_layernorm.cuh",
        "help": "Fold the second pass's two sums into one block reduction.",
    },
    {
        "name": "LAYERNORM_WARP_WIDTH",
        "label": "LayerNorm warp-per-row width",
        "kind": "int",
        "default": -1,
        "source": "csrc/add_layernorm.cuh",
        "help": "Row width up to which one warp handles a whole row. -1 keeps "
                "the built-in 256; 0 disables warp-per-row entirely and forces "
                "the block-per-row kernel.",
    },
    {
        "name": "LAYERNORM_WARP_ROWS",
        "label": "LayerNorm rows per block",
        "kind": "int",
        "default": 0,
        "source": "csrc/add_layernorm.cuh",
        "help": "Rows each warp-per-row block handles. 0 keeps the default of 4; "
                "capped at 32.",
    },
    {
        "name": "LAYERNORM_BLOCK_THREADS",
        "label": "LayerNorm block threads",
        "kind": "int",
        "default": 0,
        "source": "csrc/add_layernorm.cuh",
        "help": "Force a block size for the block-per-row kernel. 0 keeps the "
                "width-scaled rule; clamped to whole warps and to 1024.",
    },
]

ENV_BY_NAME = {knob["name"]: knob for knob in ENV_KNOBS}

# Head dimensions each attention kernel covers, from the dispatch in
# csrc/attention_dispatch.cuh. A forced --attn-impl outside its own set does
# not fall back -- it raises, by design, so that "--attn-impl scalar" can never
# quietly time ATen and report it as the scalar kernel.
# head_dims an impl will actually RUN. Keyed by impl alone now that precision is
# a separate axis: every precision a given kernel has covers the same head_dims,
# because the shared-memory budget that sets the coverage is decided per kernel
# rather than per math mode.
#
# scalar is deliberately absent: it has a generic kernel behind its six tuned
# instantiations that takes ANY head_dim up to SCALAR_MAX_HEAD_DIM, so a set of
# six would block shapes that run fine. It used to be listed here with the same
# six as the others, which stopped being true when that kernel landed.
HEAD_DIM_COVERAGE = {
    "wmma": {8, 16, 32, 64, 128, 256},
    "tile": {8, 16, 32, 64, 128, 256},
}

# Past this a query row's threads outgrow a warp and the generic scalar kernel's
# score butterfly stops working, so it declines -- csrc/attention_scalar.cuh.
# This is also auto's ceiling, since auto ends on the same kernel.
SCALAR_MAX_HEAD_DIM = 2048

TILE_IMPLS = ("tile",)

# Which arithmetic each kernel has. Mirrors the table in csrc/kernel_common.cuh
# and the impl_supports() gate in csrc/attention_dispatch.cuh -- if those change,
# this has to change with them, or the dashboard will block a run the kernel
# would have accepted (or wave through one it refuses).
PRECISION_SUPPORT = {
    "auto": {"auto", "fp32", "tf32", "fp16", "bf16"},
    "scalar": {"auto", "fp32"},
    "wmma": {"auto", "tf32", "fp16", "bf16"},
    "tile": {"auto", "fp32", "tf32", "fp16", "bf16"},
}

# Why, in the kernel's own terms, so the message says something more useful than
# "unsupported combination".
PRECISION_REFUSAL = {
    "scalar": "the scalar kernel accumulates everything in fp32 and that is the "
              "point of it -- 5e-6 against an exact reference, three orders "
              "tighter than the tensor-core paths. Narrowing it would just make "
              "a slower wmma",
    "wmma": "no shipped tensor core does a full fp32 matmul, so this is a "
            "category error rather than a missing instantiation. tf32 is the "
            "closest thing wmma has",
}

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


# The harness works out for itself how to split a batch and whether the
# baseline can run, from two module constants. Both are read from its source
# rather than copied, so retuning either there moves this too.
_HARNESS = "torch_transformer_benchmark.py"
_FALLBACK_CONSTANTS = {
    "_PEAK_ACTIVATION_FACTOR": 10,
    "_MEMORY_BUDGET_FRACTION": 0.85,
}


def harness_constants() -> Dict[str, Any]:
    found = argspec.numeric_constants(_HARNESS, list(_FALLBACK_CONSTANTS))
    return {**_FALLBACK_CONSTANTS, **found}


def stream_plan(batch: int, seq: int, d_model: int, heads: int, causal: bool,
                dtype: str, total_bytes: Optional[int],
                baseline_chunk: int = 0) -> Dict[str, Any]:
    """What the harness will do with this shape, predicted the way it decides.

    Mirrors auto_stream_rows() and baseline_can_run() in the harness:

        budget  = total * _MEMORY_BUDGET_FRACTION
        per_row = seq * d_model * elsize * _PEAK_ACTIVATION_FACTOR
        rows    = batch if per_row * batch <= budget else budget // per_row
        baseline runs iff 2 * [rows,H,S,S] fp32 scores + [S,S] mask <= budget

    The last line is what the harness decides for itself when the scores do not
    fit: rather than skip the baseline it scores one query block at a time, so
    what must fit becomes a [rows,H,block,S] slab plus that block's mask plus the
    activation peak, and the [S,S] causal mask is never built at all. Same
    arithmetic, so a shape that could only be timed can now be compared too. The
    block is mirrored here the way the harness picks it -- the largest that fits,
    rounded down to a power of two. `baseline_chunk` overrides it: positive
    forces a block, -1 turns chunking off.

    Two consequences worth stating, because both used to be treated here as
    reasons a shape could not run at all:

    * The full [B,S,D] input is never allocated when it does not fit. Rows of a
      batch do not interact in a transformer forward, so slicing is the same
      computation, and the harness slices.

    * A baseline that cannot run is skipped, not fatal. The run then measures
      the optimized model alone -- real latencies with no speedup beside them,
      because there is no reference to divide by.

    So the only genuine wall left is a single row not fitting.
    """
    constants = harness_constants()
    element = DTYPE_BYTES.get(dtype, 4)
    per_row = seq * d_model * element * int(constants["_PEAK_ACTIVATION_FACTOR"])

    if not total_bytes or per_row <= 0:
        # No device information: assume it runs whole, as the harness does.
        return {"rows": batch, "slices": 1, "per_row_bytes": per_row,
                "peak_bytes": per_row * batch, "budget_bytes": None,
                "baseline_runs": True, "one_row_fits": True, "known": False,
                "baseline_chunk": max(0, baseline_chunk),
                "baseline_chunk_auto": False, "largest_baseline_chunk": 0}

    budget = int(total_bytes * float(constants["_MEMORY_BUDGET_FRACTION"]))
    if per_row * batch <= budget:
        rows = batch
    else:
        rows = max(1, min(batch, budget // per_row))
    slices = (batch + rows - 1) // rows

    whole_scores = rows * heads * seq * seq * 4
    whole_mask = seq * seq if causal else 0
    fits_whole = (2 * whole_scores + whole_mask) <= budget

    # The largest block that fits, the way largest_baseline_chunk() prices it.
    spare = budget - per_row * rows
    per_query = 2 * rows * heads * seq * 4 + seq
    largest = min(seq, spare // per_query) if spare > 0 and per_query > 0 else 0

    if baseline_chunk > 0:
        block = min(baseline_chunk, seq)
    elif baseline_chunk < 0 or fits_whole or largest < 1:
        block = 0
    else:
        block = 1 << (int(largest).bit_length() - 1)

    if block:
        scores = rows * heads * block * seq * 4
        baseline_runs = block <= largest
    else:
        scores = whole_scores
        baseline_runs = fits_whole
    return {
        "rows": rows,
        "slices": slices,
        "per_row_bytes": per_row,
        "peak_bytes": per_row * rows,
        "budget_bytes": budget,
        "baseline_runs": baseline_runs,
        "baseline_scores_bytes": scores,
        "baseline_chunk": block,
        "baseline_chunk_auto": bool(block) and baseline_chunk == 0,
        "largest_baseline_chunk": largest,
        "one_row_fits": per_row <= budget,
        "known": True,
    }


def env_value(name: str, value: Any) -> Optional[str]:
    """What to put in the child's environment for one knob, or None.

    None means "leave it out": either the name is not a knob, or the submitted
    value is unusable, or it is simply the default -- and a knob at its default
    must not appear, so that a run with nothing touched produces exactly the
    command the harness would get by hand.

    Coercion belongs here rather than at the call sites because JSON blurs the
    types: it can carry "0" where the default is False, or 0 where it is -1.
    Two call sites deciding that separately is how the previewed command and the
    preflight warning come to disagree, so both go through this.
    """
    knob = ENV_BY_NAME.get(name)
    if knob is None or value is None or value == "":
        return None
    if knob["kind"] == "bool":
        wanted = value in (True, "true", "on", 1, "1")
        return None if wanted == knob["default"] else ("1" if wanted else "0")
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return None if number == knob["default"] else str(number)


def human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(count) < 1024 or unit == "PiB":
            return f"{count:.0f} B" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} PiB"


def preflight(form: Dict[str, Any], tile_available: Optional[bool],
              device_total_bytes: Optional[int],
              memory_fraction: float = 0.9) -> List[Dict[str, str]]:
    """Problems with a form submission, worst first.

    Returns a list of {level, message}. "error" means the run cannot produce a
    number and is blocked; "warning" means it will run but will not measure
    what was probably meant -- a tile mode under fp16, say, where the kernel
    declines and the fallback gets timed instead.

    `tile_available` may be None, meaning "not probed yet", which suppresses the
    tile-support check rather than guessing either way.
    """
    issues: List[Dict[str, str]] = []

    def error(message: str) -> None:
        issues.append({"level": "error", "message": message})

    def warn(message: str) -> None:
        issues.append({"level": "warning", "message": message})

    def integer(key: str, fallback: int) -> int:
        try:
            return int(form.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    batch = integer("batch_size", 8)
    seq = integer("seq_len", 128)
    d_model = integer("d_model", 512)
    heads = integer("heads", 8)
    causal = bool(form.get("causal"))
    dtype = form.get("dtype") or "float32"
    impl = form.get("attn_impl")
    precision = form.get("attn_precision")
    linear_gelu = form.get("linear_gelu")

    if min(batch, seq, d_model, heads, integer("layers", 6)) < 1:
        error("batch, seq_len, d_model, heads and layers must all be at least 1")
        return issues
    if d_model % heads:
        error(f"d_model {d_model} is not divisible by heads {heads}; "
              f"TransformerConfig.validate() rejects this")
        return issues

    head_dim = d_model // heads

    # --- kernel x precision ----------------------------------------------
    # Checked before coverage, because a pair the kernel has no arithmetic for
    # never gets as far as being asked about a head_dim.
    if precision and precision != "auto":
        allowed = PRECISION_SUPPORT.get(impl or "auto")
        if allowed is not None and precision not in allowed:
            reason = PRECISION_REFUSAL.get(impl)
            has = sorted(p for p in allowed if p != "auto")
            error(f"--attn-impl {impl} has no {precision} arithmetic, so this "
                  f"run would raise rather than produce a number. It supports "
                  f"{', '.join(has)}"
                  + (f" -- {reason}" if reason else "")
                  + f". Note this is the MATH type, not the tensor dtype: to "
                    f"feed {impl} {precision} tensors, set dtype instead.")

    # bf16 runs everywhere it is offered and fails the accuracy gate almost
    # everywhere, which is a warning rather than an error -- measuring how far
    # off it is is the only reason it is exposed at all.
    if precision == "bf16":
        warn("bf16 carries 8 significand bits and does not clear the harness's "
             "2e-3 accuracy gate (measured 2.9e-3 on wmma at d_model 512). "
             "Expect this run to report FAIL; it is exposed for measurement, "
             "not as something to ship.")

    # --- kernel coverage -------------------------------------------------
    # auto is included now. It used to be skipped on the grounds that auto
    # always finds something, which stopped being true at the top end: auto
    # ends on the generic scalar kernel, and that kernel has a ceiling.
    if impl in (None, "", "auto", "scalar"):
        if head_dim > SCALAR_MAX_HEAD_DIM:
            error(f"head_dim {head_dim} (d_model {d_model} / heads {heads}) is "
                  f"past every kernel's reach. The generic scalar kernel is the "
                  f"catch-all and stops at {SCALAR_MAX_HEAD_DIM}, where a query "
                  f"row's threads would outgrow a warp.")
    else:
        covered = HEAD_DIM_COVERAGE.get(impl, set())
        if head_dim not in covered:
            hint = ("" if impl not in ("wmma", "tile")
                    else "; --attn-impl scalar takes any head_dim to "
                         f"{SCALAR_MAX_HEAD_DIM}, and auto falls back to it")
            error(f"--attn-impl {impl} does not cover head_dim {head_dim} "
                  f"(d_model {d_model} / heads {heads}); it handles "
                  f"{sorted(covered)} and raises on anything else{hint}")

    if impl in TILE_IMPLS and tile_available is False:
        error(f"--attn-impl {impl} needs a build with cuTile support, and this "
              f"build reports none. Rebuild against CUDA 13.3 or newer, or pick "
              f"a wmma/scalar impl.")

    # --- dtype gates -----------------------------------------------------
    if dtype in ("float16", "bfloat16"):
        if impl in TILE_IMPLS:
            error(f"--attn-impl {impl} requires float32 tensors; the tile "
                  f"launcher checks the dtype and raises under {dtype}. The "
                  f"tile modes narrow only the GEMM operands, not the tensors.")
        if linear_gelu in (None, "auto", "tf32"):
            warn(f"the fused Linear+GELU kernel declines under {dtype} (it "
                 f"needs float32 operands) and silently falls back to cuBLAS + "
                 f"F.gelu, so this run does not measure the fusion")

    # --- backend gates ---------------------------------------------------
    # There used to be one here for --attn-backend sdpa, which switched off
    # every custom kernel and so made every knob below it inert. That backend
    # is gone: this project may not use a prebuilt attention, so "custom" is
    # the only value and there is nothing left to warn about.

    # --- flag conflicts --------------------------------------------------
    if form.get("compile_user") and form.get("cuda_graph") not in (None, "", "off"):
        error("--compile-user with --compile-mode reduce-overhead already "
              "captures CUDA graphs; optimized/cli.py raises when both are "
              "explicit. Set CUDA graph to 'off' or leave it at the default.")

    if form.get("cuda_graph") == "always":
        activation = batch * seq * d_model
        if activation > (1 << 19):
            warn(f"--cuda-graph always at {activation:,} activation elements is "
                 f"past the point where replay measured any gain (524,288); it "
                 f"pins the whole working set for no speedup")

    # --- memory, as the harness itself decides it ------------------------
    plan = stream_plan(batch, seq, d_model, heads, causal, dtype,
                       device_total_bytes, integer("baseline_chunk", 0))
    if plan["known"] and not plan["one_row_fits"]:
        error(f"a single row of this shape needs about "
              f"{human_bytes(plan['per_row_bytes'])} against a "
              f"{human_bytes(plan['budget_bytes'])} budget, so there is nothing "
              f"left to split. The harness streams the batch, but it cannot "
              f"stream below one row.")
    elif plan["slices"] > 1:
        warn(f"the full [{batch},{seq},{d_model}] input does not fit, so the "
             f"harness will stream it as {plan['slices']} slices of up to "
             f"{plan['rows']} row(s). Batch rows do not interact, so this is "
             f"the same computation -- but the reported median is per slice.")
    if plan["known"] and not plan["baseline_runs"]:
        queries = plan["baseline_chunk"] or seq
        if plan["baseline_chunk"]:
            tail = (f"A --baseline-chunk under {plan['largest_baseline_chunk']} "
                    f"is the lever here.")
        elif plan["largest_baseline_chunk"] >= 1:
            tail = (f"Chunking is off; left alone the harness would score "
                    f"{plan['largest_baseline_chunk']} queries at a time here "
                    f"and would have a reference.")
        else:
            tail = ("Chunking cannot help either -- the activation peak this "
                    "shape needs already uses the whole budget.")
        warn(f"the baseline cannot run this shape -- its [{plan['rows']},"
             f"{heads},{queries},{seq}] scores are "
             f"{human_bytes(plan['baseline_scores_bytes'])} even one row at a "
             f"time, and it needs a second copy for the fp32 softmax. The "
             f"harness will skip it and time the optimized model alone, so "
             f"this run reports latency with no speedup and no accuracy check. "
             f"{tail}")
    elif plan["known"] and plan["baseline_chunk_auto"]:
        warn(f"the baseline's [{plan['rows']},{heads},{seq},{seq}] scores do not "
             f"fit, so the harness will score {plan['baseline_chunk']} queries "
             f"at a time instead of skipping the baseline. Same arithmetic, so "
             f"this shape does get an accuracy verdict and a speedup -- but the "
             f"reference is far slower than a fitting one, so expect minutes per "
             f"forward. --baseline-chunk -1 skips the baseline as before.")

    return issues


def estimate_summary(form: Dict[str, Any],
                     device_total_bytes: Optional[int] = None) -> Dict[str, Any]:
    """The derived facts the UI shows for a shape, GPU untouched."""
    def integer(key: str, fallback: int) -> int:
        try:
            return int(form.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    batch = integer("batch_size", 8)
    seq = integer("seq_len", 128)
    d_model = integer("d_model", 512)
    heads = integer("heads", 8)
    causal = bool(form.get("causal"))
    dtype = form.get("dtype") or "float32"

    plan = stream_plan(batch, seq, d_model, heads, causal, dtype,
                       device_total_bytes, integer("baseline_chunk", 0))
    element = DTYPE_BYTES.get(dtype, 4)
    return {
        "head_dim": (d_model // heads
                     if heads and d_model and d_model % heads == 0 else None),
        "tokens": batch * seq,
        "activation_elements": batch * seq * d_model,
        "input_bytes": batch * seq * d_model * element,
        "plan": plan,
        "human": {
            "input": human_bytes(batch * seq * d_model * element),
            "peak": human_bytes(plan["peak_bytes"]),
            "per_row": human_bytes(plan["per_row_bytes"]),
        },
    }
