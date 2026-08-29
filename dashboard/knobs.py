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
                "--attn-fp16 sets the same thing through config; prefer the flag "
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
HEAD_DIM_COVERAGE = {
    "scalar": {8, 16, 32, 64, 128, 256},
    "wmma": {8, 16, 32, 64, 128, 256},
    "tile": {8, 16, 32, 64, 256},
    "tile-bf16": {8, 16, 32, 64, 256},
    "tile-tf32": {8, 16, 32, 64, 256},
    "tile-fp16": {8, 16, 32, 64, 256},
}

TILE_IMPLS = ("tile", "tile-bf16", "tile-tf32", "tile-fp16")

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
                dtype: str, total_bytes: Optional[int]) -> Dict[str, Any]:
    """What the harness will do with this shape, predicted the way it decides.

    Mirrors auto_stream_rows() and baseline_can_run() in the harness:

        budget  = total * _MEMORY_BUDGET_FRACTION
        per_row = seq * d_model * elsize * _PEAK_ACTIVATION_FACTOR
        rows    = batch if per_row * batch <= budget else budget // per_row
        baseline runs iff 2 * [rows,H,S,S] fp32 scores + [S,S] mask <= budget

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
                "baseline_runs": True, "one_row_fits": True, "known": False}

    budget = int(total_bytes * float(constants["_MEMORY_BUDGET_FRACTION"]))
    if per_row * batch <= budget:
        rows = batch
    else:
        rows = max(1, min(batch, budget // per_row))
    slices = (batch + rows - 1) // rows

    scores = rows * heads * seq * seq * 4
    mask = seq * seq if causal else 0
    return {
        "rows": rows,
        "slices": slices,
        "per_row_bytes": per_row,
        "peak_bytes": per_row * rows,
        "budget_bytes": budget,
        "baseline_runs": (2 * scores + mask) <= budget,
        "baseline_scores_bytes": scores,
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


def knob_differs(name: str, value: Any) -> bool:
    """Whether a submitted knob value actually moves it off its default."""
    return env_value(name, value) is not None


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
    backend = form.get("attn_backend")
    impl = form.get("attn_impl")
    linear_gelu = form.get("linear_gelu")

    if min(batch, seq, d_model, heads, integer("layers", 6)) < 1:
        error("batch, seq_len, d_model, heads and layers must all be at least 1")
        return issues
    if d_model % heads:
        error(f"d_model {d_model} is not divisible by heads {heads}; "
              f"TransformerConfig.validate() rejects this")
        return issues

    head_dim = d_model // heads

    # --- kernel coverage -------------------------------------------------
    if impl and impl != "auto":
        covered = HEAD_DIM_COVERAGE.get(impl, set())
        if head_dim not in covered:
            error(f"--attn-impl {impl} does not cover head_dim {head_dim} "
                  f"(d_model {d_model} / heads {heads}); it handles "
                  f"{sorted(covered)} and raises on anything else")

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
    if backend == "sdpa":
        chosen = [name for name, value in (("--attn-impl", impl),
                                           ("--attn-fp16", form.get("attn_fp16")),
                                           ("--linear-gelu", linear_gelu))
                  if value not in (None, "", "auto")]
        if chosen:
            warn("--attn-backend sdpa switches off every custom kernel, not "
                 "just attention -- LayerNorm and Linear+GELU included. "
                 + ", ".join(chosen) + " will have no effect.")
        env = form.get("env") or {}
        active = [name for name in env
                  if name in ENV_BY_NAME and knob_differs(name, env[name])]
        if active:
            warn("--attn-backend sdpa means the extension is never loaded, so "
                 + ", ".join(sorted(active)) + " will have no effect")

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
                       device_total_bytes)
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
        warn(f"the baseline cannot run this shape -- its [{plan['rows']},"
             f"{heads},{seq},{seq}] scores are "
             f"{human_bytes(plan['baseline_scores_bytes'])} even one row at a "
             f"time, and it needs a second copy for the fp32 softmax. The "
             f"harness will skip it and time the optimized model alone, so "
             f"this run reports latency with no speedup and no accuracy check.")

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
                       device_total_bytes)
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
