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
import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Above torch on purpose: importing kernel_ext preloads the driver's GPU
# compiler, which stops --attn-impl tile from exiting 0xC0000005 only if it
# happens before torch pulls the NVIDIA DLLs in. It builds nothing here -- the
# extension is still built lazily on first use. See
# kernel_ext.preload_tile_compiler().
import kernel_ext  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimized import OptimizedTransformer
from optimized import cli as optimized_cli


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
        self.norm1 = nn.LayerNorm(d_model)
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


class UserOptimizedTransformer(OptimizedTransformer, BaselineTransformer):
    """
    The optimized implementation. Its body lives in the optimized/ package so
    that this file stays the harness it started as -- see optimized/__init__.py
    for what is where.

    Both bases matter: OptimizedTransformer supplies every method, and
    BaselineTransformer keeps the two models isinstance-compatible. Nothing is
    inherited from the baseline, since every submodule has a replacement.

    Requirements this still meets:
      1. Keep the forward signature unchanged.
      2. Return a tensor with shape [batch_size, seq_len, d_model].
      3. Keep compatible parameter names, or customize copy_model_weights().
    """


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


# Automatic handling for shapes that do not fit. No flags, and every shape that
# already ran resolves to what this file did before. Shape 14 is why: its input
# is 12.21 GiB (allocated before either model exists) and the baseline's scores
# tensor is 18.63 TiB, so the batch is sliced and the baseline skipped.

# Peak device memory per forward, as a multiple of one [B,S,D] activation.
# Measured 9.03-9.33 across the large appendix shapes; small ones read higher
# only because fixed overhead dominates there.
_PEAK_ACTIVATION_FACTOR = 10

# Predicted share of device memory allowed before the batch is split. 0.85 keeps
# shape 6 -- 6.10 GiB predicted of 6.80, 5.51 actual -- running whole.
_MEMORY_BUDGET_FRACTION = 0.85

# Wall-clock the benchmark may spend before trimming its own iteration counts.
# 320 passes per model is free at milliseconds and absurd at seconds: 120 s
# leaves twelve of the thirteen runnable shapes untouched (shape 13, the
# slowest, lands at 59.5 s) and trims only shapes 6 (31 min) and 14 (15 min).
_BENCHMARK_BUDGET_SECONDS = 120.0
_MIN_WARMUP = 1
_MIN_REPEATS = 3


def _probe_forward_ms(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    """Cost of one forward in ms, timed on the second pass: lazy init, cuBLAS
    heuristics and allocator growth all land on the first and would inflate it.
    """
    with torch.inference_mode():
        model(x, valid_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(x, valid_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def scale_iterations(
    per_pass_ms: float, warmup: int, repeats: int, rounds: int
) -> Tuple[int, int, int]:
    """Trim (warmup, repeats, rounds) into _BENCHMARK_BUDGET_SECONDS, unchanged
    when they already fit. Rounds go last -- they alternate model order, so
    shedding them would trade thermal bias for time.
    """
    if per_pass_ms <= 0.0:
        return warmup, repeats, rounds
    estimate_s = per_pass_ms * (warmup + repeats * rounds) / 1000.0
    if estimate_s <= _BENCHMARK_BUDGET_SECONDS:
        return warmup, repeats, rounds

    affordable = int(_BENCHMARK_BUDGET_SECONDS * 1000.0 / per_pass_ms)
    new_warmup = max(_MIN_WARMUP, min(warmup, affordable // 10))
    timed = max(_MIN_REPEATS, affordable - new_warmup)
    new_rounds = max(1, min(rounds, timed // _MIN_REPEATS))
    new_repeats = max(_MIN_REPEATS, timed // new_rounds)
    return new_warmup, new_repeats, new_rounds


def _device_budget_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    _, total = torch.cuda.mem_get_info(device)
    return int(total * _MEMORY_BUDGET_FRACTION)


def auto_stream_rows(
    config: TransformerConfig, device: torch.device, dtype: torch.dtype
) -> int:
    """Rows per slice: config.batch_size when the whole batch fits, else fewer."""
    budget = _device_budget_bytes(device)
    if budget is None:
        return config.batch_size
    element = torch.empty((), dtype=dtype).element_size()
    per_row = config.seq_len * config.d_model * element * _PEAK_ACTIVATION_FACTOR
    if per_row <= 0 or per_row * config.batch_size <= budget:
        return config.batch_size
    return max(1, min(config.batch_size, budget // per_row))


def baseline_can_run(
    config: TransformerConfig, device: torch.device, dtype: torch.dtype, rows: int
) -> bool:
    """Whether the baseline's [rows,H,S,S] scores, its fp32 softmax copy and the
    causal mask fit. Checked, not discovered by OOM: on Windows an oversubscribed
    allocation does not raise, it spills to system RAM and crawls.
    """
    budget = _device_budget_bytes(device)
    if budget is None:
        return True
    scores = rows * config.num_heads * config.seq_len * config.seq_len * 4
    mask = config.seq_len * config.seq_len if config.causal else 0
    return (2 * scores + mask) <= budget


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    batch = config.batch_size if rows is None else rows
    x = torch.randn(
        batch,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            batch, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(batch,),
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

    rows = auto_stream_rows(config, device, dtype)
    if not baseline_can_run(config, device, dtype, rows):
        scores_gib = (
            rows * config.num_heads * config.seq_len * config.seq_len * 4 / 1024 ** 3
        )
        print(
            f"skipped: the baseline cannot run this shape. Its scores tensor is "
            f"[{rows},{config.num_heads},{config.seq_len},{config.seq_len}] = "
            f"{scores_gib:,.0f} GiB even one row at a time, and it needs a second "
            f"copy for the fp32 softmax."
        )
        print(
            "         Not a device limit: at ~3.3e14 MACs per layer it would also "
            "need ~200 s per forward. There is no reference to compare against, so "
            "only the optimized model is benchmarked below."
        )
        return True

    if rows != config.batch_size:
        n = (config.batch_size + rows - 1) // rows
        print(
            f"streaming: {n} slices of up to {rows} row(s) -- the full "
            f"[{config.batch_size},{config.seq_len},{config.d_model}] case does not "
            f"fit, and batch rows do not interact, so this is the same comparison."
        )
    slices = [
        (lo, min(lo + rows, config.batch_size))
        for lo in range(0, config.batch_size, rows)
    ]

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
          for slice_index, (lo, hi) in enumerate(slices):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                # Spaced so no (trial, slice) pair repeats another's data.
                seed=seed + trial + slice_index * (trials + 1),
                padding_ratio=padding_ratio,
                input_scale=input_scale,
                rows=(hi - lo) if len(slices) > 1 else None,
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

    # Same two decisions as the accuracy phase, from the same shape, so a run
    # cannot stream in one phase and not the other.
    rows = auto_stream_rows(config, device, dtype)
    n_slices = (config.batch_size + rows - 1) // rows
    with_baseline = baseline_can_run(config, device, dtype, rows)

    if n_slices > 1:
        print(
            f"streaming: {n_slices} slices of {rows} row(s); a full pass over the "
            f"batch is {n_slices}x the per-slice figure below"
        )
    if not with_baseline:
        print("baseline: cannot run this shape (see above), so no speedup is reported")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
        rows=rows if n_slices > 1 else None,
    )

    # One probe per model, then trim the iteration counts if the defaults would
    # spend half an hour re-measuring a number that is already stable. Both
    # models are probed because the benchmark pays for both, and on most shapes
    # the baseline is the slower of the two.
    per_pass_ms = _probe_forward_ms(optimized, x, valid_mask, device)
    if with_baseline:
        per_pass_ms += _probe_forward_ms(baseline, x, valid_mask, device)
    scaled_warmup, scaled_repeats, scaled_rounds = scale_iterations(
        per_pass_ms, warmup, repeats, rounds
    )
    if (scaled_warmup, scaled_repeats, scaled_rounds) != (warmup, repeats, rounds):
        print(
            f"one pass over every model measured {per_pass_ms:.1f} ms, so "
            f"--warmup {warmup} --repeats {repeats} --benchmark-rounds {rounds} "
            f"would take {per_pass_ms * (warmup + repeats * rounds) / 1000.0:,.0f} s"
        )
        print(
            f"scaled down to warmup {scaled_warmup}, repeats {scaled_repeats}, "
            f"rounds {scaled_rounds} to stay near {_BENCHMARK_BUDGET_SECONDS:.0f} s. "
            f"Pass the flags explicitly to override; latency is unaffected, only "
            f"the sample count behind the median."
        )
        warmup, repeats, rounds = scaled_warmup, scaled_repeats, scaled_rounds

    # Warm up before collecting any timing data.
    if with_baseline:
        warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if not with_baseline:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        elif round_index % 2 == 0:
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

    optimized_result = TimingResult(optimized_samples)
    tokens_per_call = config.batch_size * config.seq_len
    optimized_tokens_per_second = (
        tokens_per_call * 1000.0 / (optimized_result.median_ms * n_slices)
    )

    if not with_baseline:
        unit = "ms/slice" if n_slices > 1 else "ms"
        print(
            f"optimized: median={optimized_result.median_ms:.4f} {unit} | "
            f"mean={optimized_result.mean_ms:.4f} ms | "
            f"p90={optimized_result.p90_ms:.4f} ms | "
            f"min={optimized_result.min_ms:.4f} ms | "
            f"throughput={optimized_tokens_per_second:.2f} token/s"
        )
        if n_slices > 1:
            print(
                f"optimized: {optimized_result.median_ms * n_slices:.4f} ms for the "
                f"whole batch of {config.batch_size}"
            )
        return

    baseline_result = TimingResult(baseline_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    baseline_tokens_per_second = (
        tokens_per_call * 1000.0 / (baseline_result.median_ms * n_slices)
    )

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
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
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
    optimized_cli.add_arguments(parser)
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
    optimized_cli.validate_args(args, device, dtype)


def main() -> int:
    args = parse_args()
    optimized_cli.apply_overrides(args)

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

    # The baseline is still built and weight-copied on the CPU above, which is
    # what the optimized model's weights come from. It only reaches the device if
    # it can actually run the shape.
    if baseline_can_run(config, device, dtype, auto_stream_rows(config, device, dtype)):
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
