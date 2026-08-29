"""Harness stdout -> a row of numbers.

scripts/compare_backends.py already scrapes the same output with two regexes:

    re.search(r"summary: (PASS|FAIL)", out)
    re.search(r"speedup\\s+: ([\\d.]+)x", out)

So the format is already load-bearing somewhere else in the repo, and this is
the same contract read more completely rather than a new one. Both of those
patterns appear below unchanged, which means a change to the harness's output
breaks this file and compare_backends.py together -- and the mismatch will be
obvious, because the raw log is displayed next to the parsed table in the UI.

Everything here is incremental: `feed()` takes one line at a time and updates
the result in place, so a table row fills in while the run is still going
rather than only at the end. That matters for the long shapes, where the
accuracy trials alone can take minutes.

Nothing raises. A line that does not match any pattern is simply not a line
this cares about; a run that dies early yields a result with None in the
fields it never got to print, and `ok` False.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# One float, in any of the formats the harness's own format specs emit:
# .4f for milliseconds, .6g for errors (so "1.234e-05" must match), .2f for
# throughput, .3f for the speedup.
_NUM = r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)"

_CONFIG = re.compile(r"^TransformerConfig\((.*)\)\s*$")
_DEVICE = re.compile(r"^device=(\S+?),\s*dtype=(\S+?),\s*torch=(\S+)\s*$")
_GPU = re.compile(r"^gpu=(.+?)\s*$")
_CRITERION = re.compile(r"^criterion:\s*(.+?)\s*$")

_TRIAL = re.compile(
    r"^trial\s+(\d+)/(\d+):\s+(PASS|FAIL)\s*\|"
    r"\s*max_abs=" + _NUM + r"\s*\|"
    r"\s*max_rel=" + _NUM + r"\s*\|"
    r"\s*failed=(\d+)/(\d+)"
)
_SUMMARY = re.compile(
    r"^summary:\s*(PASS|FAIL)\s*\|"
    r"\s*max_abs=" + _NUM + r"\s*\|"
    r"\s*max_rel=" + _NUM + r"\s*\|"
    r"\s*failed=(\d+)/(\d+)"
)
# The throughput field is optional and the median may be labelled "ms/slice":
# when the full [B,S,D] input does not fit, the harness streams the batch and
# reports per-slice timings, because that is what it actually measured.
_TIMING = re.compile(
    r"^(baseline|optimized)\s*:\s*"
    r"median=" + _NUM + r"\s*ms(?:/slice)?\s*\|"
    r"\s*mean=" + _NUM + r"\s*ms\s*\|"
    r"\s*p90=" + _NUM + r"\s*ms\s*\|"
    r"\s*min=" + _NUM + r"\s*ms"
    r"(?:\s*\|\s*throughput=" + _NUM + r"\s*token/s)?"
)

# The follow-up line a streamed run prints: the whole-batch cost, which is the
# number that corresponds to one full forward pass rather than one slice.
_WHOLE_BATCH = re.compile(
    r"^optimized\s*:\s*" + _NUM + r"\s*ms for the whole batch of\s*(\d+)"
)

# "streaming: 32 slices of up to 1 row(s) -- ..."
_STREAMING = re.compile(r"^streaming:\s*(\d+)\s*slices?\s*of\s*(?:up to\s*)?(\d+)")

# The baseline being skipped is not an error: at some shapes its [B,H,S,S]
# scores cannot be held by any GPU, so the harness times the optimized model
# alone. Real latencies, no speedup, no accuracy verdict.
_BASELINE_SKIPPED = re.compile(r"^skipped:\s*the baseline cannot run this shape")

_SPEEDUP = re.compile(r"^speedup\s*:\s*" + _NUM + r"x")

# The harness prints this instead of a benchmark when accuracy failed and
# --benchmark-on-failure was not passed. Without it, "no timing lines" and
# "crashed before timing" look identical in the table.
_SKIPPED = re.compile(r"^Performance benchmark skipped because accuracy")

# Anything the child writes that looks like a Python traceback or a torch error
# is worth surfacing on the row rather than leaving buried in the log.
_ERROR_HINT = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"^\w*(Error|Exception):|"
    r"torch\.OutOfMemoryError|"
    r"CUDA out of memory|"
    r"RuntimeError:)"
)


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass
class Timing:
    median_ms: Optional[float] = None
    mean_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    min_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


@dataclass
class HarnessResult:
    """Everything the harness printed, plus how it exited.

    `ok` is not the same as `accuracy_passed`: a run can pass accuracy and then
    die in the benchmark, and a run launched with --benchmark-on-failure can
    fail accuracy and still produce timings worth looking at. The table shows
    both.
    """

    config_line: str = ""
    device: str = ""
    dtype: str = ""
    torch_version: str = ""
    gpu: str = ""
    criterion: str = ""

    accuracy_passed: Optional[bool] = None
    max_abs_error: Optional[float] = None
    max_relative_error: Optional[float] = None
    failed_elements: Optional[int] = None
    total_elements: Optional[int] = None
    trials: List[Dict[str, Any]] = field(default_factory=list)

    baseline: Timing = field(default_factory=Timing)
    optimized: Timing = field(default_factory=Timing)
    speedup: Optional[float] = None
    benchmark_skipped: bool = False

    # --stream-batch / --skip-baseline. `total_ms` is the summed cost of all
    # slices, i.e. one whole forward pass; `optimized.median_ms` is one slice.
    baseline_skipped: bool = False
    total_ms: Optional[float] = None
    slices: Optional[int] = None
    rows_per_slice: Optional[int] = None

    exit_code: Optional[int] = None
    error_lines: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def has_timing(self) -> bool:
        """True when the run produced numbers, with or without a speedup.

        Keyed on the optimized median rather than on the speedup: under
        --skip-baseline there is no speedup by construction, and treating that
        as "no timing" would blank a row that has perfectly good latencies in
        it.
        """
        return self.optimized.median_ms is not None

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["has_timing"] = self.has_timing
        return data


class HarnessParser:
    """Fed one line at a time; `result` is valid to read at any point."""

    def __init__(self) -> None:
        self.result = HarnessResult()

    def feed(self, line: str) -> None:
        line = line.rstrip("\r\n")
        stripped = line.strip()
        result = self.result

        match = _WHOLE_BATCH.match(stripped)
        if match:
            result.total_ms = _to_float(match.group(1))
            return

        match = _STREAMING.match(stripped)
        if match:
            result.slices = int(match.group(1))
            result.rows_per_slice = int(match.group(2))
            return

        if _BASELINE_SKIPPED.match(stripped):
            result.baseline_skipped = True
            return

        match = _TIMING.match(stripped)
        if match:
            timing = Timing(
                median_ms=_to_float(match.group(2)),
                mean_ms=_to_float(match.group(3)),
                p90_ms=_to_float(match.group(4)),
                min_ms=_to_float(match.group(5)),
                tokens_per_second=_to_float(match.group(6)),
            )
            if match.group(1) == "baseline":
                result.baseline = timing
            else:
                result.optimized = timing
            return

        match = _SPEEDUP.match(stripped)
        if match:
            result.speedup = _to_float(match.group(1))
            return

        match = _SUMMARY.match(stripped)
        if match:
            result.accuracy_passed = match.group(1) == "PASS"
            result.max_abs_error = _to_float(match.group(2))
            result.max_relative_error = _to_float(match.group(3))
            result.failed_elements = int(match.group(4))
            result.total_elements = int(match.group(5))
            return

        match = _TRIAL.match(stripped)
        if match:
            result.trials.append({
                "index": int(match.group(1)),
                "total": int(match.group(2)),
                "passed": match.group(3) == "PASS",
                "max_abs": _to_float(match.group(4)),
                "max_rel": _to_float(match.group(5)),
                "failed": int(match.group(6)),
                "elements": int(match.group(7)),
            })
            return

        match = _CONFIG.match(stripped)
        if match:
            result.config_line = stripped
            return

        match = _DEVICE.match(stripped)
        if match:
            result.device, result.dtype, result.torch_version = match.groups()
            return

        match = _GPU.match(stripped)
        if match:
            result.gpu = match.group(1)
            return

        match = _CRITERION.match(stripped)
        if match:
            result.criterion = match.group(1)
            return

        if _SKIPPED.match(stripped):
            result.benchmark_skipped = True
            return

        if stripped and _ERROR_HINT.search(stripped):
            # Keep a bounded slice: a torch traceback can run to dozens of
            # lines and the full text is in the log pane anyway.
            if len(result.error_lines) < 12:
                result.error_lines.append(stripped)

    def finish(self, exit_code: int) -> HarnessResult:
        self.result.exit_code = exit_code
        return self.result
