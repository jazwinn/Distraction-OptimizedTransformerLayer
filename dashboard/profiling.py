"""Nsight Systems, wrapped the way the rest of the dashboard wraps things.

The benchmark answers *how fast*. It cannot answer *why*, and the questions this
project is on now are why-questions: which kernel owns the time, whether a shape
is compute-bound or launch-bound, what an Amdahl ceiling actually is for a given
shape. Those are profiler answers, and until now they were obtained by hand.

Three facts about this machine shape everything here, all of them checked rather
than assumed:

* `nsys` profiles without elevation. `ncu` does not -- collecting hardware
  counters returns ERR_NVGPUCTRPERM unless the process is elevated or
  RmProfilingAdminOnly is cleared in the registry. So this module drives nsys,
  and reports ncu's state honestly instead of offering a button that fails.

* `nsys stats` emits CSV on stdout, one block per report, prefixed by a
  "Processing [...] with [...]" line. That is a line-oriented stream, which is
  exactly what jobs.py already knows how to consume, so StatsParser below
  implements the same tiny protocol HarnessParser does.

* GPU work is attributed to a model only through NVTX. Both models run in one
  process and share ATen and cuBLAS kernels by name; the ranges added to
  torch_transformer_benchmark.py under BENCH_NVTX=1 are the only thing that
  separates them.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import re
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Finding the tools
# --------------------------------------------------------------------------

_NVIDIA_DIR = r"C:\Program Files\NVIDIA Corporation"

# The reports the Profile view is built from. Ordered cheapest-looking first so
# the streamed output fills the page top-down as nsys works through them.
REPORTS = (
    "nvtx_kern_sum",        # kernels, split by model -- the table that matters
    "cuda_gpu_kern_sum",    # the same kernels without the split, as a control
    "cuda_api_sum",         # launch counts and their CPU cost
    "cuda_gpu_mem_time_sum",  # copies and memsets
    "nvtx_gpu_proj_trace",  # range boundaries, projected onto the GPU timeline
    "cuda_gpu_trace",       # every launch: start, duration, geometry
)


def _version_key(path: str) -> List[int]:
    """Sort 'Nsight Systems 2026.1.3' above 'Nsight Systems 2025.5.2'."""
    match = re.search(r"(\d+(?:\.\d+)*)", os.path.basename(path.rstrip("\\/")))
    if not match:
        return [0]
    return [int(part) for part in match.group(1).split(".")]


def _newest(pattern: str, *leaf: str) -> Optional[str]:
    """The newest install whose executable actually exists.

    Several Nsight versions coexist on a dev box -- three of Nsight Systems
    here -- and only some of them are complete. Checking for the binary rather
    than the directory keeps a half-removed install from winning the sort.
    """
    found = []
    for base in glob.glob(pattern):
        for tail in leaf:
            candidate = os.path.join(base, tail)
            if os.path.isfile(candidate):
                found.append((base, candidate))
                break
    if not found:
        return None
    found.sort(key=lambda pair: _version_key(pair[0]))
    return found[-1][1]


def find_nsys() -> Optional[str]:
    override = os.environ.get("NSYS_PATH")
    if override:
        return override if os.path.isfile(override) else None
    return _newest(os.path.join(_NVIDIA_DIR, "Nsight Systems *"),
                   os.path.join("target-windows-x64", "nsys.exe"))


def find_nsys_ui() -> Optional[str]:
    return _newest(os.path.join(_NVIDIA_DIR, "Nsight Systems *"),
                   os.path.join("host-windows-x64", "nsys-ui.exe"))


def find_ncu() -> Optional[str]:
    override = os.environ.get("NCU_PATH")
    if override:
        return override if os.path.isfile(override) else None
    return _newest(os.path.join(_NVIDIA_DIR, "Nsight Compute *"),
                   os.path.join("target", "windows-desktop-win7-x64", "ncu.exe"),
                   "ncu.bat")


def is_elevated() -> Optional[bool]:
    """Whether this process is running as administrator.

    ncu is spawned by this server and inherits its token, so the server's own
    elevation is what decides whether counter collection is permitted.
    """
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:                                   # noqa: BLE001
        return None


def _registry_permission() -> Tuple[Optional[bool], str]:
    """Whether a *non-elevated* process may read GPU performance counters.

    Read from the registry rather than by attempting a profile: the honest test
    costs a torch import and a CUDA context, which is far too much to spend on
    filling in a status line. Absent means admin-only, which is the NVIDIA
    default on Windows and the reason ncu returns ERR_NVGPUCTRPERM here.
    """
    key = r"HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"(Get-ItemProperty '{key}' -ErrorAction SilentlyContinue)"
             f".RmProfilingAdminOnly"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"could not read the registry: {error}"

    value = out.stdout.strip()
    if value == "0":
        return True, "RmProfilingAdminOnly is 0, so counters are open to any user"
    if value == "":
        return False, ("RmProfilingAdminOnly is not set, which means admin-only "
                       "-- the NVIDIA default on Windows")
    return False, f"RmProfilingAdminOnly is {value}, meaning admin-only"


# A CUDA program small enough that the profile costs only the torch import.
_NCU_PROBE_SOURCE = (
    "import torch; x = torch.zeros(1024, device='cuda'); "
    "x.relu_(); torch.cuda.synchronize()"
)


def probe_ncu(ncu: Optional[str] = None) -> Dict[str, Any]:
    """Actually collect one counter, and report what happened.

    The only honest test. Both routes to permission -- the registry value and
    elevation -- can look satisfied while collection still fails: the registry
    one until the driver reloads, which means a reboot.
    """
    ncu = ncu or find_ncu()
    if not ncu:
        return {"probed": True, "ok": False, "error": "Nsight Compute not found"}
    try:
        done = subprocess.run(
            [ncu, "--metrics", "sm__cycles_elapsed.avg", "--csv",
             sys.executable, "-c", _NCU_PROBE_SOURCE],
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"probed": True, "ok": False,
                "error": "ncu did not finish within 10 minutes"}
    except (OSError, subprocess.SubprocessError) as error:
        return {"probed": True, "ok": False, "error": f"{type(error).__name__}: {error}"}

    output = done.stdout + done.stderr
    if "ERR_NVGPUCTRPERM" in output:
        return {
            "probed": True, "ok": False, "code": "ERR_NVGPUCTRPERM",
            "error": "The driver refused counter access. If you have just set "
                     "RmProfilingAdminOnly, it takes effect only when the "
                     "driver reloads -- reboot. Running elevated works without "
                     "a reboot.",
        }
    # A row of CSV with the metric in it is the proof; a clean exit alone is not.
    if "sm__cycles_elapsed" in output:
        return {"probed": True, "ok": True, "error": ""}
    return {"probed": True, "ok": False,
            "error": (output.strip()[-800:] or f"ncu exited {done.returncode} "
                                               "with no output")}


def tool_status() -> Dict[str, Any]:
    """What is installed and what it will actually let us do."""
    nsys = find_nsys()
    ncu = find_ncu()
    elevated = is_elevated()
    open_to_all, why = _registry_permission() if ncu else (None, "")

    # Neither of these is a verdict. The registry value does nothing until the
    # driver reloads, and elevation is only inherited if this server has it, so
    # both can read "satisfied" while collection still fails. probe_ncu() is the
    # answer; this is only the explanation that goes with it.
    if elevated:
        why = ("this server is elevated, which should satisfy the "
               "administrator-only restriction; " + why)
    if open_to_all:
        why += (" -- note that this takes effect only after the driver "
                "reloads, so a reboot is needed if you set it recently")
    allowed = None
    return {
        "nsys": nsys,
        "nsys_ui": find_nsys_ui(),
        "nsys_available": bool(nsys),
        "ncu": ncu,
        "ncu_available": bool(ncu),
        "ncu_counters_allowed": allowed,
        "ncu_reason": why,
        "elevated": elevated,
        # Stated once, here, so the UI does not have to know the fix by heart.
        "ncu_fixes": [
            "Run the dashboard from an elevated terminal; ncu inherits that. "
            "Restart it after elevating -- this status is read once at startup.",
            "Or, as administrator, set RmProfilingAdminOnly (DWORD) to 0 under "
            "HKLM\\SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\Global\\NVTweak "
            "and reboot.",
        ],
    }


# --------------------------------------------------------------------------
# Building the commands
# --------------------------------------------------------------------------

SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_profile_shim.py")


def through_shim(argv: List[str]) -> List[str]:
    """Insert the shim between the interpreter and the script it is running.

    Without it torch cannot resolve the MSVC linker inside a traced process --
    `where cl` returns empty under injection -- so the extension fails to build
    and the harness silently profiles its SDPA fallback instead of the kernels
    under test. See _profile_shim.py for the mechanism.
    """
    for index, part in enumerate(argv):
        if part.endswith(".py"):
            return argv[:index] + [SHIM] + argv[index:]
    return argv


def capture_argv(nsys: str, argv: List[str], report_path: str) -> List[str]:
    """Wrap an existing harness command in an nsys capture.

    --sample none and --cpuctxsw none: CPU sampling and context-switch tracing
    cost real overhead and answer none of the questions here, all of which are
    about GPU work.

    --cuda-graph-trace=node is not optional. The optimized model captures its
    forward into a CUDA graph, and nsys defaults to recording a graph launch as
    a single opaque entry -- which made a whole transformer layer stack read as
    "2 kernels per forward" and hid every kernel inside the graph, including the
    ones this project wrote. At node granularity the graph's kernels are traced
    individually, which is the only way this view says anything true about a
    graphed model.
    """
    return [
        nsys, "profile",
        "--force-overwrite", "true",
        "-o", report_path,
        "--trace", "cuda,nvtx",
        "--cuda-graph-trace", "node",
        "--sample", "none",
        "--cpuctxsw", "none",
    ] + through_shim(argv)


def stats_argv(nsys: str, report_path: str) -> List[str]:
    # --force-export: nsys refuses to run, with a usage error rather than a
    # useful one, when a stale .sqlite sits beside the report. Re-exporting
    # every time costs one pass and removes the failure mode.
    argv = [nsys, "stats", "--format", "csv", "--force-export=true"]
    for report in REPORTS:
        argv.extend(["--report", report])
    argv.append(report_path)
    return argv


def capture_env(env: Dict[str, str]) -> Dict[str, str]:
    """The harness's environment plus the NVTX switch."""
    merged = dict(env)
    merged["BENCH_NVTX"] = "1"
    return merged


# --------------------------------------------------------------------------
# Parsing `nsys stats`
# --------------------------------------------------------------------------

_PROCESSING = re.compile(r"^Processing \[.*\] with \[.*[\\/](\w+)\.py\]")
_SKIPPED = re.compile(r"^SKIPPED:\s*(.*)$")


class StatsResult:
    """Tables keyed by report name, plus whatever can be derived from them."""

    def __init__(self) -> None:
        self.tables: Dict[str, List[Dict[str, str]]] = {}
        self.skipped: Dict[str, str] = {}
        self.error_lines: List[str] = []

    def as_dict(self) -> Dict[str, Any]:
        # cuda_gpu_trace can run to tens of thousands of rows. It is read to
        # derive the busy/idle figures and then dropped: shipping it to a
        # browser would cost megabytes per poll to display nothing.
        tables = {name: rows for name, rows in self.tables.items()
                  if name != "cuda_gpu_trace"}
        return {
            "tables": tables,
            "skipped": self.skipped,
            "analysis": analyse(self.tables),
            "error_lines": self.error_lines[-20:],
        }


class StatsParser:
    """`nsys stats` stdout -> tables, one line at a time.

    Same shape as parse.HarnessParser -- feed(), .result, finish() -- so
    jobs.py can stream this exactly as it streams a benchmark, and a long
    analysis fills the page as it goes rather than arriving all at once.
    """

    def __init__(self) -> None:
        self.result = StatsResult()
        self._report: Optional[str] = None
        self._header: Optional[List[str]] = None
        self._buffer: List[str] = []

    def feed(self, line: str) -> None:
        text = line.rstrip("\r\n")

        match = _PROCESSING.match(text)
        if match:
            self._flush()
            self._report = match.group(1)
            self._header = None
            return

        match = _SKIPPED.match(text)
        if match:
            if self._report:
                self.result.skipped[self._report] = match.group(1)
            self._report = None
            return

        if not self._report:
            return
        if not text.strip():
            # A blank line ends a block, but nsys also prints blank lines
            # between its own notices, so only a started table is closed here.
            if self._header is not None:
                self._flush()
                self._report = None
            return

        self._buffer.append(text)
        if self._header is None:
            self._header = next(csv.reader([text]))
            self._buffer = []
            self.result.tables.setdefault(self._report, [])

    def _flush(self) -> None:
        """Turn the buffered rows into dicts.

        Parsed with csv rather than str.split, because kernel names are C++
        signatures full of commas and quotes -- 'std::array<char *, 2>' is a
        single field, and splitting it makes silent nonsense of every column
        after it.
        """
        if not self._report or self._header is None or not self._buffer:
            self._buffer = []
            return
        rows = self.result.tables.setdefault(self._report, [])
        for record in csv.reader(io.StringIO("\n".join(self._buffer))):
            if not record:
                continue
            rows.append(dict(zip(self._header, record)))
        self._buffer = []

    def finish(self, returncode: int) -> StatsResult:
        self._flush()
        if returncode != 0:
            self.result.error_lines.append(
                f"nsys stats exited {returncode}")
        return self.result


# --------------------------------------------------------------------------
# Deriving the answers
# --------------------------------------------------------------------------

_KERNEL_DEF = re.compile(
    r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?(?:void\s+)?([A-Za-z_]\w*)")


def _own_kernel_names() -> List[str]:
    """Every __global__ in csrc/, read from csrc/.

    Hand-listing them missed gemm_bias_gelu_kernel -- the largest single kernel
    in the optimized model -- so the share attributed to this repository read
    half what it was. Reading the source means a kernel that is renamed or added
    is picked up without anyone remembering to update a pattern here.
    """
    names = set()
    csrc = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "csrc")
    try:
        entries = os.listdir(csrc)
    except OSError:
        return []
    for entry in entries:
        if not entry.endswith((".cu", ".cuh")):
            continue
        try:
            with open(os.path.join(csrc, entry), encoding="utf-8",
                      errors="replace") as handle:
                names.update(_KERNEL_DEF.findall(handle.read()))
        except OSError:
            continue
    names.discard("void")
    return sorted(names)


_OWN_NAMES = _own_kernel_names()
# The fallback matters when csrc/ cannot be read: better a rough answer than
# reporting every kernel as somebody else's.
_OURS = re.compile(
    "|".join(re.escape(name) for name in _OWN_NAMES) if _OWN_NAMES
    else r"wmma_|tile_|fused_attention|add_layernorm|gemm_bias_gelu",
    re.IGNORECASE,
)


def _number(row: Dict[str, str], *names: str) -> Optional[float]:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                return None
    return None


def _strip_domain(name: str) -> str:
    """':optimized' -> 'optimized', leaving other ranges alone.

    nsys writes a range as <domain>:<name>, and ours have no domain, so they
    arrive with a bare leading colon. Only that prefix is removed. Splitting on
    every colon instead would rename CUB's own ranges -- and a trace really does
    contain them, so 'CCCL:cub::DeviceReduce::Sum' became a range called 'Sum'.
    """
    if not name:
        return ""
    return name[1:] if name.startswith(":") else name


def _union_ns(intervals: List[Tuple[float, float]]) -> float:
    """Total time covered by these intervals, counting overlap once.

    Summing durations would be wrong: kernels on different streams run at the
    same time, and adding them produces a "busy" figure larger than the wall
    clock, which then reports negative idle time.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for begin, finish in ordered[1:]:
        if begin > end:
            total += end - start
            start, end = begin, finish
        elif finish > end:
            end = finish
    return total + (end - start)


def _launch_table(rows: List[Dict[str, str]]) -> List[Tuple[float, float, str]]:
    out = []
    for row in rows:
        start = _number(row, "Start (ns)")
        duration = _number(row, "Duration (ns)")
        if start is None or duration is None:
            continue
        out.append((start, duration, row.get("Name", "")))
    out.sort()
    return out


def _occupied(launches: List[Tuple[float, float, str]],
              window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Busy, idle and the worst gap, over the whole trace or one window."""
    if window:
        low, high = window
        chosen = [(s, d, n) for s, d, n in launches if s >= low and s + d <= high]
    else:
        chosen = launches
    if not chosen:
        return {}

    intervals = [(s, s + d) for s, d, _ in chosen]
    busy = _union_ns(intervals)
    span_start = window[0] if window else chosen[0][0]
    span_end = window[1] if window else max(end for _, end in intervals)
    wall = span_end - span_start

    # The worst stall, and what ran just before it -- the name is half the
    # diagnosis, since a gap after a tiny kernel is a launch problem and a gap
    # after a big one usually is not.
    worst = 0.0
    worst_after = ""
    reach = chosen[0][0] + chosen[0][1]
    previous = chosen[0][2]
    for start, duration, name in chosen[1:]:
        gap = start - reach
        if gap > worst:
            worst, worst_after = gap, previous
        if start + duration > reach:
            reach = start + duration
            previous = name
    return {
        "kernels": len(chosen),
        "busy_ns": busy,
        "wall_ns": wall,
        "idle_ns": max(0.0, wall - busy),
        "busy_fraction": (busy / wall) if wall > 0 else None,
        "largest_gap_ns": worst,
        "largest_gap_after": worst_after,
    }


def analyse(tables: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """The derived numbers the Profile view leads with."""
    analysis: Dict[str, Any] = {}

    # --- kernels, split by which model issued them ------------------------
    ranges: Dict[str, Dict[str, Any]] = {}
    for row in tables.get("nvtx_kern_sum", []):
        name = _strip_domain(row.get("NVTX Range", ""))
        if not name:
            name = "outside any range"
        total = _number(row, "Total Time (ns)") or 0.0
        bucket = ranges.setdefault(name, {"total_ns": 0.0, "kernels": []})
        bucket["total_ns"] += total
        bucket["kernels"].append({
            "name": row.get("Kernel Name", ""),
            "instances": int(_number(row, "Kern Inst") or 0),
            "total_ns": total,
            "avg_ns": _number(row, "Avg (ns)"),
            "med_ns": _number(row, "Med (ns)"),
            "min_ns": _number(row, "Min (ns)"),
            "max_ns": _number(row, "Max (ns)"),
        })
    for bucket in ranges.values():
        bucket["kernels"].sort(key=lambda k: k["total_ns"], reverse=True)
        whole = bucket["total_ns"] or 1.0
        for kernel in bucket["kernels"]:
            kernel["share"] = kernel["total_ns"] / whole
    analysis["ranges"] = ranges

    # Whether any of this repository's own kernels ran at all. A profile whose
    # optimized model is entirely ATen and cuBLAS is not a slow kernel -- it is
    # a kernel that never loaded, and saying so beats presenting a breakdown of
    # somebody else's code as if it were yours.
    optimized = ranges.get("optimized", {}).get("kernels", [])
    ours = [k for k in optimized if _OURS.search(k["name"])]
    analysis["custom_kernels"] = {
        "count": len(ours),
        "share": sum(k["share"] for k in ours),
        "names": [k["name"] for k in ours[:8]],
    }

    # --- launch overhead and idle time ------------------------------------
    launches = _launch_table(tables.get("cuda_gpu_trace", []))
    analysis["overall"] = _occupied(launches)

    # The same figures for the optimized model alone, which is the number
    # actually being optimized. Its ranges may be scattered through the trace,
    # so each is measured and the results added.
    # Projected, not push/pop. A range is closed on the CPU while the kernels it
    # queued are still running, so the push/pop timestamps exclude most of the
    # work they are meant to bound -- measured against a real trace, they caught
    # 2 kernels out of a forward's several dozen. nsys projects each range onto
    # the GPU timeline for this exact reason.
    windows = []
    for row in tables.get("nvtx_gpu_proj_trace", []):
        if _strip_domain(row.get("Name", "")) != "optimized":
            continue
        start = _number(row, "Projected Start (ns)")
        duration = _number(row, "Projected Duration (ns)")
        if start is None or duration is None:
            continue
        windows.append((start, start + duration))
    if windows and launches:
        parts = [_occupied(launches, window) for window in windows]
        parts = [part for part in parts if part]
        if parts:
            # The median forward, not the sum of all of them. The first time a
            # model runs it pays for cuBLAS library load and kernel module
            # loading -- on the probe trace that one forward was 7 ms of wall
            # around 33 us of kernel, and summing it in reported the model as
            # 1% busy. The harness reports median latency for the same reason;
            # this is the same discipline applied to the same problem.
            # The gap is taken from the median forward too, not the worst.
            # The first forward stalls for as long as the extension takes to
            # build and cuBLAS to load -- 24 s in one measured run -- and
            # reporting that as "the largest gap" says nothing about steady
            # state, which is the only regime worth optimizing.
            typical = sorted(parts, key=lambda part: part["wall_ns"])[len(parts) // 2]
            busy = statistics.median(part["busy_ns"] for part in parts)
            wall = statistics.median(part["wall_ns"] for part in parts)
            analysis["optimized"] = {
                "forwards": len(parts),
                "kernels": int(statistics.median(part["kernels"] for part in parts)),
                "busy_ns": busy,
                "wall_ns": wall,
                "idle_ns": max(0.0, wall - busy),
                "busy_fraction": (busy / wall) if wall > 0 else None,
                "largest_gap_ns": typical["largest_gap_ns"],
                "largest_gap_after": typical["largest_gap_after"],
                # Stated so the numbers above cannot be misread as a total.
                "per_forward": True,
                "slowest_wall_ns": max(part["wall_ns"] for part in parts),
            }

    # --- what the launches cost on the CPU --------------------------------
    for row in tables.get("cuda_api_sum", []):
        if row.get("Name") == "cudaLaunchKernel":
            analysis["launch_api"] = {
                "calls": int(_number(row, "Num Calls") or 0),
                "total_ns": _number(row, "Total Time (ns)"),
                "avg_ns": _number(row, "Avg (ns)"),
            }
            break

    return analysis


def human_ns(value: Optional[float]) -> str:
    if value is None:
        return "-"
    for unit, scale in (("ns", 1.0), ("us", 1e3), ("ms", 1e6), ("s", 1e9)):
        if abs(value) < scale * 1000 or unit == "s":
            return f"{value / scale:.2f} {unit}"
    return f"{value:.0f} ns"
