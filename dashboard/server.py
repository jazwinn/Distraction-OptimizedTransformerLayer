"""The HTTP layer: JSON routes plus the static page.

Loopback only, always. This server's entire purpose is to run programs on
request, so exposing it on a routable address would be handing the machine out.
`__main__.py` binds 127.0.0.1 and there is no option to change it.

The routes are small on purpose -- everything interesting happens in jobs.py
and runspec.py, and this file is the boundary that validates what comes in:

    GET  /                     the page
    GET  /static/<file>        its assets
    GET  /api/spec             flags, knobs, presets, scripts, gpu
    POST /api/preflight        issues + memory estimate + previewed command
    POST /api/run              queue a job              -> {id}
    GET  /api/job/<id>?since=N state and new log lines
    POST /api/job/<id>/stop    kill the process tree
    GET  /api/queue            what is running and what is waiting
    GET  /api/history          finished jobs, newest first
    GET  /api/gpu              nvidia-smi, cached
    GET  /api/presets          saved shapes, with what can be derived
    POST /api/presets          validate and write presets.json
    POST /api/probe            does this build have cuTile support?

The client polls /api/job while a job runs. Polling rather than SSE is a
deliberate trade: the whole client is one file of vanilla JS with no reconnect
logic to get wrong, and a 500 ms poll against a local socket costs nothing next
to the benchmark it is watching.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from . import argspec, knobs, presets, profiling, runspec
from .jobs import Job, JobQueue, RUNS_DIR, Step, read_history

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
REPO = runspec.REPO

# The only address this ever binds. It is defined here rather than in
# __main__.py because _same_origin() needs it too, and two copies of "which
# host are we" is how one of them ends up wrong.
HOST = "127.0.0.1"

QUEUE = JobQueue()

# Body size cap. A form submission is a couple of kilobytes; anything near this
# is a mistake or an attack, and either way is not worth reading into memory.
MAX_BODY = 1 << 20

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


# --------------------------------------------------------------------------
# GPU status, cached
# --------------------------------------------------------------------------

# Two locks on purpose. _gpu_lock guards the cache dict and is held only for
# the moment it takes to read or write it; _gpu_fetch_lock serialises the
# nvidia-smi call itself, so several tabs polling at once share one invocation
# instead of each spawning their own -- which is what the cache was for.
_gpu_lock = threading.Lock()
_gpu_fetch_lock = threading.Lock()
_gpu_cache: Dict[str, Any] = {"at": 0.0, "value": None}
GPU_TTL = 1.5

_GPU_FIELDS = ("name", "memory.used", "memory.total", "utilization.gpu",
               "temperature.gpu")


def gpu_status() -> Dict[str, Any]:
    """nvidia-smi, at most once every GPU_TTL seconds however many tabs ask.

    Shelling out to nvidia-smi rather than asking torch is the whole point: it
    reports what the *card* is doing, including the benchmark child process,
    and it costs this server no CUDA context of its own.
    """
    def cached() -> Optional[Dict[str, Any]]:
        with _gpu_lock:
            if (_gpu_cache["value"] is not None
                    and time.time() - _gpu_cache["at"] < GPU_TTL):
                return _gpu_cache["value"]
        return None

    fresh = cached()
    if fresh is not None:
        return fresh

    with _gpu_fetch_lock:
        # Re-check: whoever held this lock has just refreshed the cache, and
        # this caller's own poll is now satisfied by their result.
        fresh = cached()
        if fresh is not None:
            return fresh
        return _fetch_gpu()


def _fetch_gpu() -> Dict[str, Any]:
    value: Dict[str, Any]
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        used, total = int(parts[1]), int(parts[2])
        value = {
            "available": True,
            "name": parts[0],
            "memory_used_mib": used,
            "memory_total_mib": total,
            "memory_total_bytes": total * 1024 * 1024,
            "utilization": int(parts[3]),
            "temperature": int(parts[4]),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        value = {"available": False}

    with _gpu_lock:
        _gpu_cache["at"] = time.time()
        _gpu_cache["value"] = value
    return value


# --------------------------------------------------------------------------
# Extension probe
# --------------------------------------------------------------------------

_probe_lock = threading.Lock()
_probe_cache: Optional[Dict[str, Any]] = None

# Runs in a throwaway process, so whatever CUDA context it creates dies with it.
# get_kernels() is what actually answers the question, and on a cold tree it
# compiles for over a minute -- hence the generous timeout and the fact that
# this is only ever triggered on request, never at startup.
_PROBE_SOURCE = """
import json, sys
sys.path.insert(0, r"{repo}")
info = {{"loaded": False, "tile": False, "error": "", "compiler": ""}}
try:
    import kernel_ext
    info["compiler"] = kernel_ext.tile_compiler_status()
    module = kernel_ext.get_kernels()
    info["loaded"] = module is not None
    info["tile"] = bool(kernel_ext.tile_enabled())
    error = kernel_ext.load_error()
    if error is not None:
        info["error"] = str(error)[:2000]
except Exception as exc:
    info["error"] = "{{}}: {{}}".format(type(exc).__name__, exc)[:2000]
print("PROBE" + json.dumps(info))
"""

# The same probe, but it exits non-zero when the extension did not load. The
# profile and counter jobs run this first precisely to guarantee the build; a
# step that reports success while the build failed sends them on to profile code
# that was never loaded.
_PREPARE_SOURCE = _PROBE_SOURCE + """
if not info["loaded"]:
    sys.stderr.write("[prepare] the CUDA extension did not build, so there is "
                     "nothing of csrc/ to profile\\n")
    raise SystemExit(3)
"""


def probe_extension(force: bool = False) -> Dict[str, Any]:
    """Whether the built extension loads, and whether it has cuTile support.

    Cached for the life of the server. The answer only changes when csrc/ is
    rebuilt, and the UI has a button to re-run it for exactly that case.
    """
    global _probe_cache
    with _probe_lock:
        if _probe_cache is not None and not force:
            return _probe_cache

    result: Dict[str, Any]
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", _PROBE_SOURCE.format(repo=REPO)],
            cwd=REPO, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("PROBE"):
                payload = json.loads(line[len("PROBE"):])
        if payload is None:
            result = {"probed": True, "loaded": False, "tile": None,
                      "error": (completed.stdout + completed.stderr)[-2000:]
                               or "probe produced no output"}
        else:
            result = {"probed": True, **payload, "tile": bool(payload["tile"])}
    except subprocess.TimeoutExpired:
        result = {"probed": True, "loaded": False, "tile": None,
                  "error": "probe timed out after 10 minutes (a cold build of "
                           "the tensor-core kernel takes about 70 s, so this "
                           "usually means the build is stuck)"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result = {"probed": True, "loaded": False, "tile": None,
                  "error": f"{type(exc).__name__}: {exc}"}

    with _probe_lock:
        _probe_cache = result
    return result


_TOOLS: Dict[str, Any] = {}
_TOOLS_LOCK = threading.Lock()


def tools_cached() -> Dict[str, Any]:
    """Which profilers exist and what they will let us do.

    Cached because it shells out to PowerShell for the counter-permission
    registry read, and the answer only changes when someone installs a tool or
    reboots after changing the key -- neither of which happens mid-session.
    """
    with _TOOLS_LOCK:
        if not _TOOLS:
            _TOOLS.update(profiling.tool_status())
        return dict(_TOOLS)


def probe_cached() -> Dict[str, Any]:
    with _probe_lock:
        return _probe_cache or {"probed": False, "loaded": None, "tile": None,
                                "error": ""}


# --------------------------------------------------------------------------
# scripts/ listing
# --------------------------------------------------------------------------

# Scripts whose runtime is measured in tens of minutes or worse, flagged in the
# UI so nobody starts one expecting a quick answer. tune_block_shapes builds a
# separate CUDA extension per candidate shape.
LONG_RUNNING = {"tune_block_shapes.py", "compare_backends.py",
                "tune_linear_gelu.py", "bench_shape14.py"}


def list_scripts() -> List[Dict[str, Any]]:
    directory = os.path.join(REPO, "scripts")
    try:
        names = sorted(name for name in os.listdir(directory)
                       if name.endswith(".py") and not name.startswith("_"))
    except OSError:
        return []
    entries = []
    for name in names:
        path = os.path.join(directory, name)
        fields = argspec.script_arguments(path)
        entries.append({
            "name": name,
            "summary": argspec.module_summary(path),
            "fields": fields,
            "has_args": bool(fields),
            "long_running": name in LONG_RUNNING,
        })
    return entries


def describe_presets() -> List[Dict[str, Any]]:
    """The saved shapes, each with what can be derived without touching the GPU.

    head_dim, token count, estimated footprint, and whether preflight would
    refuse it -- computed here so the shape list can say what will happen
    before anything is queued.

    The interesting cases are the ones the harness handles rather than refuses.
    Shape 14 streams as 32 slices of one row and skips its baseline, so it runs
    and produces latencies but no speedup; shape 6 fits whole at 6.1 GiB. Both
    facts are worth seeing beside the checkbox rather than discovering in a log.
    """
    data = presets.load()
    gpu = gpu_status()
    total_bytes = gpu.get("memory_total_bytes") if gpu.get("available") else None
    tile = probe_cached().get("tile")

    described = []
    for preset in data["presets"]:
        form = {key: preset[key] for key in presets.SHAPE_KEYS if key in preset}
        form.setdefault("dtype", "float32")
        issues = knobs.preflight(form, tile, total_bytes)
        blockers = [i["message"] for i in issues if i["level"] == "error"]
        summary = knobs.estimate_summary(form, total_bytes)
        plan = summary["plan"]
        described.append({
            **preset,
            "head_dim": summary["head_dim"],
            "tokens": summary["tokens"],
            "memory": summary["human"]["peak"],
            "memory_bytes": plan["peak_bytes"],
            "slices": plan["slices"],
            "baseline_runs": plan["baseline_runs"],
            "baseline_chunk": plan["baseline_chunk"],
            "blocked": bool(blockers),
            "blocked_reason": blockers[0] if blockers else "",
            "notes": [i["message"] for i in issues if i["level"] == "warning"],
        })
    return described


# --------------------------------------------------------------------------
# Job construction
# --------------------------------------------------------------------------

def _blocking_issues(form: Dict[str, Any]) -> List[Dict[str, str]]:
    gpu = gpu_status()
    issues = knobs.preflight(
        form,
        tile_available=probe_cached().get("tile"),
        device_total_bytes=gpu.get("memory_total_bytes") if gpu.get("available") else None,
    )
    return issues


def _report_path(name: Any) -> Optional[str]:
    """Resolve a report name against runs/, or None.

    Matched against a directory listing rather than sanitised, the same way
    runspec.script_path() resolves a script: the check is then on the file that
    will actually be opened or deleted, and no spelling of a name reaches
    outside runs/.
    """
    if not isinstance(name, str) or not name:
        return None
    base = os.path.basename(name)
    try:
        entries = os.listdir(RUNS_DIR)
    except OSError:
        return None
    if base not in entries or not base.endswith(".nsys-rep"):
        return None
    full = os.path.join(RUNS_DIR, base)
    return full if os.path.isfile(full) else None


def report_info(name: str) -> Dict[str, Any]:
    """Size on disk, for a view that has to say what it is keeping."""
    path = _report_path(name)
    if path is None:
        return {"exists": False}
    total = 0
    for suffix in (".nsys-rep", ".sqlite"):
        mate = os.path.splitext(path)[0] + suffix
        try:
            total += os.path.getsize(mate)
        except OSError:
            pass
    return {"exists": True, "name": os.path.basename(path),
            "path": path, "bytes": total,
            "human": knobs.human_bytes(total)}


def _shape_label(form: Dict[str, Any]) -> str:
    def value(key: str, fallback: Any) -> Any:
        got = form.get(key)
        return fallback if got in (None, "") else got
    return (f"b{value('batch_size', 8)} s{value('seq_len', 128)} "
            f"d{value('d_model', 512)} h{value('heads', 8)}"
            + (" causal" if form.get("causal") else ""))


def _impl_label(form: Dict[str, Any]) -> str:
    parts = [str(form.get(key)) for key in
             ("attn_backend", "attn_impl", "attn_precision", "linear_gelu",
              "cuda_graph")
             if form.get(key) not in (None, "")]
    return " ".join(parts) if parts else "defaults"


# A traced run should be short. Tracing records every launch, and the analysis
# step then pays for each one -- a default benchmark run is tens of thousands of
# them, which turns a 10 s measurement into minutes of export for no extra
# insight. These are starting points; the form overrides any of them.
PROFILE_DEFAULTS: Dict[str, Any] = {
    "accuracy_trials": 1,
    "warmup": 1,
    "repeats": 5,
    "benchmark_rounds": 1,
}


def build_job(payload: Dict[str, Any]) -> Tuple[Optional[Job], List[Dict[str, str]]]:
    """Validate a /api/run payload and turn it into a queued-shaped Job.

    Returns (job, issues). A job is only returned when nothing at the "error"
    level came back from preflight -- warnings are passed through and shown,
    but do not block, because "this will run but will not measure what you
    think" is a judgement the person at the keyboard gets to make.
    """
    mode = payload.get("mode")

    def form_of(key: str) -> Optional[Dict[str, Any]]:
        value = payload.get(key)
        if value is None:
            return {}
        return value if isinstance(value, dict) else None

    bad_shape = [{"level": "error",
                  "message": "malformed request: the form fields must be objects"}]

    if mode == "script":
        name = payload.get("script")
        form = form_of("form")
        extra = payload.get("extra") or ""
        if form is None or not isinstance(name, str) or not isinstance(extra, str):
            return None, bad_shape
        spec = runspec.for_script(name, form, extra)
        if spec is None:
            return None, [{"level": "error",
                           "message": f"unknown script {name!r}"}]
        step = Step(spec=spec, label=name, parse_output=False)
        return Job("script", [step], title=name), []

    if mode == "single":
        form = form_of("form")
        if form is None:
            return None, bad_shape
        issues = _blocking_issues(form)
        if any(issue["level"] == "error" for issue in issues):
            return None, issues
        label = f"{_shape_label(form)} | {_impl_label(form)}"
        step = Step(spec=runspec.for_harness(form, label), label=label)
        step.meta = {"config": _impl_label(form), "shape": _shape_label(form)}
        return Job("single", [step], title=label), issues

    if mode == "compare":
        form_a = form_of("form_a")
        form_b = form_of("form_b")
        if form_a is None or form_b is None:
            return None, bad_shape
        shapes = payload.get("shapes") or []
        if not isinstance(shapes, list):
            return None, bad_shape
        want_control = bool(payload.get("control"))

        label_a = f"A: {_impl_label(form_a)}"
        label_b = f"B: {_impl_label(form_b)}"

        def sided(form: Dict[str, Any], side: str) -> List[Dict[str, str]]:
            return [dict(issue, message=f"{side}: {issue['message']}")
                    for issue in _blocking_issues(form)]

        def pair(merged_a: Dict[str, Any], merged_b: Dict[str, Any],
                 name: str) -> List[Step]:
            """The runs for one shape, in the order they execute.

            A, then B, then the control -- adjacent in time on purpose. Clocks
            and temperature drift over a long job; running the pair back to back
            moves both sides together instead of landing the drift on whichever
            config happened to be queued later.
            """
            made: List[Step] = []
            for role, form, fallback in (("A", merged_a, label_a),
                                         ("B", merged_b, label_b)):
                text = f"{name} · {role}" if name else fallback
                step = Step(spec=runspec.for_harness(form, text), label=text)
                step.meta = {"role": role, "config": _impl_label(form),
                             "shape": _shape_label(form), "preset": name}
                made.append(step)
            if want_control:
                # A second run of config A. Its ratio against the first A is the
                # noise floor for this machine right now -- the same control
                # column every A/B script in scripts/ prints. An A-vs-B
                # difference smaller than this one is not a result.
                #
                # It runs per shape because the floor is a property of the shape:
                # a 32-token sequence is far noisier than a long one, so a floor
                # borrowed from another row would make this row's verdict
                # over-confident.
                text = f"{name} · control" if name else "control: A again"
                step = Step(spec=runspec.for_harness(merged_a, text), label=text)
                step.meta = {"role": "control", "config": _impl_label(merged_a),
                             "shape": _shape_label(merged_a), "preset": name}
                made.append(step)
            return made

        if not shapes:
            issues = sided(form_a, "A") + sided(form_b, "B")
            if any(issue["level"] == "error" for issue in issues):
                return None, issues
            return (Job("compare", pair(form_a, form_b, ""),
                        title=f"{label_a} vs {label_b}"), issues)

        per_shape = 3 if want_control else 2
        if len(shapes) * per_shape > 96:
            return None, [{"level": "error",
                           "message": f"{len(shapes)} shapes at {per_shape} runs "
                                      f"each is {len(shapes) * per_shape} runs, "
                                      f"more than the 96 this will queue at once"}]

        steps = []
        issues = []
        kept = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            merged_a = dict(form_a)
            merged_b = dict(form_b)
            for key in presets.SHAPE_KEYS:
                if key in shape:
                    merged_a[key] = shape[key]
                    merged_b[key] = shape[key]
            name = shape.get("name") or _shape_label(merged_a)
            found = sided(merged_a, "A") + sided(merged_b, "B")
            if any(issue["level"] == "error" for issue in found):
                # Skip the whole shape, not the offending half: a pair with one
                # side missing is not a comparison. And skip rather than fail --
                # one uncovered head_dim should not cost the other thirteen rows.
                issues.extend(dict(issue, message=f"{name}: {issue['message']}")
                              for issue in found
                              if issue["level"] == "error")
                continue
            issues.extend(dict(issue, message=f"{name}: {issue['message']}")
                          for issue in found)
            steps.extend(pair(merged_a, merged_b, name))
            kept += 1

        if not steps:
            issues.append({"level": "error",
                           "message": "every selected shape was rejected"})
            return None, issues
        return (Job("compare", steps,
                    title=f"{label_a} vs {label_b} · {kept} shapes"), issues)

    if mode == "profile":
        form = form_of("form")
        if form is None:
            return None, bad_shape

        tools = tools_cached()
        if not tools["nsys_available"]:
            return None, [{"level": "error",
                           "message": "Nsight Systems was not found. Install it, "
                                      "or point NSYS_PATH at nsys.exe."}]

        merged = dict(PROFILE_DEFAULTS)
        merged.update({key: value for key, value in form.items()
                       if value not in (None, "")})
        issues = _blocking_issues(merged)
        if any(issue["level"] == "error" for issue in issues):
            return None, issues

        label = f"{_shape_label(merged)} | {_impl_label(merged)}"
        spec = runspec.for_harness(merged, label)
        report = os.path.join(RUNS_DIR, "PLACEHOLDER")

        # Build the extension first, in a process nsys is not watching. Under
        # the profiler the build fails -- torch's JIT loader runs ninja on every
        # load, nsys injects into it, and the compile dies -- and optimized/
        # then falls back to SDPA silently, so the profile measures ATen and
        # says nothing. A warm cache loads fine under the same profiler, so one
        # untraced build up front removes the whole failure mode.
        prepare_argv = [sys.executable, "-u", "-c", _PREPARE_SOURCE.format(repo=REPO)]
        if form.get("devenv"):
            prepare_argv = runspec.through_devenv(prepare_argv)

        prepare = Step(
            spec=runspec.RunSpec(
                argv=prepare_argv,
                cwd=REPO,
                label="prepare",
            ),
            label="prepare",
            parse_output=False,
            required=True,
        )
        prepare.meta = {"role": "prepare"}

        # Two steps, because `nsys stats` exports the trace to SQLite first and
        # on a real run that is slow enough to look like a hang. Split, the step
        # list says which half is working.
        # Under the profiler the extension's build can fail where it succeeds
        # untraced, and the harness then runs the fallback without saying so.
        # This is the escape hatch for that; the kernel table reports whether it
        # made any difference.
        capture_argv = profiling.capture_argv(tools["nsys"], spec.argv, report)
        if form.get("devenv"):
            capture_argv = runspec.through_devenv(capture_argv)

        capture = Step(
            spec=runspec.RunSpec(
                argv=capture_argv,
                env=profiling.capture_env(spec.env),
                cwd=spec.cwd,
                label="capture",
            ),
            label="capture",
            # Timings from a traced run are inflated and are not benchmark
            # numbers. Parsing them would put a speedup on screen that means
            # something different from every other speedup in this tool.
            parse_output=False,
        )
        capture.meta = {"role": "capture", "shape": _shape_label(merged),
                        "config": _impl_label(merged)}

        analyse = Step(
            spec=runspec.RunSpec(
                argv=profiling.stats_argv(tools["nsys"], report + ".nsys-rep"),
                cwd=spec.cwd,
                label="analyse",
            ),
            label="analyse",
            parser_factory=profiling.StatsParser,
        )
        analyse.meta = {"role": "analyse"}

        job = Job("profile", [prepare, capture, analyse], title=f"profile {label}")
        # The report is named after the job, which only exists once the job
        # does, so the placeholder is replaced now that the id is known.
        for step in job.steps:
            step.spec.argv = [part.replace("PLACEHOLDER", job.id)
                              for part in step.spec.argv]
        analyse.meta["report"] = os.path.join(RUNS_DIR, job.id + ".nsys-rep")
        return job, issues

    if mode == "ncu":
        form = form_of("form")
        if form is None:
            return None, bad_shape

        tools = tools_cached()
        if not tools["ncu_available"]:
            return None, [{"level": "error",
                           "message": "Nsight Compute was not found. Install it, "
                                      "or point NCU_PATH at ncu.exe."}]

        merged = dict(PROFILE_DEFAULTS)
        merged.update({key: value for key, value in form.items()
                       if value not in (None, "")})
        issues = _blocking_issues(merged)
        if any(issue["level"] == "error" for issue in issues):
            return None, issues

        try:
            launches = int(form.get("launch_count") or 12)
        except (TypeError, ValueError):
            launches = 12
        launches = max(1, min(launches, 200))

        label = f"{_shape_label(merged)} | {_impl_label(merged)}"
        spec = runspec.for_harness(merged, label)

        prepare_argv = [sys.executable, "-u", "-c", _PREPARE_SOURCE.format(repo=REPO)]
        if form.get("devenv"):
            prepare_argv = runspec.through_devenv(prepare_argv)
        prepare = Step(
            spec=runspec.RunSpec(argv=prepare_argv, cwd=REPO, label="prepare"),
            label="prepare",
            parse_output=False,
            required=True,
        )
        prepare.meta = {"role": "prepare"}

        detail = "full" if str(form.get("ncu_detail")) == "full" else "essential"
        collect_argv = profiling.ncu_argv(tools["ncu"], spec.argv,
                                          launch_count=launches, detail=detail)
        if form.get("devenv"):
            collect_argv = runspec.through_devenv(collect_argv)
        collect = Step(
            spec=runspec.RunSpec(
                argv=collect_argv,
                # NVTX costs nothing here and makes the ncu report readable in
                # the GUI if it is opened there later.
                env=profiling.capture_env(spec.env),
                cwd=spec.cwd,
                label="collect",
            ),
            label="collect",
            parser_factory=profiling.NcuParser,
        )
        collect.meta = {"role": "collect", "shape": _shape_label(merged),
                        "config": _impl_label(merged), "launches": launches,
                        "detail": detail}

        return (Job("ncu", [prepare, collect],
                    title=f"counters {label}"), issues)

    if mode == "sweep":
        form = form_of("form")
        if form is None:
            return None, bad_shape
        shapes = payload.get("shapes") or []
        if not isinstance(shapes, list) or not shapes:
            return None, [{"level": "error", "message": "no shapes selected"}]
        if len(shapes) > 64:
            return None, [{"level": "error",
                           "message": f"{len(shapes)} shapes is more than the "
                                      f"64 this will queue at once"}]

        steps: List[Step] = []
        issues: List[Dict[str, str]] = []
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            merged = dict(form)
            for key in presets.SHAPE_KEYS:
                if key in shape:
                    merged[key] = shape[key]
            name = shape.get("name") or _shape_label(merged)
            found = _blocking_issues(merged)
            if any(issue["level"] == "error" for issue in found):
                # Skip the shape rather than failing the whole sweep: one
                # uncovered head_dim should not cost the other thirteen rows.
                issues.extend(dict(issue, message=f"{name}: {issue['message']}")
                              for issue in found
                              if issue["level"] == "error")
                continue
            issues.extend(dict(issue, message=f"{name}: {issue['message']}")
                          for issue in found)
            step = Step(spec=runspec.for_harness(merged, name), label=name)
            step.meta = {"config": _impl_label(merged),
                         "shape": _shape_label(merged), "preset": name}
            steps.append(step)

        if not steps:
            issues.append({"level": "error",
                           "message": "every selected shape was rejected"})
            return None, issues
        return Job("sweep", steps,
                   title=f"sweep {len(steps)} shapes | {_impl_label(form)}"), issues

    return None, [{"level": "error", "message": f"unknown mode {mode!r}"}]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "BenchDashboard/1.0"
    protocol_version = "HTTP/1.1"

    # --- helpers ---------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        if status >= 400:
            # The request may have been rejected without its body being read.
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A tab closed mid-poll. Normal, and not worth a traceback.
            pass

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _same_origin(self) -> bool:
        """Reject a POST that a page on another site made the browser send.

        Binding to loopback keeps other machines out; it does nothing about the
        user's own browser, which will happily POST to 127.0.0.1 from any tab.
        Since this server runs programs on request, that is worth blocking.

        Two checks, and the first is the one that does the work. A cross-origin
        POST that skips the CORS preflight can only carry a Content-Type of
        form-urlencoded, multipart or text/plain -- never application/json. This
        server answers no preflight, so requiring JSON makes an unattended
        cross-site POST impossible. The Origin check is the belt to that:
        browsers set it on cross-origin requests, and curl does not set it at
        all, so an absent Origin stays fine.
        """
        content_type = (self.headers.get("Content-Type") or "").split(";")[0]
        if content_type.strip().lower() != "application/json":
            return False
        origin = self.headers.get("Origin")
        if origin:
            allowed = {f"http://{HOST}:{self.server.server_address[1]}",
                       f"http://localhost:{self.server.server_address[1]}"}
            if origin not in allowed:
                return False
        return True

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            # Leave the socket unusable rather than half-read: with keep-alive
            # on, an undrained body would be parsed as the next request line.
            self.close_connection = True
            return None
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _form(self, payload: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
        """A sub-object of the payload, or None when it is the wrong shape.

        The client always sends objects here, so anything else is a bug or a
        probe; either way it should come back as a 400 rather than as a
        TypeError inside the handler thread, which the browser only sees as a
        dropped connection.
        """
        value = payload.get(key)
        if value is None:
            return {}
        return value if isinstance(value, dict) else None

    def log_message(self, fmt: str, *args: Any) -> None:
        # The default logs every poll, which at 500 ms buries anything useful.
        return

    # --- routes ----------------------------------------------------------

    def _guard(self, handler) -> None:
        """Run a handler, turning any escaped exception into a JSON 500.

        Without this an exception propagates to socketserver, which closes the
        connection -- the browser reports a network failure and the traceback
        only exists in the terminal the server was started from. A 500 with the
        exception text in it is visible where the person is actually looking.
        """
        try:
            handler()
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:                                   # noqa: BLE001
                pass

    def do_GET(self) -> None:                                   # noqa: N802
        self._guard(self._get)

    def do_POST(self) -> None:                                  # noqa: N802
        self._guard(self._post)

    def _get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])

        if path == "/api/spec":
            flags = argspec.load()
            preset_data = presets.load()
            return self._json({
                "fields": flags["fields"],
                "flag_source": flags["source"],
                "env_knobs": knobs.ENV_KNOBS,
                "presets": describe_presets(),
                "presets_path": preset_data["path"],
                "presets_error": preset_data["error"],
                "scripts": list_scripts(),
                "gpu": gpu_status(),
                "extension": probe_cached(),
                "tools": tools_cached(),
                "profile_defaults": PROFILE_DEFAULTS,
                "repo": REPO,
                "shape_keys": list(presets.SHAPE_KEYS),
            })

        if path == "/api/presets":
            data = presets.load()
            return self._json({"presets": describe_presets(),
                               "path": data["path"], "error": data["error"]})

        if path == "/api/profile/report":
            name = (query.get("name") or [""])[0]
            return self._json(report_info(name))

        if path == "/api/gpu":
            return self._json(gpu_status())

        if path == "/api/queue":
            return self._json(QUEUE.status())

        if path == "/api/history":
            try:
                limit = int((query.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            return self._json({"entries": read_history(max(1, min(limit, 500)))})

        if path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            job = QUEUE.get(job_id)
            if job is None:
                return self._json({"error": "unknown job"}, 404)
            try:
                since = int((query.get("since") or ["0"])[0])
            except ValueError:
                since = 0
            return self._json(job.snapshot(max(0, since)))

        return self._json({"error": "not found"}, 404)

    def _post(self) -> None:
        path = urlparse(self.path).path

        if not self._same_origin():
            return self._json({"error": "this endpoint accepts only same-origin "
                                        "application/json requests"}, 403)

        if path == "/api/preflight":
            payload = self._read_json()
            if payload is None:
                return self._json({"error": "bad request"}, 400)
            form = self._form(payload, "form")
            if form is None:
                return self._json({"error": "'form' must be an object"}, 400)
            gpu = gpu_status()
            return self._json({
                "issues": knobs.preflight(
                    form,
                    tile_available=probe_cached().get("tile"),
                    device_total_bytes=(gpu.get("memory_total_bytes")
                                        if gpu.get("available") else None),
                ),
                "estimate": knobs.estimate_summary(
                    form, gpu.get("memory_total_bytes")
                    if gpu.get("available") else None),
                "command": runspec.for_harness(form).describe(),
            })

        if path == "/api/run":
            payload = self._read_json()
            if payload is None:
                return self._json({"error": "bad request"}, 400)
            job, issues = build_job(payload)
            if job is None:
                return self._json({"error": "rejected", "issues": issues}, 400)
            QUEUE.submit(job)
            return self._json({"id": job.id, "issues": issues,
                               "steps": len(job.steps)})

        if path == "/api/presets":
            payload = self._read_json()
            if payload is None:
                return self._json({"error": "bad request"}, 400)
            result = presets.save(payload.get("presets"))
            if not result["ok"]:
                return self._json({"error": result["error"] or "invalid presets",
                                   "problems": result["problems"]}, 400)
            # Hand back the described list, so the editor and every other tab
            # can refresh from one response rather than re-fetching.
            return self._json({"saved": result["count"],
                               "presets": describe_presets()})

        if path == "/api/probe":
            # Probing loads (and on a cold tree builds) the extension, which
            # takes the GPU. Running that beside a timed benchmark would
            # corrupt the very measurement the queue exists to protect.
            if QUEUE.status()["running"]:
                return self._json({"error": "a benchmark is running; probing "
                                            "would build CUDA alongside it and "
                                            "spoil the measurement"}, 409)
            return self._json(probe_extension(force=True))

        if path == "/api/profile/ncu-probe":
            # Takes the GPU for a few seconds, so it waits its turn like every
            # other thing here that opens a CUDA context.
            if QUEUE.status()["running"]:
                return self._json({"error": "a benchmark is running; probing "
                                            "would take the GPU alongside it"}, 409)
            result = profiling.probe_ncu()
            with _TOOLS_LOCK:
                _TOOLS["ncu_counters_allowed"] = result["ok"]
                if result.get("error"):
                    _TOOLS["ncu_reason"] = result["error"]
                elif result["ok"]:
                    _TOOLS["ncu_reason"] = "measured: ncu collected a counter"
            return self._json(result)

        if path in ("/api/profile/open", "/api/profile/delete"):
            report = _report_path(payload.get("report"))
            if report is None:
                return self._json({"error": "no such report"}, 404)
            if path.endswith("/open"):
                viewer = tools_cached().get("nsys_ui")
                if not viewer:
                    return self._json({"error": "Nsight Systems UI not found"}, 404)
                # Detached: the viewer outlives this request, and waiting on a
                # GUI would hold the handler thread open for as long as it is.
                subprocess.Popen([viewer, report], cwd=REPO)
                return self._json({"opened": report})
            removed = []
            for suffix in (".nsys-rep", ".sqlite"):
                mate = os.path.splitext(report)[0] + suffix
                try:
                    os.remove(mate)
                    removed.append(os.path.basename(mate))
                except OSError:
                    pass
            return self._json({"removed": removed})

        if path.startswith("/api/job/") and path.endswith("/stop"):
            job_id = path[len("/api/job/"):-len("/stop")]
            return self._json({"stopped": QUEUE.stop(job_id)})

        return self._json({"error": "not found"}, 404)

    # --- static ----------------------------------------------------------

    def _static(self, name: str) -> None:
        # Resolve and confirm containment rather than filtering the name: the
        # check is then on the path that will actually be opened.
        target = os.path.realpath(os.path.join(STATIC_DIR, name))
        root = os.path.realpath(STATIC_DIR)
        if not (target == root or target.startswith(root + os.sep)):
            return self._json({"error": "not found"}, 404)
        try:
            with open(target, "rb") as handle:
                body = handle.read()
        except OSError:
            return self._json({"error": "not found"}, 404)
        extension = os.path.splitext(target)[1].lower()
        self._send(200, body, CONTENT_TYPES.get(extension,
                                                "application/octet-stream"))


def serve(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
