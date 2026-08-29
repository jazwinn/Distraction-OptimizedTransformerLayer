"""The serial job queue, and the child processes it spawns.

One worker thread, one queue, and therefore **exactly one benchmark process
alive at any moment**. That is not a simplification -- it is the measurement
rule. Two runs sharing a GPU contend for SMs and for memory bandwidth and both
report numbers that mean nothing, and this project's own tuning notes put the
noise floor at ~4.3% with the card to itself. Queueing is what keeps a
dashboard from being a machine for generating plausible wrong numbers.

A job is a list of steps, each step one child process:

    single    1 step
    compare   2 steps, or 3 with the control
    sweep     one step per shape
    script    1 step, output not parsed

Steps run in order and the table fills in as each finishes, so a long sweep is
readable while it is still going.

Stop is a process-tree kill. `python torch_transformer_benchmark.py` can spawn
ninja and cl.exe underneath it when the extension needs rebuilding, and killing
only the parent would leave a compiler running and the GPU still busy -- so on
Windows this shells out to `taskkill /T`, which is the only reliable way to get
the whole tree.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .parse import HarnessParser
from .runspec import RunSpec

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
HISTORY_PATH = os.path.join(RUNS_DIR, "history.jsonl")

# Which child is running right now, so a server that dies without unwinding
# leaves a trail. shutdown() covers Ctrl-C; nothing in-process can cover a
# force-kill, a power loss or a segfault, and the child that outlives one of
# those keeps the GPU with no UI left to stop it from. reap_orphan() reads this
# at startup and cleans up.
RUNNING_PATH = os.path.join(RUNS_DIR, "running.json")

# Job ids carry a per-server-process stamp. A bare counter restarts at 1 with
# the server, and history.jsonl outlives it -- so two different runs would end
# up claiming the same id and the same runs/<id>.log, and following an old row
# back to its log would land on the wrong one.
SESSION = format(int(time.time()) % 46656, "0>3x")

# Lines kept in memory per job. A run that rebuilds the extension emits a few
# thousand; a runaway one could emit millions, and the browser has to render
# whatever it is handed. The full text is always on disk in runs/<id>.log.
MAX_LINES = 4000

# How long to wait for a killed process to actually die before giving up on it
# and marking the job stopped anyway.
KILL_TIMEOUT = 10.0

_IS_WINDOWS = os.name == "nt"


@dataclass
class Step:
    """One child process within a job."""

    spec: RunSpec
    label: str
    parse_output: bool = True
    # A step whose output is not the harness's supplies its own parser, as long
    # as it offers the same feed()/result/finish() surface HarnessParser does.
    # nsys stats uses this to stream its CSV in through the same machinery.
    parser_factory: Optional[Callable[[], Any]] = None
    # Filled in as it runs.
    status: str = "pending"          # pending | running | done | failed | stopped
    exit_code: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "command": self.spec.describe(),
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": (round(self.finished_at - self.started_at, 1)
                           if self.started_at and self.finished_at else None),
            "result": self.result,
            "meta": self.meta,
        }


class Job:
    """A queued unit of work, and the only thing the HTTP layer holds on to."""

    _ids = itertools.count(1)

    def __init__(self, mode: str, steps: List[Step], title: str = "") -> None:
        self.id = f"{SESSION}-{next(self._ids):03d}"
        self.mode = mode
        self.title = title or mode
        self.steps = steps
        self.status = "queued"       # queued | running | done | failed | stopped
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.lines: List[str] = []
        self.truncated = 0
        self.error: Optional[str] = None

        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._log_file = None

    # --- log plumbing ----------------------------------------------------

    def _open_log(self) -> None:
        os.makedirs(RUNS_DIR, exist_ok=True)
        try:
            self._log_file = open(os.path.join(RUNS_DIR, f"{self.id}.log"),
                                  "w", encoding="utf-8", errors="replace")
        except OSError:
            self._log_file = None

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def append(self, line: str) -> None:
        with self._lock:
            if self._log_file is not None:
                try:
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except OSError:
                    pass
            if len(self.lines) >= MAX_LINES:
                # Drop from the front: the tail is what someone watching a
                # running job is looking at.
                drop = MAX_LINES // 4
                del self.lines[:drop]
                self.truncated += drop
            self.lines.append(line)

    def snapshot(self, since: int = 0) -> Dict[str, Any]:
        """State plus whatever log lines the caller has not seen yet.

        `since` counts lines emitted over the job's whole life, so it stays
        correct across the truncation above -- a client that falls behind gets
        told how many lines it missed rather than being silently resynced.
        """
        with self._lock:
            emitted = self.truncated + len(self.lines)
            start = max(0, since - self.truncated)
            new = self.lines[start:] if since < emitted else []
            missed = max(0, self.truncated - since)
            return {
                "id": self.id,
                "mode": self.mode,
                "title": self.title,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "duration_s": (round((self.finished_at or time.time())
                                     - self.started_at, 1)
                               if self.started_at else None),
                "steps": [step.as_dict() for step in self.steps],
                "lines": new,
                "next_line": emitted,
                "missed_lines": missed,
            }

    # --- execution -------------------------------------------------------

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            process = self._process
        if process is not None and process.poll() is None:
            _kill_tree(process)

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def run(self) -> None:
        self.status = "running"
        self.started_at = time.time()
        self._open_log()
        try:
            for step in self.steps:
                if self.stop_requested:
                    step.status = "stopped"
                    continue
                self._run_step(step)
                if step.status == "stopped":
                    break
            if self.stop_requested:
                self.status = "stopped"
            elif any(step.status == "failed" for step in self.steps):
                self.status = "failed"
            else:
                self.status = "done"
        except Exception as exc:                       # noqa: BLE001
            # A bug in this file must not take the server down with it, and the
            # user needs to see why their job vanished.
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.append(f"[dashboard] {self.error}")
        finally:
            self.finished_at = time.time()
            self._close_log()
            _record_history(self)

    def _run_step(self, step: Step) -> None:
        step.status = "running"
        step.started_at = time.time()

        self.append("")
        self.append(f"[dashboard] === {step.label} ===")
        self.append(f"[dashboard] {step.spec.describe()}")
        self.append("")

        environment = os.environ.copy()
        environment.update(step.spec.env)
        # Piped stdout is block-buffered; -u on the command line plus this makes
        # the log pane fill in live instead of all at once at the end.
        environment["PYTHONUNBUFFERED"] = "1"

        if step.parser_factory is not None:
            parser = step.parser_factory()
        else:
            parser = HarnessParser() if step.parse_output else None

        try:
            process = subprocess.Popen(
                step.spec.argv,
                cwd=step.spec.cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if _IS_WINDOWS else 0),
            )
        except OSError as exc:
            step.status = "failed"
            step.exit_code = -1
            step.finished_at = time.time()
            self.append(f"[dashboard] could not start: {exc}")
            return

        # Publish the handle and re-read the stop flag under the same lock.
        # request_stop() can land between run()'s check at the top of the step
        # and the Popen above, and it kills only what _process points at -- so
        # without this re-check that child spawns and runs to completion with
        # the stop flag already set, which reads as "Stop did nothing".
        with self._lock:
            self._process = process
            stop_now = self._stop_requested
        _mark_running(process.pid, step.spec.argv, self.id)
        if stop_now:
            _kill_tree(process)

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\r\n")
            self.append(line)
            if parser is not None:
                parser.feed(line)
                # Publish as it goes, so the row fills in mid-run rather than
                # appearing only once the process exits.
                step.result = parser.result.as_dict()

        process.wait()
        with self._lock:
            self._process = None
        _clear_running()

        step.exit_code = process.returncode
        step.finished_at = time.time()
        if parser is not None:
            step.result = parser.finish(process.returncode).as_dict()

        if self.stop_requested:
            step.status = "stopped"
        elif process.returncode == 0:
            step.status = "done"
        elif (process.returncode == 2 and parser is not None
              and step.parser_factory is None):
            # The harness's documented "accuracy failed" exit. The run produced
            # a real verdict, so this is a result, not a crash.
            step.status = "done"
        else:
            step.status = "failed"

        self.append(f"[dashboard] {step.label} exited {process.returncode} "
                    f"after {step.finished_at - step.started_at:.1f}s")


def _mark_running(pid: int, argv: List[str], job_id: str) -> None:
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        with open(RUNNING_PATH, "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "argv": argv, "job": job_id,
                       "at": time.time()}, handle)
    except (OSError, TypeError, ValueError):
        pass


def _clear_running() -> None:
    try:
        os.remove(RUNNING_PATH)
    except OSError:
        pass


def _command_line_of(pid: int) -> Optional[str]:
    """The command line of a live process, or None if it is gone.

    Used to tell "our orphan" from "some unrelated process that happens to have
    inherited that pid". Windows recycles pids freely, so killing on a pid
    match alone could take out something innocent.
    """
    if _IS_WINDOWS:
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}')"
                 f".CommandLine"],
                capture_output=True, text=True, timeout=15)
            line = completed.stdout.strip()
            return line or None
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            # /proc/<pid>/cmdline separates arguments with NULs.
            return handle.read().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return None


def reap_orphan() -> Optional[str]:
    """Kill a benchmark left behind by a server that did not shut down cleanly.

    Returns a description of what was killed, or None. Identification is by pid
    *and* command line: the pid alone is not enough, because the operating
    system will have handed it to something else by now if our child is gone.
    """
    try:
        with open(RUNNING_PATH, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        pid = int(record["pid"])
        argv = record.get("argv") or []
    except (OSError, ValueError, KeyError, TypeError):
        _clear_running()
        return None

    command = _command_line_of(pid)
    if command is None:
        _clear_running()
        return None

    # The script path is the distinctive part; if the live process is not
    # running the same one, the pid was reused and this is not ours.
    script = next((part for part in argv if part.endswith(".py")), None)
    if not script or os.path.basename(script) not in command:
        _clear_running()
        return None

    try:
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=KILL_TIMEOUT)
        else:
            os.kill(pid, 9)
    except (OSError, subprocess.SubprocessError):
        return f"pid {pid} ({os.path.basename(script)}) could not be killed"
    finally:
        _clear_running()
    return f"pid {pid} running {os.path.basename(script)}"


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill a child and everything under it.

    terminate() alone is not enough here: a harness run that needs to rebuild
    the extension has ninja and cl.exe below it, and orphaning those leaves a
    compiler chewing the machine after the UI says the job stopped.
    """
    if _IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=KILL_TIMEOUT)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        process.terminate()
        process.wait(timeout=KILL_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _record_history(job: Job) -> None:
    """Append a one-line summary of a finished job to runs/history.jsonl.

    Summary only -- the full log is already in runs/<id>.log. Losing a history
    line is not worth failing a job over, so every error here is swallowed.
    """
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        entry = {
            "id": job.id,
            "mode": job.mode,
            "title": job.title,
            "status": job.status,
            "created_at": job.created_at,
            "duration_s": (round(job.finished_at - job.started_at, 1)
                           if job.started_at and job.finished_at else None),
            "steps": [step.as_dict() for step in job.steps],
        }
        with open(HISTORY_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def read_history(limit: int = 50) -> List[Dict[str, Any]]:
    """The most recent finished jobs, newest first."""
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    entries = []
    for line in reversed(lines):
        if len(entries) >= limit:
            break
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


class JobQueue:
    """One worker thread, so at most one benchmark process exists at a time."""

    def __init__(self, keep: int = 60) -> None:
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._keep = keep
        self._lock = threading.Lock()
        self._current: Optional[Job] = None
        self._worker = threading.Thread(target=self._loop, daemon=True,
                                        name="dashboard-runner")
        self._worker.start()

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                self._current = job
            try:
                if job.stop_requested:
                    # Stopped while still queued, so run() never executed and
                    # never recorded it. Do both here, otherwise the job just
                    # disappears from the history table.
                    job.status = "stopped"
                    job.started_at = job.started_at or time.time()
                    job.finished_at = time.time()
                    for step in job.steps:
                        step.status = "stopped"
                    _record_history(job)
                else:
                    job.run()
            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()

    _FINISHED = ("done", "failed", "stopped")

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Bound the in-memory set; the log files and history survive on
            # disk. Only finished jobs are evictable -- dropping one that is
            # still queued or running would leave it unstoppable, with its
            # /api/job poll answering 404 while the process kept going.
            evictable = [i for i in self._order
                         if i in self._jobs
                         and self._jobs[i].status in self._FINISHED]
            excess = len(self._order) - self._keep
            for stale in evictable[:max(0, excess)]:
                self._order.remove(stale)
                self._jobs.pop(stale, None)
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            current = self._current
            recent = [self._jobs[i] for i in reversed(self._order[-12:])
                      if i in self._jobs]
        return {
            "running": current.id if current else None,
            "queued": self._queue.qsize(),
            "recent": [{"id": job.id, "title": job.title, "mode": job.mode,
                        "status": job.status, "created_at": job.created_at}
                       for job in recent],
        }

    def stop(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in self._FINISHED:
            return False
        job.request_stop()
        return True

    def shutdown(self) -> None:
        """Stop whatever is running, on the way out.

        Without this, Ctrl-C on the server leaves the benchmark child alive:
        the worker thread is a daemon so the interpreter exits without waiting,
        but the child was spawned by the OS, not by that thread, and keeps
        running -- still holding the GPU, with the dashboard that started it
        gone and no way left to stop it from the UI.
        """
        with self._lock:
            current = self._current
            pending = [self._jobs[i] for i in self._order
                       if i in self._jobs
                       and self._jobs[i].status not in self._FINISHED]
        for job in pending:
            job.request_stop()
        if current is not None:
            current.request_stop()
