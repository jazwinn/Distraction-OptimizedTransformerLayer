"""
Loader for the custom CUDA extension in csrc/.

Import `get_kernels()` to lazily JIT-build and fetch the extension module.
Returns None if the build is unavailable (no MSVC on PATH, no CUDA, compile
error), so callers can fall back to a pure-PyTorch path rather than crashing.

Building needs MSVC (cl.exe), which is not on PATH in a plain shell. Rather
than requiring every invocation to go through scripts/devenv.bat, get_kernels()
calls _ensure_msvc_on_path() first, which runs the same vswhere/vcvarsall.bat
discovery devenv.bat does and merges the resulting environment into this
process. That makes a plain

    python torch_transformer_benchmark.py

work from any shell. scripts/devenv.bat still exists for cases that need the
compiler on PATH before Python even starts (e.g. invoking cl.exe directly).

Importing this module also preloads the display driver's GPU compiler, which is
what stops a run that touched a cuTile kernel from exiting 0xC0000005 after
printing its results. That only works if it happens before torch is imported,
so every entry point that can reach a tile kernel imports this module first --
see preload_tile_compiler() for the diagnosis and the evidence.
"""

from __future__ import annotations

import ctypes
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
_CSRC = os.path.join(_THIS_DIR, "csrc")
# tile_attention.cu is always compiled. Without TRANSFORMER_HAVE_TILE it builds
# into a stub that reports "unavailable", which keeps fused_attention.cu's
# reference to it resolvable whether or not this toolchain can do tiles.
_SOURCES = [
    os.path.join(_CSRC, "fused_attention.cu"),
    os.path.join(_CSRC, "tile_attention.cu"),
]

_kernels = None
_load_attempted = False
_load_error: Optional[BaseException] = None
_tile_enabled = False

# Handle to nvgpucomp64.dll, held for the life of the process. See
# preload_tile_compiler().
_tile_compiler = None
_tile_compiler_status = "not attempted"


def _nvgpucomp_candidates() -> List[str]:
    """Paths that might hold the driver's GPU compiler, newest first.

    Not resolvable by name: it lives in the DriverStore, which is not on the
    DLL search path, so ctypes.CDLL("nvgpucomp64.dll") fails. The folder name
    carries a per-install hash (nv_dispi.inf_amd64_<hash>) and changes with
    every driver update, so it is globbed rather than written down.
    """
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    system32 = os.path.join(system_root, "System32")
    patterns = [
        os.path.join(system32, "DriverStore", "FileRepository", "nv_disp*",
                     "nvgpucomp64.dll"),
        os.path.join(system32, "nvgpucomp64.dll"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found


def preload_tile_compiler() -> str:
    """Load the driver's GPU compiler before torch does, and keep it loaded.

    Works around an access violation (0xC0000005) raised *after* the
    interpreter has finished, in any process that has run a cuTile kernel. A
    run prints its results, reports success, and then exits 0xC0000005, so a
    passing run looks like a failing one to any caller that reads the exit
    code.

    The fault is not in this repo's kernels and not in CUDA. Windows Error
    Reporting names the faulting module every time, at the same offset:

        Faulting module name: nvgpucomp64.dll
        Exception code: 0xc0000005   Fault offset: 0x53ad3e

    -- the GPU compiler that ships with the display driver, pulled in because a
    tile object carries a *tile IR* fatbin section the driver finalises at
    module load. Loading nvgpucomp64.dll first, so that it is detached last,
    fixes it: 5/5 clean exits with the preload, 0/5 without. REPORT.md, under
    "Notes and known limits", has the full diagnosis and the four hypotheses
    that were ruled out on the way -- do not re-try those.

    Must run before ``import torch``, which is what puts the NVIDIA DLLs in the
    loader's list; preloading afterwards measures as no fix at all. That is why
    every entry point that can reach a tile kernel imports this module above
    torch, and why this runs at import rather than from get_kernels().

    Returns a short status string, also available from tile_compiler_status().
    Every failure is silent and harmless: without the preload the kernels still
    build, still run and still give the same numbers -- only the exit code of a
    tile run is wrong.
    """
    global _tile_compiler, _tile_compiler_status

    if _tile_compiler is not None:
        return _tile_compiler_status
    if sys.platform != "win32":
        _tile_compiler_status = "not needed (not windows)"
        return _tile_compiler_status

    # Late is not harmful, but it is not a fix either, and a caller staring at
    # a 0xC0000005 exit deserves to be told which of the two happened.
    late = "torch" in sys.modules

    candidates = _nvgpucomp_candidates()
    if not candidates:
        _tile_compiler_status = "not found"
        return _tile_compiler_status

    for path in candidates:
        try:
            _tile_compiler = ctypes.CDLL(path)
        except OSError:
            continue
        _tile_compiler_status = (
            f"loaded too late to help (torch was already imported): {path}"
            if late else f"loaded: {path}"
        )
        return _tile_compiler_status

    _tile_compiler_status = f"found but would not load: {candidates[0]}"
    return _tile_compiler_status


def tile_compiler_status() -> str:
    """What preload_tile_compiler() did, for diagnostics."""
    return _tile_compiler_status


# At import, not from get_kernels(): by the time anything calls get_kernels()
# torch is long since imported, which is too late for this to do anything.
preload_tile_compiler()


def _ensure_msvc_on_path() -> None:
    """If cl.exe isn't already reachable, locate it via vswhere + vcvarsall.bat
    and merge the resulting environment into this process.

    Mirrors scripts/devenv.bat: same vswhere query, same vcvarsall.bat call,
    same VCVARS_VER pin (CUDA 13.0 rejects the newest MSVC toolset outright,
    or worse, accepts it far enough that cudafe++ dies with an access
    violation). The difference is devenv.bat sets up the environment for a
    child process it launches; this sets it up for the interpreter already
    running, so no wrapper script is needed to invoke it.

    Silently gives up on any failure (vswhere missing, no VS install, no
    vcvarsall for the pinned toolset) and leaves cl.exe absent -- the existing
    check right after this call turns that into the usual clear error message.
    """
    if sys.platform != "win32" or shutil.which("cl") is not None:
        return

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(
        program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe"
    )
    if not os.path.isfile(vswhere):
        return

    try:
        vs_path = subprocess.check_output(
            [vswhere, "-products", "*", "-latest", "-property", "installationPath"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return
    if not vs_path:
        return

    vcvarsall = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        return

    vcvars_ver = os.environ.get("VCVARS_VER", "14.44")
    # "&& set" dumps the environment vcvarsall.bat left behind so it can be
    # parsed back out. shell=True is required both to run a .bat at all and
    # because building this as a ["cmd.exe", "/c", command] list instead lets
    # subprocess's own list2cmdline() re-quote the already-quoted vcvarsall
    # path, which cmd.exe then fails to parse.
    command = f'"{vcvarsall}" x64 -vcvars_ver={vcvars_ver} && set'
    try:
        output = subprocess.check_output(
            command, shell=True, text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, OSError):
        return

    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value


def _find_tile_cuda_home() -> Optional[str]:
    """Newest CUDA toolkit that ships <cuda_tile.h>, or None.

    The tile programming model needs CUDA 13.3+. That is usually *not* the
    toolkit torch was built against, and several toolkits are typically
    installed side by side, so the one to build with is found by looking for
    the header rather than by trusting whatever is first on PATH. Note that
    13.0/13.1 ship a `crt/cuda_tile.h` that is not this: the public
    `include/cuda_tile.h` is the one that matters.

    CUDA_TILE_HOME overrides the search.
    """
    override = os.environ.get("CUDA_TILE_HOME")
    if override:
        return override if os.path.isfile(os.path.join(override, "include", "cuda_tile.h")) else None

    roots = []
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        roots.append(os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA"))
    else:
        roots.append("/usr/local")

    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            home = os.path.join(root, entry)
            if os.path.isfile(os.path.join(home, "include", "cuda_tile.h")):
                # "v13.3" / "cuda-13.3" -> (13, 3) so the newest wins.
                digits = "".join(c if c.isdigit() or c == "." else " " for c in entry).split()
                version = tuple(int(p) for p in digits[-1].split(".")) if digits else (0,)
                candidates.append((version, home))

    return max(candidates)[1] if candidates else None


def tile_enabled() -> bool:
    """True when the last build actually compiled the cuTile kernel (impl=3)."""
    return _tile_enabled


# Process names that mean a build really is in flight. ninja is the one that
# carries the check: it stays resident for the whole build, so the gap between
# two compile steps does not read as an idle lock.
_BUILD_PROCESS_NAMES = (
    "ninja", "nvcc", "cicc", "ptxas", "cudafe++", "nvlink", "fatbinary", "cl",
)

# torch.utils.cpp_extension serialises concurrent builds with a FileBaton on
# <build_dir>/lock: whoever creates the file builds, everyone else sits in
# `while os.path.exists(lock): sleep(0.1)`. Nothing is recorded in the file, the
# wait has no timeout and staleness is never checked -- so a builder that dies
# before releasing it (Ctrl-C, the dashboard's Stop button, a killed subprocess)
# leaves the lock behind, and every later load() in every later process blocks
# forever at ~0.6% of a core. That looks exactly like a very slow compile, which
# is what makes it expensive: the tell is that no compiler process exists.
#
# The dashboard spawns a build step and a probe subprocess per run, so it wedges
# two processes per attempt and the probe only gives up at its own 10-minute
# timeout.
#
# So before handing control to load(), drop a lock that demonstrably has no
# owner. "No owner" needs two independent signals, because deleting a LIVE lock
# is worse than waiting on a dead one -- it lets two ninja invocations write the
# same object files at once:
#
#   age       torch writes the lock within a second of deciding to build, so
#             anything older than the threshold has long outlived that window
#   liveness  ninja is resident for the whole build, so if no compiler process
#             exists at all, nothing can be holding it
#
# If the liveness probe cannot run (no tasklist or ps, a sandbox), it says so
# rather than guessing, and we fall back to age alone at a threshold no build of
# this project comes close to.
_STALE_LOCK_SECONDS = 90
_STALE_LOCK_SECONDS_BLIND = 20 * 60


def _a_build_is_running() -> Optional[bool]:
    """True / False, or None when the check itself could not be made."""
    try:
        if sys.platform == "win32":
            proc = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                                  capture_output=True, text=True, timeout=20)
            names = tuple(n + ".exe" for n in _BUILD_PROCESS_NAMES)
        else:
            proc = subprocess.run(["ps", "-eo", "comm="],
                                  capture_output=True, text=True, timeout=20)
            names = _BUILD_PROCESS_NAMES
        if proc.returncode != 0:
            return None
    except Exception:
        return None
    listed = proc.stdout.lower()
    # A substring test, so a name that merely contains one of these counts as a
    # build. That errs toward waiting, which is the safe direction.
    return any(name in listed for name in names)


def _clear_stale_build_lock() -> None:
    lock_path = os.path.join(_BUILD_DIR, "lock")
    try:
        # getmtime, NOT getctime, and not Windows' CreationTime either. NTFS
        # file tunneling hands a file recreated in the same directory shortly
        # after a delete the *deleted* file's creation time -- measured here at
        # 12 minutes stale on a lock that was seconds old. Age off that and this
        # function would delete live locks. mtime is set when the baton creates
        # the file and is not tunnelled.
        age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return  # no lock, or it vanished under us -- either way nothing to do

    building = _a_build_is_running()
    if building is True:
        return
    if age < (_STALE_LOCK_SECONDS if building is False else _STALE_LOCK_SECONDS_BLIND):
        return

    # A third signal, free and exact, on Windows only: FileBaton keeps the fd
    # open for as long as it holds the lock (`release()` is close-then-remove),
    # so deleting a lock whose owner is alive fails with PermissionError and
    # deleting an orphaned one succeeds. Trying is therefore safe even if both
    # checks above were somehow wrong. POSIX unlinks open files happily, so
    # there the age and liveness gates are the whole of it.
    try:
        os.remove(lock_path)
    except OSError:
        return  # someone does own it after all -- leave it alone

    # Only now, having taken the real lock, is ninja's safe to drop too.
    try:
        os.remove(os.path.join(_BUILD_DIR, ".ninja_lock"))
    except OSError:
        pass

    # Never silent: deleting another process's lock is exactly the kind of thing
    # that should show up in the log of the run that did it.
    print(
        f"[kernel_ext] removed a stale build lock ({age:.0f}s old, no compiler "
        f"process running, not held open by anyone). A previous build was "
        f"interrupted before releasing it; every load() would otherwise have "
        f"waited on it forever.",
        file=sys.stderr,
    )


def get_kernels(verbose: bool = False):
    """JIT-build (once) and return the extension module, or None if unavailable."""
    global _kernels, _load_attempted, _load_error, _tile_enabled

    if _load_attempted:
        return _kernels
    _load_attempted = True

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Point the build at a tile-capable toolkit *before* importing
        # cpp_extension: it resolves CUDA_HOME once, at module import, and
        # every nvcc path in the generated ninja file comes from that.
        tile_home = _find_tile_cuda_home()
        if tile_home is not None:
            os.environ["CUDA_HOME"] = tile_home
            os.environ["PATH"] = os.path.join(tile_home, "bin") + os.pathsep + os.environ["PATH"]

        from torch.utils.cpp_extension import load

        _ensure_msvc_on_path()

        # Bail out before calling load() when the host compiler is missing.
        # torch probes for cl.exe itself, but it logs a full traceback at
        # WARNING level when the probe fails, which looks like a crash for what
        # is really just "not built, using the fallback". Checking first keeps
        # that noise out of a run that is working as intended.
        if sys.platform == "win32" and shutil.which("cl") is None:
            raise RuntimeError(
                "MSVC (cl.exe) could not be found on PATH or via vswhere, so "
                "the CUDA extension cannot be built. Install the \"Desktop "
                "development with C++\" workload for Visual Studio, or run "
                "through scripts/devenv.bat."
            )

        # Build for the card that is actually present rather than a hard-coded
        # arch: the tensor-core path needs SM 8.0+ to exist at all, and an
        # SM 8.6 cubin on an SM 8.9 card is compiled without knowing the
        # register file it will land on. PTX for the same virtual arch is
        # emitted alongside so the cached build still loads on another card of
        # the same family.
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        os.makedirs(_BUILD_DIR, exist_ok=True)
        # Before anything can block on it -- see _clear_stale_build_lock.
        _clear_stale_build_lock()

        base_cuda_flags = [
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
        ]

        def do_load(with_tile: bool):
            cuda_flags = list(base_cuda_flags)
            cflags = None
            if with_tile:
                cuda_flags += [
                    "-std=c++20",       # <cuda_tile.h> refuses anything older
                    "-enable-tile",     # without it __tile_global__ is ignored
                    "-DTRANSFORMER_HAVE_TILE",
                    # CUDA 13.3's CCCL headers break under MSVC's traditional
                    # preprocessor; this is the fix NVIDIA's own error names.
                    "-Xcompiler", "/Zc:preprocessor",
                ]
                cflags = ["/std:c++20", "/Zc:preprocessor"]
            return load(
                name="transformer_kernels",
                sources=_SOURCES,
                build_directory=_BUILD_DIR,
                extra_cflags=cflags,
                extra_cuda_cflags=cuda_flags,
                verbose=verbose,
            )

        if tile_home is not None:
            try:
                _kernels = do_load(with_tile=True)
                _tile_enabled = True
            except Exception as tile_error:
                # A tile-capable toolkit that still cannot build the tile
                # kernel must not cost us the other two. Retry without it;
                # tile_attention.cu then compiles to its declining stub.
                #
                # If the retry ALSO fails, report the tile build's error rather
                # than the retry's. An ordinary compile error in
                # fused_attention.cu fails both, and the retry's flags lack
                # /Zc:preprocessor -- so the second failure surfaces as CUDA
                # 13.3's CCCL preprocessor complaint and buries the real
                # message. That cost a debugging session once.
                try:
                    _kernels = do_load(with_tile=False)
                    _tile_enabled = False
                except Exception as plain_error:
                    raise RuntimeError(
                        f"{tile_error}\n\n"
                        f"[kernel_ext] the retry without cuTile also failed; "
                        f"its error is usually the less informative of the two "
                        f"and is omitted. Retry error type: "
                        f"{type(plain_error).__name__}"
                    ) from tile_error
        else:
            _kernels = do_load(with_tile=False)
            _tile_enabled = False
    except BaseException as exc:  # noqa: BLE001 - report anything, never crash the caller
        _kernels = None
        _load_error = exc

    return _kernels


def load_error() -> Optional[BaseException]:
    """Whatever went wrong during the last get_kernels() attempt, if anything."""
    return _load_error
