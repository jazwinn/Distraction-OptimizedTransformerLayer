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
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

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
            except Exception:
                # A tile-capable toolkit that still cannot build the tile
                # kernel must not cost us the other two. Retry without it;
                # tile_attention.cu then compiles to its declining stub.
                _kernels = do_load(with_tile=False)
                _tile_enabled = False
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
