"""
Loader for the custom CUDA extension in csrc/.

Import `get_kernels()` to lazily JIT-build and fetch the extension module.
Returns None if the build is unavailable (no MSVC on PATH, no CUDA, compile
error), so callers can fall back to a pure-PyTorch path rather than crashing.

Building needs MSVC (cl.exe) on PATH. On Windows that means running from an
"x64 Native Tools Command Prompt for VS 2022", or going through
scripts/devenv.bat which calls vcvarsall.bat first:

    cmd.exe /c scripts\\devenv.bat python torch_transformer_benchmark.py
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
_SOURCES = [os.path.join(_THIS_DIR, "csrc", "fused_attention.cu")]

_kernels = None
_load_attempted = False
_load_error: Optional[BaseException] = None


def get_kernels(verbose: bool = False):
    """JIT-build (once) and return the extension module, or None if unavailable."""
    global _kernels, _load_attempted, _load_error

    if _load_attempted:
        return _kernels
    _load_attempted = True

    try:
        import torch
        from torch.utils.cpp_extension import load

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Bail out before calling load() when the host compiler is missing.
        # torch probes for cl.exe itself, but it logs a full traceback at
        # WARNING level when the probe fails, which looks like a crash for what
        # is really just "not built, using the fallback". Checking first keeps
        # that noise out of a run that is working as intended.
        if sys.platform == "win32" and shutil.which("cl") is None:
            raise RuntimeError(
                "MSVC (cl.exe) is not on PATH, so the CUDA extension cannot be "
                "built. Run through scripts/devenv.bat to put it there."
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
        _kernels = load(
            name="transformer_kernels",
            sources=_SOURCES,
            build_directory=_BUILD_DIR,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                f"-gencode=arch=compute_{arch},code=compute_{arch}",
            ],
            verbose=verbose,
        )
    except BaseException as exc:  # noqa: BLE001 - report anything, never crash the caller
        _kernels = None
        _load_error = exc

    return _kernels


def load_error() -> Optional[BaseException]:
    """Whatever went wrong during the last get_kernels() attempt, if anything."""
    return _load_error
