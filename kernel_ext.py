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

        os.makedirs(_BUILD_DIR, exist_ok=True)
        _kernels = load(
            name="transformer_kernels",
            sources=_SOURCES,
            build_directory=_BUILD_DIR,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_86,code=sm_86",  # RTX 3070 (SM 8.6)
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
