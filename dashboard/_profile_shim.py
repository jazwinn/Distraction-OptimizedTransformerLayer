"""Run the harness under a profiler with its MSVC lookup repaired.

torch's ninja generator resolves the MSVC linker by shelling out to
`where cl` (torch/utils/cpp_extension.py, the `rule link` block). Inside a
process Nsight Systems has injected, that subprocess returns **empty output** --
PATH is correct and `shutil.which("cl")` finds cl.exe, but `where.exe` itself
comes back blank. torch does not notice, because its guard is
`if len(cl_paths) >= 1`, which is true even for `[""]`; it takes
`os.path.dirname("")` and writes

    command = "/link.exe" $in /nologo $ldflags /out:$out

into build.ninja. The link fails, kernel_ext retries without cuTile, that retry
cannot compile on CUDA 13.3 either, `get_kernels()` returns None, and
`optimized/` falls back to SDPA **without reporting an error** -- so the profile
measures ATen and cuBLAS and looks entirely plausible.

This patches that one call and nothing else: when `where cl` comes back empty,
answer from `shutil.which` instead. It cannot change what is measured; it only
lets the extension under test actually load. Used solely for profile runs, and
visible in the command the dashboard previews before it runs.
"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys

_real_check_output = subprocess.check_output


def _is_where_cl(args) -> bool:
    try:
        parts = [str(part).lower() for part in args]
    except TypeError:
        return False
    return len(parts) == 2 and parts[0].endswith("where") and parts[1] == "cl"


def _check_output(args, *rest, **kwargs):
    """`where cl`, answered from the PATH this process can actually see."""
    try:
        out = _real_check_output(args, *rest, **kwargs)
    except subprocess.CalledProcessError:
        # `where` exits non-zero when it finds nothing, which under injection it
        # also does for things that are plainly on PATH.
        if not _is_where_cl(args):
            raise
        out = b""

    if not _is_where_cl(args) or (out and out.strip()):
        return out

    found = shutil.which("cl")
    if not found:
        return out
    text = found + "\r\n"
    # Match whatever the caller asked for; torch calls this without text=True.
    return text if isinstance(out, str) else text.encode()


subprocess.check_output = _check_output


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: _profile_shim.py <script.py> [args...]")
    script = os.path.abspath(sys.argv[1])
    # `python script.py` puts the script's directory on sys.path; runpy does
    # not, and the harness imports optimized/ and kernel_ext from there.
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
