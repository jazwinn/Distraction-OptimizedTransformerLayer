"""A local web dashboard that drives the existing benchmark harness.

Nothing in here imports torch, and nothing in here touches the GPU. The server
is stdlib-only and idles at 0 MB of VRAM; every measurement happens in a short
lived child process that exits and gives its memory back.

    python -m dashboard

Reading order, highest level first:

    __main__.py   argument parsing and startup
    server.py     the HTTP routes
    jobs.py       the serial job queue and the subprocesses it spawns
    runspec.py    a form submission -> an argv list and an environment
    argspec.py    the harness's own argparse calls, read out of the source
    knobs.py      the environment-variable knobs, which have no argparse
    parse.py      harness stdout -> a row of numbers
    presets.py    saved shapes

argspec, knobs, parse and presets import nothing from the rest of the package,
so any of them can be read on its own.
"""

__all__ = ["argspec", "jobs", "knobs", "parse", "presets", "runspec", "server"]
