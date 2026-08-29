"""Start the dashboard.

    python -m dashboard
    python -m dashboard --port 8123 --no-browser

Binds 127.0.0.1 and nothing else. The server exists to run programs on request,
so there is deliberately no flag to make it reachable from another machine.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from . import argspec, presets
from .jobs import reap_orphan
from .server import HOST, QUEUE, serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard",
        description="A local web dashboard for the transformer benchmark. "
                    "The server never imports torch and never touches the GPU; "
                    "each benchmark runs in a child process that exits.",
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="port to listen on (default: 8000); if it is "
                             "taken, the next few are tried")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    return parser.parse_args()


def _bind(preferred: int, attempts: int = 20):
    """The first free port at or after `preferred`.

    Restarting the dashboard while a browser tab is still polling can leave the
    old port in TIME_WAIT, and failing outright for that would be a poor
    welcome. The chosen port is printed either way.
    """
    last_error = None
    for offset in range(attempts):
        try:
            return serve(preferred + offset, HOST)
        except OSError as exc:
            last_error = exc
    raise SystemExit(
        f"could not bind any port in {preferred}-{preferred + attempts - 1} "
        f"on {HOST}: {last_error}"
    )


def main() -> int:
    args = parse_args()
    httpd = _bind(args.port)
    port = httpd.server_address[1]
    url = f"http://{HOST}:{port}"

    flags = argspec.load()
    preset_data = presets.load()

    # A previous server that was force-killed or crashed cannot have unwound,
    # so its benchmark may still be running and holding the card. Clear it
    # before anything else -- otherwise the first run of this session competes
    # with a ghost and every number it produces is wrong.
    orphan = reap_orphan()

    # flush=True throughout: stdout is block-buffered when redirected to a
    # file or a pipe, and a startup banner that only appears once the server
    # exits is worse than none.
    print(f"benchmark dashboard  {url}", flush=True)
    print(f"  flags   : {flags['count']} read from the harness source "
          f"({flags['source']})", flush=True)
    print(f"  presets : {len(preset_data['presets'])} from "
          f"{preset_data['path']}", flush=True)
    if preset_data["error"]:
        print(f"            [warning] {preset_data['error']}", flush=True)
    print("  gpu     : this server holds no CUDA context; benchmarks run in "
          "child processes", flush=True)
    if orphan:
        print(f"  cleanup : killed a benchmark left over from a previous "
              f"session ({orphan})", flush=True)
    print("  press Ctrl-C to stop", flush=True)

    if not args.no_browser:
        # Deferred so the message above is on screen before the tab steals
        # focus, and threaded so a machine with no default browser cannot hang
        # startup.
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        # Order matters: stop accepting requests, then kill any benchmark still
        # running. A child outliving the server would keep the GPU busy with
        # nothing left to stop it from -- the dashboard that owned it is gone.
        httpd.shutdown()
        httpd.server_close()
        if QUEUE.status()["running"]:
            print("  killing the benchmark still running", flush=True)
        QUEUE.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
