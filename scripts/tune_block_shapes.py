"""Sweep the tf32 block shapes in csrc/tile_attention.cu and print the winners.

The shapes are compiled in, so each candidate needs its own build. TF32_M_*/
TF32_N_* are -D overrides precisely so this can search without editing the
source; the numbers it prints go back into BlockCfg<HEAD_DIM, MathMode::Tf32>.

tf32 is the mode where the two pressures pull apart: it costs the same 32 bits
per element as fp32 (so the spill cliff sits where fp32's does) but runs on the
MMA units (which the narrow fp32 shapes starve). Neither mode's swept shapes
transfer, hence this.

    python scripts/tune_tile_tf32.py            # dense shapes, all head_dims
    python scripts/tune_tile_tf32.py 32 64      # just these head_dims
    python scripts/tune_tile_tf32.py --causal   # the causal shapes instead

The causal kernel is swept separately because its grid is triangular: block m
walks m+1 key tiles rather than S/BLOCK_N, so BLOCK_M sets both how many blocks
there are and how uneven they are. The two mask modes do not want the same
shape -- at head_dim 64 the dense winner loses by 12-25% on causal cases.
Results go into TF32_M_*/TF32_N_* (dense) or TF32_CM_*/TF32_CN_* (causal).
"""

from __future__ import annotations

import itertools
import os
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import kernel_ext  # noqa: E402

HEAD_DIMS = (8, 16, 32, 64)

# cuTile requires every tile extent to be a power of two (`is_pow2` in
# crt/cuda_tile.h:749), so the candidate set is {16,32,64,128} and FlashAttention's
# tuned kBlockN values of 112 and 96 are not expressible here -- 64 and 128 are
# the legal neighbours. FA2 returns kBlockM=128 unconditionally on every arch and
# head_dim, which is why 128 is in the M set; its kBlockN for the headdim<=64
# bucket that all four of our head_dims fall into is 112, which is why 128 must
# be in the N set even though a first pass capped N at 64.
# Override with TUNE_CANDIDATES="128x64,128x128" to search a focused region.
_DEFAULT_CANDIDATES = tuple(itertools.product((16, 32, 64, 128), (16, 32, 64, 128)))
_env = os.environ.get("TUNE_CANDIDATES", "").strip()
CANDIDATES = (tuple(tuple(int(x) for x in c.split("x")) for c in _env.split(","))
              if _env else _DEFAULT_CANDIDATES)

# Shapes the kernel is actually used at. Timed together so a shape is chosen for
# aggregate behaviour rather than for one lucky sequence length.
#
# The short case is not optional. A first pass swept only 512 and 2048 and chose
# BLOCK_M=128 at head_dim 64; at seq_len 128 that is a single query tile, so the
# grid collapses to batch*heads = 64 blocks on a 46-SM card and the kernel loses
# to shapes that were worse on long sequences. Occupancy at the short end and
# the spill cliff at the long end pull in opposite directions, so both have to
# be in the score or the winner is tuned for half the workload.
ALL_CASES = (
    # (batch, heads, seq_len, causal)
    (8, 8, 128, False),
    (8, 8, 128, True),
    (4, 8, 512, False),
    (4, 8, 512, True),
    (1, 8, 2048, False),
    (1, 8, 2048, True),
)

# --causal scores only the causal cases and writes the TF32_CM_*/TF32_CN_*
# macros; the default scores only the dense ones. Scoring a shape on cases it
# will never run is how a single shape ended up serving both mask modes.
CAUSAL = "--causal" in sys.argv
CASES = tuple(c for c in ALL_CASES if c[3] == CAUSAL)
PREFIX = "TF32_CM" if CAUSAL else "TF32_M"
NPREFIX = "TF32_CN" if CAUSAL else "TF32_N"

TILE_TF32 = 5


def build(name: str, defines: dict[str, int], workdir: str):
    """Build the extension with these -D overrides into a private directory.

    The module name must be unique per candidate: load() imports by name and
    Python caches in sys.modules, so reusing one name would silently hand back
    the first build for every subsequent shape.
    """
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    flags = [
        "-O3", "--use_fast_math",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-std=c++20", "-enable-tile", "-DTRANSFORMER_HAVE_TILE",
        "-Xcompiler", "/Zc:preprocessor",
    ]
    flags += [f"-D{k}={v}" for k, v in defines.items()]
    return load(
        name=name,
        sources=[os.path.join(kernel_ext._CSRC, "fused_attention.cu"),
                 os.path.join(kernel_ext._CSRC, "tile_attention.cu")],
        build_directory=workdir,
        extra_cflags=["/std:c++20", "/Zc:preprocessor"],
        extra_cuda_cflags=flags,
        verbose=False,
    )


def time_ms(fn, iters: int = 30) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(True) for _ in range(iters)]
    ends = [torch.cuda.Event(True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def make_cases(head_dim: int):
    """Materialise inputs once, so timing does not include allocation."""
    dev = torch.device("cuda")
    out = []
    for b, h, sl, causal in CASES:
        q, k, v = (torch.randn(b, h, sl, head_dim, device=dev) for _ in range(3))
        out.append(((q, k, v), causal, head_dim ** -0.5))
    return out


def time_round(mod, cases) -> float:
    """One pass over every case; total ms, or inf if this build declines them."""
    total = 0.0
    try:
        with torch.inference_mode():
            for (q, k, v), causal, scale in cases:
                total += time_ms(lambda: mod.fused_attention_forward(
                    q, k, v, None, causal, scale, TILE_TF32), iters=15)
    except RuntimeError:
        return float("inf")
    return total


def score_all(mods: dict, head_dim: int, rounds: int = 5) -> dict:
    """Time every candidate round-robin; return each one's best round.

    Timing candidate A to completion and then candidate B measures whatever
    else changed between them as much as the shapes themselves: run-to-run
    variance here was large enough to invert rankings. Cycling all candidates
    within each round spreads that variance across them evenly, and taking
    each candidate's best round approximates its uncontended cost.

    Corollary: numbers from two separate runs of this script are NOT
    comparable. Any candidate you want to rank must be in the same run.
    """
    cases = make_cases(head_dim)
    best = {name: float("inf") for name in mods}
    for _ in range(rounds):
        for name, mod in mods.items():
            best[name] = min(best[name], time_round(mod, cases))
    return best


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    kernel_ext._ensure_msvc_on_path()
    tile_home = kernel_ext._find_tile_cuda_home()
    if tile_home is None:
        print("no CUDA toolkit with <cuda_tile.h> (needs 13.3+)")
        return 1
    os.environ["CUDA_HOME"] = tile_home
    os.environ["PATH"] = os.path.join(tile_home, "bin") + os.pathsep + os.environ["PATH"]

    wanted = [int(a) for a in sys.argv[1:] if not a.startswith("-")] or list(HEAD_DIMS)
    print(f"{torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}  "
          f"{'causal' if CAUSAL else 'dense'} shapes")

    winners = {}
    for hd in wanted:
        print(f"\nhead_dim {hd}")
        # Build every candidate before timing any of them, so the measurement
        # phase is not interrupted by minute-long nvcc runs.
        mods, workdirs = {}, []
        for m, n in CANDIDATES:
            # Only the head_dim under test varies; the others stay at a shape
            # known to compile so the build stays valid.
            defines = {f"{PREFIX}_{d}": (m if d == hd else 64) for d in HEAD_DIMS}
            defines.update({f"{NPREFIX}_{d}": (n if d == hd else 64) for d in HEAD_DIMS})
            workdir = tempfile.mkdtemp(prefix=f"tf32_{hd}_{m}x{n}_")
            workdirs.append(workdir)
            try:
                mods[f"{m}x{n}"] = build(f"tile_tf32_{hd}_{m}x{n}", defines, workdir)
            except Exception as exc:                       # compile failure
                print(f"  {f'{m}x{n}':>10}  build failed: {str(exc)[:60]}")

        if mods:
            scores = score_all(mods, hd)
            print(f"  {'M x N':>10}{'best ms':>12}")
            for name, t in sorted(scores.items(), key=lambda kv: kv[1]):
                shown = f"{t:.3f}" if t != float("inf") else "declined"
                print(f"  {name:>10}{shown:>12}")
            best_name = min(scores, key=lambda k: scores[k])
            if scores[best_name] != float("inf"):
                m, n = (int(x) for x in best_name.split("x"))
                winners[hd] = (scores[best_name], m, n)
                ok = sorted(t for t in scores.values() if t != float("inf"))
                margin = f", {ok[1] / ok[0]:.2f}x over next" if len(ok) > 1 else ""
                print(f"  -> best {best_name} at {scores[best_name]:.3f} ms{margin}")
        for w in workdirs:
            shutil.rmtree(w, ignore_errors=True)


    print("\nPaste into BlockCfg<HEAD_DIM, MathMode::Tf32>:")
    for hd in wanted:
        if hd in winners:
            _, m, n = winners[hd]
            print(f"  head_dim {hd:<3} M = {m:<4} N = {n}")
    return 0


if __name__ == "__main__":
    rc = main()
    # A sweep loads a few dozen CUDA extension modules into one interpreter and
    # Windows reliably faults during their teardown, long after the results are
    # printed and returned. Exiting hard skips the teardown so the exit status
    # reflects the sweep rather than the cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
