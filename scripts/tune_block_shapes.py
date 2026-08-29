"""Sweep the compiled-in block shapes of every attention backend and print the winners.

Shapes are template parameters, so each candidate needs its own build. Every
backend exposes `-D` overrides for exactly that reason -- TF32_M_*/TF32_N_* in
tile_attention.cu, FP32_*/BF16_* beside them, WMMA_M_*/WMMA_N_* in
attention_wmma.cuh -- so this can search without editing the sources. The
numbers it prints go back into those macro defaults.

    python scripts/tune_block_shapes.py                      # every backend, dense
    python scripts/tune_block_shapes.py --backend wmma       # just one
    python scripts/tune_block_shapes.py --backend wmma 64    # one backend, one head_dim
    python scripts/tune_block_shapes.py --causal             # the causal shapes instead
    python scripts/tune_block_shapes.py --backend wmma --dtype float16
    python scripts/tune_block_shapes.py --backend wmma --frag tf32   # pre-fp16

Four backends, and they are not interchangeable:

  tile-fp32   cuTile, CUDA cores.       Spill cliff dominates; wants narrow N.
  tile-bf16   cuTile, tensor cores.     Half the operand width moves the cliff.
  tile-tf32   cuTile, tensor cores.     fp32's 32 bits, bf16's MMA units -- the
                                        one mode where the two pressures pull
                                        apart, so it inherits neither's shapes.
  wmma        nvcuda::wmma fragments.   The shipping default for every dtype.

wmma has a third axis the other three do not: the tensors it is handed and the
fragments it contracts them in are different types. `maybe_launch_wmma` narrows
an fp32 tensor to __half whenever `wmma_fp16_flag()` is set -- which is the
kernel's DEFAULT -- and instantiates `WmmaCfg` on __half. So `--dtype float32`
alone does not say what the shared-memory budget is; `--frag` does, and it is
what the candidate filter models. Getting this wrong does not produce a wrong
timing, it produces a wrong SEARCH: a run that filtered on the 4-byte budget
while the kernel ran 2-byte fragments never built (32,64) or (32,80) at head_dim
64, both of which fit. `--frag tf32` gets the old behaviour back.

Two structural differences the flags have to respect:

* The tile kernels take a compile-time MaskMode, so dense and causal can carry
  different shapes and `--causal` sweeps them separately. The wmma kernel takes
  `is_causal` as a *runtime* argument, so one shape must serve both; wmma is
  therefore always scored on dense and causal cases together and rejects
  `--causal`. Making it mask-tunable means templating the kernel on the mask,
  which is a kernel change, not a macro.

* cuTile requires every tile extent to be a power of two (`is_pow2` in
  crt/cuda_tile.h:749). wmma does not: BLOCK_M only has to be a multiple of 16
  (it *is* the warp count times 16) and BLOCK_N a multiple of
  FragTraits<scalar_t>::K. So the wmma candidate set includes 48/80/96/112, and
  FlashAttention-2's kBlockN=112 -- inexpressible in the tile kernel -- is
  reachable for the 16-bit types.

Read csrc/TUNING.md's two rules before trusting any number here: never compare
timings across runs, and score short and long sequences together.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Above torch on purpose: importing kernel_ext preloads the driver's GPU
# compiler, which stops a cuTile run from exiting 0xC0000005 only if it happens
# before torch pulls the NVIDIA DLLs in. See kernel_ext.preload_tile_compiler().
import kernel_ext  # noqa: E402

import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

# cuTile: every extent a power of two. FA2 returns kBlockM=128 unconditionally
# for every arch and head_dim, which is why 128 is in the M set; its kBlockN for
# the headdim<=64 bucket all four of our head_dims fall into is 112, which is
# not expressible here at all -- 64 and 128 are the legal neighbours.
_TILE_CANDIDATES = tuple(itertools.product((16, 32, 64, 128), (16, 32, 64, 128)))

# wmma: BLOCK_M % 16 == 0 (and it is the warp count times 16, so 256 is 8 warps
# and the practical ceiling), BLOCK_N % WK == 0. Illegal shapes are pruned by
# _wmma_legal below rather than discovered by a failed build.
_WMMA_M = (16, 32, 64, 128)
_WMMA_N = (16, 32, 48, 64, 80, 96, 112, 128)

# Bytes per element and fragment K/pad, keyed by the type the fragments
# CONTRACT IN -- not by the dtype of the tensors handed to the kernel. Mirrors
# FragTraits in csrc/attention_wmma.cuh.
_FRAG = {
    torch.float32:  (4, 8, 4),    # tf32 fragments: 16x16x8, ldm % 4 == 0
    torch.float16:  (2, 16, 8),   # ldm % 8 == 0
    torch.bfloat16: (2, 16, 8),
}


def frag_dtype(dtype, fp16_frags: bool):
    """The type the fragments contract in, which is what sizes every staged tile.

    Only an fp32 tensor has a choice to make: maybe_launch_wmma narrows it to
    __half when wmma_fp16_flag() is set, and WmmaCfg is then instantiated on
    __half -- so an fp32 run under the fp16 flag has a 2-byte budget, not a
    4-byte one, and shapes the 4-byte model rejects are legal. half and
    bfloat16 tensors contract in the type they already are. Mirrors
    maybe_launch_wmma in csrc/attention_wmma.cuh.
    """
    return torch.float16 if (dtype is torch.float32 and fp16_frags) else dtype


def _wmma_smem(frag, head_dim: int, m: int, n: int):
    """(total, scratch) shared bytes WmmaCfg<compute_t, HEAD_DIM> asks for here.

    `frag` is compute_t, the fragment type -- see frag_dtype. The kernel asks
    SUPPORTED of compute_t, so that is the type this model has to mirror.

    `scratch` is the QO/K/V/S span the accumulator probe borrows before the
    first key tile is staged; SUPPORTED requires it to cover PROBE_BYTES.

    Kept in step with csrc/attention_wmma.cuh by hand. If the two drift, the
    only cost is a wasted build or a shape skipped that would have compiled --
    the kernel's own SUPPORTED check is still what decides, and a shape that
    slips through anyway declines at run time and scores as `inf`.
    """
    esz, _, pad = _FRAG[frag]
    pdim = 16 if head_dim < 16 else head_dim
    kv_ld, o_ld, s_ld = pdim + pad, pdim + 4, n + pad
    qo = max(esz * m * kv_ld, 4 * m * o_ld)
    kv = esz * n * kv_ld
    s_bytes = 4 * m * s_ld
    p_bytes = 0 if frag is torch.float32 else esz * m * s_ld    # p aliases s for tf32
    scratch = qo + 2 * kv + s_bytes
    return scratch + p_bytes + 3 * (4 * m), scratch


def _wmma_legal(frag, head_dim: int, m: int, n: int) -> bool:
    _, wk, _ = _FRAG[frag]
    pdim = 16 if head_dim < 16 else head_dim
    if pdim % wk or pdim % 16 or n % wk or m % 16:
        return False
    total, scratch = _wmma_smem(frag, head_dim, m, n)
    if scratch < 4 * (m // 16) * 512:        # the accumulator probe needs this much
        return False
    # 48 KB is the cap that keeps two blocks resident without opting in to the
    # larger dynamic shared-memory carveout.
    return total <= 48 * 1024


def _wmma_candidates(frag, head_dim: int):
    return tuple((m, n) for m in _WMMA_M for n in _WMMA_N
                 if _wmma_legal(frag, head_dim, m, n))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend:
    """One sweepable kernel: its impl + precision codes, macro prefix, and
    shape space.

    `mask_split` is False for wmma, whose kernel takes is_causal at run time and
    so cannot carry a per-mask shape. That is not a gap in the macros; it is a
    property of the kernel, and scoring wmma on dense cases alone would tune it
    for half of what it actually runs.
    """

    def __init__(self, name, impl, prec, prefix, head_dims, mask_split, dtypes):
        self.name = name
        self.impl = impl
        self.prec = prec
        self.prefix = prefix
        self.head_dims = head_dims
        self.mask_split = mask_split
        self.dtypes = dtypes

    def candidates(self, frag, head_dim):
        if self.name == "wmma":
            return _wmma_candidates(frag, head_dim)
        return _TILE_CANDIDATES

    def macros(self, dtype, head_dim: int, m: int, n: int, causal: bool) -> dict:
        """-D overrides pinning `head_dim` to (m, n) and every other head_dim to
        a shape known to compile, so one bad candidate cannot fail the build for
        reasons unrelated to the shape under test.

        The safe shape is head_dim-dependent for the same reason the defaults
        are: four of the five live tiles scale with head_dim, so the tile
        backends' 64x64 is 27 KB at head_dim 8 and 205 KB at head_dim 128. A
        head_dim not under test still gets compiled into every candidate build,
        so pinning it to a shape that spills would tax -- or fail -- every build
        in the sweep for a kernel this run never times."""
        mp = f"{self.prefix}_{'CM' if causal else 'M'}"
        np_ = f"{self.prefix}_{'CN' if causal else 'N'}"
        out = {}
        for d in self.head_dims:
            if self.name == "wmma":
                safe_m, safe_n = 32, 16
            else:
                safe_m, safe_n = (16, 16) if d >= 128 else (64, 64)
            out[f"{mp}_{d}"] = m if d == head_dim else safe_m
            out[f"{np_}_{d}"] = n if d == head_dim else safe_n
        return out


BACKENDS = {
    # impl 3 is the tile kernel for all three; what differs is the precision.
    "tile-fp32": Backend("tile-fp32", 3, 1, "FP32", (8, 16, 32, 64, 128), True,
                         (torch.float32,)),
    "tile-bf16": Backend("tile-bf16", 3, 4, "BF16", (8, 16, 32, 64, 128), True,
                         (torch.float32,)),
    "tile-tf32": Backend("tile-tf32", 3, 2, "TF32", (8, 16, 32, 64, 128), True,
                         (torch.float32,)),
    "wmma":      Backend("wmma",      2, 0, "WMMA", (8, 16, 32, 64, 128), False,
                         (torch.float32, torch.float16, torch.bfloat16)),
}

# Shapes the kernels are actually used at. Timed together so a shape is chosen
# for aggregate behaviour rather than for one lucky sequence length.
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


def build(name: str, defines: dict, workdir: str):
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


def make_cases(head_dim: int, dtype, cases):
    """Materialise inputs once, so timing does not include allocation."""
    dev = torch.device("cuda")
    out = []
    for b, h, sl, causal in cases:
        q, k, v = (torch.randn(b, h, sl, head_dim, device=dev, dtype=dtype)
                   for _ in range(3))
        out.append(((q, k, v), causal, head_dim ** -0.5))
    return out


def time_round(mod, cases, impl: int, prec: int) -> float:
    """One pass over every case; total ms, or inf if this build declines them."""
    total = 0.0
    try:
        with torch.inference_mode():
            for (q, k, v), causal, scale in cases:
                total += time_ms(lambda: mod.fused_attention_forward(
                    q, k, v, None, causal, scale, impl, 0, prec), iters=15)
    except RuntimeError:
        return float("inf")
    return total


def score_all(mods: dict, head_dim: int, dtype, cases, impl: int, prec: int,
              rounds: int = 5) -> dict:
    """Time every candidate round-robin; return each one's best round.

    Timing candidate A to completion and then candidate B measures whatever
    else changed between them as much as the shapes themselves: run-to-run
    variance here was large enough to invert rankings. Cycling all candidates
    within each round spreads that variance across them evenly, and taking
    each candidate's best round approximates its uncontended cost.

    Corollary: numbers from two separate runs of this script are NOT
    comparable. Any candidate you want to rank must be in the same run.
    """
    materialised = make_cases(head_dim, dtype, cases)
    best = {name: float("inf") for name in mods}
    for _ in range(rounds):
        for name, mod in mods.items():
            best[name] = min(best[name], time_round(mod, materialised, impl, prec))
    return best


def sweep_backend(be: Backend, dtype, frag, head_dims, causal: bool,
                  rounds: int) -> dict:
    """Sweep one backend over the requested head_dims; return {head_dim: (ms, m, n)}."""
    # A mask-split backend is scored only on the cases its shape will serve.
    # Scoring a shape on cases it never runs is how one shape ended up serving
    # both mask modes. wmma's shape serves both, so it is scored on both.
    cases = tuple(c for c in ALL_CASES if c[3] == causal) if be.mask_split else ALL_CASES

    label = f"{be.name}  {str(dtype).replace('torch.', '')}"
    if be.name == "wmma" and frag is not dtype:
        label += f"->{str(frag).replace('torch.', '')} frags"
    label += "  "
    label += ("causal" if causal else "dense") if be.mask_split else "dense+causal"
    print(f"\n=== {label} ===")

    winners = {}
    for hd in head_dims:
        cands = be.candidates(frag, hd)
        if not cands:
            print(f"\nhead_dim {hd}: no legal shape")
            continue
        print(f"\nhead_dim {hd}  ({len(cands)} candidates)")

        # Build every candidate before timing any of them, so the measurement
        # phase is not interrupted by minute-long nvcc runs.
        mods, workdirs = {}, []
        for m, n in cands:
            defines = be.macros(dtype, hd, m, n, causal)
            workdir = tempfile.mkdtemp(prefix=f"{be.prefix}_{hd}_{m}x{n}_")
            workdirs.append(workdir)
            try:
                mods[f"{m}x{n}"] = build(
                    f"tune_{be.prefix.lower()}_{hd}_{m}x{n}", defines, workdir)
            except Exception as exc:                       # compile failure
                print(f"  {f'{m}x{n}':>10}  build failed: {str(exc)[:60]}")

        if mods:
            scores = score_all(mods, hd, dtype, cases, be.impl, be.prec, rounds)
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
    return winners


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Sweep attention block shapes.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("head_dims", nargs="*", type=int,
                   help="head_dims to sweep (default: all the backend supports)")
    p.add_argument("--backend", action="append", choices=sorted(BACKENDS),
                   help="repeatable; default is every backend")
    p.add_argument("--causal", action="store_true",
                   help="sweep the causal shapes (tile backends only)")
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float16", "bfloat16"),
                   help="input dtype; only wmma accepts anything but float32")
    p.add_argument("--frag", default="auto", choices=("auto", "fp16", "tf32"),
                   help="wmma only: what fp32 tensors contract in. auto (=fp16) "
                        "matches the kernel's own default; tf32 is the pre-fp16 "
                        "behaviour. Sets WMMA_FP16 and, more importantly, picks "
                        "which shared-memory budget the candidate filter models")
    p.add_argument("--rounds", type=int, default=5,
                   help="interleaved timing rounds per candidate")
    return p.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
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

    dtype = getattr(torch, args.dtype)
    names = args.backend or sorted(BACKENDS)

    # Set before the first launch in any built module: wmma_fp16_flag() is a
    # function-local static seeded from the environment on first call, per
    # loaded extension. Setting it here covers every candidate build.
    fp16_frags = args.frag != "tf32"
    os.environ["WMMA_FP16"] = "1" if fp16_frags else "0"
    frag = frag_dtype(dtype, fp16_frags)

    note = ""
    if frag is not dtype:
        note = (f"  frags={str(frag).replace('torch.', '')} (WMMA_FP16=1; the "
                f"candidate filter models its 2-byte budget)")
    elif dtype is torch.float32:
        note = "  frags=tf32 (WMMA_FP16=0)"
    print(f"{torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}  "
          f"dtype={args.dtype}{note}")

    results = {}
    for name in names:
        be = BACKENDS[name]
        if dtype not in be.dtypes:
            takes = ", ".join(str(d).replace("torch.", "") for d in be.dtypes)
            print(f"\n=== {name} === skipped: takes {takes}, not {args.dtype}")
            continue
        if args.causal and not be.mask_split:
            print(f"\n=== {name} === skipped: its kernel takes is_causal at run "
                  f"time, so one shape serves both mask modes. Run without "
                  f"--causal; it is scored on dense and causal cases together.")
            continue
        head_dims = [h for h in (args.head_dims or be.head_dims) if h in be.head_dims]
        if not head_dims:
            have = ", ".join(map(str, be.head_dims))
            print(f"\n=== {name} === skipped: supports head_dim {{{have}}}")
            continue
        # The tile backends pick their math mode by precision, not by WMMA_FP16, so
        # their budget is the tensor dtype's either way.
        be_frag = frag if be.name == "wmma" else dtype
        results[name] = sweep_backend(be, dtype, be_frag, head_dims, args.causal,
                                      args.rounds)

    print("\n" + "=" * 60)
    print("Winners. Paste into the macro defaults these override.")
    for name, winners in results.items():
        be = BACKENDS[name]
        mp = f"{be.prefix}_{'CM' if args.causal and be.mask_split else 'M'}"
        np_ = f"{be.prefix}_{'CN' if args.causal and be.mask_split else 'N'}"
        print(f"\n{name}:")
        for hd in sorted(winners):
            _, m, n = winners[hd]
            print(f"  #define {mp}_{hd:<4} {m:<4}   #define {np_}_{hd:<4} {n}")
    return 0


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    # A sweep loads a few dozen CUDA extension modules into one interpreter and
    # Windows reliably faults during their teardown, long after the results are
    # printed and returned. Exiting hard skips the teardown so the exit status
    # reflects the sweep rather than the cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
