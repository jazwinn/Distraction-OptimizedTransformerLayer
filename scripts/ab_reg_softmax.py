"""A/B the register-resident softmax (WMMA_REG_SOFTMAX) against `s_s`.

`WMMA_REG_SOFTMAX` is a **compile-time** switch, because what it buys is an
*allocation*: deleting `s_s` frees `4 * BLOCK_M * S_LD` bytes -- 10.00 KB at
head_dim 32 -- which is what holds that head_dim to 2 resident blocks per SM.
Row 12 of docs/OPTIMIZATION_LEDGER.md established that a runtime flag cannot measure
an allocation, so both arms have to be separate builds timed round-robin inside
one process. That is what this does.

Two things are checked, and they need different gates:

  * **accuracy** -- this is NOT expected to be bit-identical, unlike the
    narrowings in rows 5-10. The row max and the `l` sum now come from a 2-step
    quad butterfly instead of a single 16-offset shuffle over two lanes, so the
    reduction ORDER changes and the last bits legitimately move. The gate is the
    harness's own budget (rtol 2e-2 / atol 2e-3) against an fp64 reference, with
    both arms reported side by side so a regression is visible as a difference
    between them rather than as an absolute number.

  * **speed** -- per case, on the grading shapes that use each head_dim, with
    the reference arm re-timed alongside the challenger every round.
"""
import argparse
import os
import shutil
import statistics
import sys
import tempfile

import torch

sys.path.insert(0, ".")
import kernel_ext  # noqa: E402  (must precede torch)
sys.path.insert(0, "scripts")
from ab_common import balanced_order  # noqa: E402
from tune_block_shapes import GRADING_CASES, build  # noqa: E402

IMPL_WMMA = 2
PREC_FP16 = 3


def reference(q, k, v, causal, scale):
    """fp64 attention, so neither arm is the yardstick for the other."""
    qd, kd, vd = q.double(), k.double(), v.double()
    sc = (qd @ kd.transpose(-1, -2)) * scale
    if causal:
        sl = q.shape[-2]
        tri = torch.triu(torch.ones(sl, sl, device=q.device, dtype=torch.bool), 1)
        sc = sc.masked_fill(tri, float("-inf"))
    return torch.softmax(sc, dim=-1) @ vd


def time_ms(fn, iters=15, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def accuracy(mods, head_dim, dtype):
    """Both arms against fp64, on shapes that exercise the masked paths too."""
    print(f"\n  accuracy, head_dim {head_dim} "
          f"(gate: abs<=2e-3 OR rel<=2e-2)")
    print(f"    {'case':<26}" + "".join(f"{n:>16}" for n in mods))
    cases = [
        (2, 4, 128, True, "causal s128"),
        (2, 4, 128, False, "dense s128"),
        (1, 4, 32, True, "causal s32 (ragged)"),
        (2, 4, 1024, True, "causal s1024"),
        (3, 2, 130, True, "causal s130 (odd)"),
    ]
    ok = True
    for b, h, sl, causal, label in cases:
        g = torch.Generator(device="cuda").manual_seed(7)
        q, k, v = (torch.randn(b, h, sl, head_dim, generator=g,
                               device="cuda", dtype=dtype) for _ in range(3))
        scale = head_dim ** -0.5
        ref = reference(q, k, v, causal, scale)
        cells = ""
        for name, mod in mods.items():
            with torch.inference_mode():
                got = mod.fused_attention_forward(q, k, v, None, causal, scale,
                                                  IMPL_WMMA, 0, PREC_FP16)
            err = (got.double() - ref).abs()
            rel = err / ref.abs().clamp_min(1e-30)
            bad = ((err > 2e-3) & (rel > 2e-2)).sum().item()
            cells += f"{err.max().item():>10.2e}{'' if bad == 0 else '!'}"
            cells += f"{bad:>5d}" if bad else "     "
            ok &= bad == 0
        print(f"    {label:<26}{cells}")
    print(f"    {'max_abs then failing-element count per arm':<26}")
    return ok


def speed(mods, head_dim, dtype, rounds, self_control, target_ms):
    cases = GRADING_CASES.get(head_dim)
    if not cases:
        print(f"  [skip] head_dim {head_dim}: no grading shape uses it")
        return
    dev = torch.device("cuda")
    mat = []
    for b, h, sl, causal in cases:
        g = torch.Generator(device=dev).manual_seed(11)
        q, k, v = (torch.randn(b, h, sl, head_dim, generator=g, device=dev,
                               dtype=dtype) for _ in range(3))
        mat.append(((q, k, v), causal, head_dim ** -0.5))

    names = list(mods)
    ref_name = names[0]
    arms = {n: (mods[ref_name] if self_control else mods[n]) for n in names}

    # Iterations scaled to a fixed wall clock per case, not a fixed count.
    # A grading case at head_dim 32 spans 0.014 ms (batch 1) to 0.72 ms
    # (seq 1024) -- 50x. A single `iters` therefore gives the small cases a
    # sub-millisecond sample window, where one context switch is a 10%
    # reading: the first self-control run of this script read 0.895x-1.052x
    # per case on an arm that was byte-identical to itself. This is the fix
    # the ledger's standing measurement facts prescribe.
    iters = []
    with torch.inference_mode():
        for (q, k, v), causal, scale in mat:
            probe = time_ms(lambda: arms[ref_name].fused_attention_forward(
                q, k, v, None, causal, scale, IMPL_WMMA, 0, PREC_FP16), iters=3)
            iters.append(max(5, min(2000, int(round(target_ms / max(probe, 1e-4))))))

    per = {n: [[] for _ in mat] for n in names}
    for rnd in range(rounds):
        for n in balanced_order(names, rnd):
            mod = arms[n]
            with torch.inference_mode():
                for ci, ((q, k, v), causal, scale) in enumerate(mat):
                    per[n][ci].append(time_ms(
                        lambda: mod.fused_attention_forward(
                            q, k, v, None, causal, scale, IMPL_WMMA, 0, PREC_FP16),
                        iters=iters[ci]))

    med = {n: [statistics.median(c) for c in per[n]] for n in names}
    tag = "  SELF-CONTROL" if self_control else ""
    print(f"\n  speed, head_dim {head_dim}{tag}  "
          f"(cases: {', '.join('b%dh%ds%d%s' % c for c in cases)})")
    print(f"    {'iters':<12}" + "".join(f"{i:>9d}" for i in iters))
    for n in names:
        print(f"    {n:<12}" + "".join(f"{t:>9.4f}" for t in med[n]))
    ratios = [a / b for a, b in zip(med[ref_name], med[names[-1]])]
    print(f"    {'ratio':<12}" + "".join(f"{r:>9.4f}" for r in ratios))
    import math
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"    geomean {gm:.4f}x   (>1 means {names[-1]} is faster)"
          + ("   <-- should read 1.000x" if self_control else ""))
    # The spread, not the geomean, is what says whether an effect is readable:
    # a geomean of 1.000x can sit on top of a +/-10% per-case scatter.
    print(f"    spread  {min(ratios):.4f}x - {max(ratios):.4f}x"
          f"   (+/-{100.0 * max(abs(1.0 - min(ratios)), abs(max(ratios) - 1.0)):.1f}%)")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head_dims", nargs="*", type=int, default=[32])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=120.0,
                   help="wall clock per timed sample; sets iters per case")
    ap.add_argument("--self-control", action="store_true",
                   help="put the reference build in both slots; must read 1.000x")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    kernel_ext._ensure_msvc_on_path()
    tile_home = kernel_ext._find_tile_cuda_home()
    if tile_home is None:
        print("no CUDA toolkit with <cuda_tile.h>")
        return 1
    os.environ["CUDA_HOME"] = tile_home
    os.environ["PATH"] = os.path.join(tile_home, "bin") + os.pathsep + os.environ["PATH"]
    os.environ["WMMA_FP16"] = "1"

    print(f"{torch.cuda.get_device_name(0)}  building both arms "
          f"(this is two nvcc runs)")
    mods, workdirs = {}, []
    for label, val in (("s_s", 0), ("registers", 1)):
        wd = tempfile.mkdtemp(prefix=f"regsm_{val}_")
        workdirs.append(wd)
        try:
            mods[label] = build(f"ab_regsm_{val}", {"WMMA_REG_SOFTMAX": val}, wd)
        except Exception as exc:
            print(f"  {label}: BUILD FAILED\n{str(exc)[-3000:]}")
            for w in workdirs:
                shutil.rmtree(w, ignore_errors=True)
            return 1
        print(f"  {label:<10} built")

    rc = 0
    try:
        for hd in args.head_dims:
            if not args.skip_accuracy:
                if not accuracy(mods, hd, torch.float16):
                    print(f"  ACCURACY FAIL at head_dim {hd}")
                    rc = 1
            speed(mods, hd, torch.float16, args.rounds, args.self_control,
                  args.target_ms)
    finally:
        for w in workdirs:
            shutil.rmtree(w, ignore_errors=True)
    return rc


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
