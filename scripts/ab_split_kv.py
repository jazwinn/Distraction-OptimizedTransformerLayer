"""
A/B the tile kernel's split-KV path against its single-pass path, interleaved.

Methodology, which matters more than the numbers here: run-to-run variance on
this card is +/-10-15% on the causal long-sequence cases -- larger than most of
what this is trying to measure. A cross-run comparison has already inverted a
block-shape ranking in this project once. So both variants are timed inside one
process, round-robin, several rounds, and each keeps its *best* round. Anything
under ~2% is a tie and should be read as one.

Prints a per-case ratio and the geometric mean per (impl, mask mode) group,
because summing raw milliseconds weights seq_len 2048 about ten times seq_len
128 and would let a shape that tanks the short cases still win.

The cases the launcher declines to split are kept in the table on purpose: both
sides then run *identical* code, so their ratio is a direct measurement of this
harness's own noise. It is printed at the bottom as a control, and no split
result smaller than it means anything. Cases whose kernel is too fast to resolve
at all are flagged rather than reported -- at 14 microseconds a launch, the
event timing is measuring the launch, not the kernel.

    cmd.exe /c scripts\\devenv.bat python scripts\\ab_split_kv.py
    cmd.exe /c scripts\\devenv.bat python scripts\\ab_split_kv.py --rounds 7
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_common import balanced_order  # noqa: E402

# Above torch on purpose: importing kernel_ext preloads the driver's GPU
# compiler, which stops a cuTile run from exiting 0xC0000005 only if it happens
# before torch pulls the NVIDIA DLLs in. See kernel_ext.preload_tile_compiler().
import kernel_ext  # noqa: E402

import torch  # noqa: E402

from verify_kernel import build_case  # noqa: E402

# Spans the region where the launcher currently splits: small batch*heads, and
# sequence lengths from where the grid first starves up to where it fills again.
# Includes shapes it declines to split so a regression there shows up as a tie
# rather than being invisible.
#
# (batch, heads, seq_len, head_dim, causal, padded)
CASES = [
    (2, 2,  256, 64, False, False),
    (2, 2,  512, 64, False, False),
    (2, 2, 1024, 64, False, False),
    (2, 2, 2048, 64, False, False),
    (2, 2,  512, 32, False, False),
    (2, 2, 1024, 32, False, False),
    (2, 2, 2048, 32, False, False),
    (2, 2,  512, 16, False, False),
    (2, 2, 1024, 16, False, False),
    (1, 8,  512,  8, False, False),
    (2, 2, 1024,  8, False, False),
    (2, 2,  512, 64, True,  False),
    (2, 2, 1024, 64, True,  False),
    (2, 2, 2048, 64, True,  False),
    (2, 2, 1024, 32, True,  False),
    (2, 2, 2048, 32, True,  False),
    (2, 2, 1024, 16, True,  False),
    (2, 2, 1024,  8, True,  False),
    (2, 2,  512, 64, False, True),
    (2, 2, 1024, 64, False, True),
    (2, 2, 1024, 32, False, True),
    (2, 2, 2048, 32, False, True),
]

# (precision code, label). All three are the SAME impl now -- the tile kernel
# -- differing only in the arithmetic, which is what the split-KV gate cares
# about since operand width moves the block-shape cliff.
TILE_IMPL = 3
IMPLS = ((0, "tile fp32"), (4, "tile bf16"), (2, "tile tf32"))


# Below this, a single launch is dominated by launch overhead and CUDA event
# resolution rather than by the kernel, and the ratio is noise. Calibrated from
# the control rows: with a 20-iteration median, no-split cases (identical code
# on both sides, so a true ratio of exactly 1.00) came out anywhere from 0.35x
# to 1.53x once the kernel dropped under ~0.05 ms.
RESOLUTION_MS = 0.06


def time_once(fn, budget_ms=8.0, min_iters=30, max_iters=600):
    """Median of per-launch times, over enough launches to fill budget_ms.

    Fixed iteration counts are the trap here: 20 iterations of a 0.014 ms
    kernel is 0.3 ms of work, which the timer cannot separate from its own
    overhead, while 20 iterations of a 1.4 ms kernel is plenty. Sizing the
    count from a pilot measurement puts every case on the same footing.
    """
    torch.cuda.synchronize()
    pilot_start = torch.cuda.Event(enable_timing=True)
    pilot_end = torch.cuda.Event(enable_timing=True)
    pilot_start.record()
    for _ in range(5):
        fn()
    pilot_end.record()
    torch.cuda.synchronize()
    per = max(pilot_start.elapsed_time(pilot_end) / 5, 1e-4)
    iters = int(min(max(budget_ms / per, min_iters), max_iters))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--impl", default=None,
                    help="restrict to one impl name, e.g. 'tile tf32'")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    kernels = kernel_ext.get_kernels(verbose=False)
    if kernels is None:
        print(f"custom kernel failed to build: {kernel_ext.load_error()}")
        return 1
    if not hasattr(kernels, "tile_set_split_kv"):
        print("this build predates split-KV; rebuild the extension")
        return 1

    device = torch.device("cuda")
    impls = [x for x in IMPLS if args.impl is None or x[1] == args.impl]

    # Build every case and closure up front, then time them all round-robin.
    # Building inside the timing loop would let allocator state drift between
    # the two variants, which is the sort of thing that produces a stable-
    # looking 5% that is not there.
    work = []
    for b, h, s, d, causal, padded in CASES:
        q, k, v, attn_mask, is_causal = build_case(
            b, h, s, d, causal, padded, device, torch.float32
        )
        scale = d ** -0.5
        mode = "explicit" if padded else ("causal" if causal else "dense")
        for prec, name in impls:
            ws = kernels.tile_workspace_bytes(
                B=b, H=h, S=s, head_dim=d, is_causal=is_causal, precision=prec
            )
            splits = 0 if ws == 0 else ws // (b * h * s * (d + 2) * 4)

            def make(split, prec=prec, q=q, k=k, v=v, m=attn_mask,
                     c=is_causal, sc=scale):
                def go():
                    kernels.tile_set_split_kv(enabled=split)
                    kernels.fused_attention_forward(
                        q, k, v, m, c, sc, TILE_IMPL, 0, prec)
                return go

            work.append({
                "label": f"b{b} h{h} s{s} d{d} {mode}",
                "impl": name, "mode": mode, "splits": splits,
                "off": make(False), "on": make(True),
                "best_off": math.inf, "best_on": math.inf,
            })

    for w in work:              # warm up both paths before any timing
        for _ in range(5):
            w["off"]()
            w["on"]()

    for r in range(args.rounds):
        for w in work:
            # off then on every round would hand "on" the later slot each time,
            # and the later slot is faster. See ab_common.balanced_order.
            for side in balanced_order(("off", "on"), r):
                key = "best_" + side
                w[key] = min(w[key], time_once(w[side]))
        print(f"  round {r + 1}/{args.rounds} done", file=sys.stderr)

    row = "{:<26} {:>10} {:>7} {:>10} {:>10} {:>8}"
    print(row.format("case", "impl", "splits", "1pass_ms", "split_ms", "ratio"))
    print("-" * 84)
    groups: dict[tuple[str, str], list[float]] = {}
    control: list[float] = []
    unresolved = 0
    for w in work:
        ratio = w["best_off"] / w["best_on"]
        slow = min(w["best_off"], w["best_on"])
        if slow < RESOLUTION_MS:
            flag = "  too fast to resolve"
            unresolved += 1
        elif w["splits"] >= 2:
            flag = ""
            groups.setdefault((w["impl"], w["mode"]), []).append(ratio)
        else:
            flag = "  (control: same code)"
            control.append(ratio)
        print(row.format(w["label"], w["impl"],
                         str(w["splits"]) if w["splits"] else "-",
                         f"{w['best_off']:.3f}", f"{w['best_on']:.3f}",
                         f"{ratio:.2f}x") + flag)

    print("-" * 84)
    if control:
        spread = max(abs(r - 1.0) for r in control)
        print(f"control: {len(control)} cases running identical code both "
              f"sides, max deviation {spread * 100:.1f}%")
        print(f"         nothing below {1 + spread:.2f}x / above "
              f"{1 - spread:.2f}x is a real effect")
    if unresolved:
        print(f"{unresolved} cases under {RESOLUTION_MS} ms excluded: at that "
              f"size the timer measures the launch, not the kernel")
    print()
    print("geometric mean over cases that actually split "
          "(>1 = split-KV faster):")
    for (impl, mode), ratios in sorted(groups.items()):
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        worst = min(ratios)
        print(f"  {impl:<10} {mode:<9} n={len(ratios):<3} "
              f"gmean {gm:.3f}x   worst {worst:.2f}x")

    kernels.tile_set_split_kv(enabled=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
