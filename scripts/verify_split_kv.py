"""
Check the tile kernel's split-KV (Flash-Decoding) path against its own
single-pass path, and both against an exact reference.

Split-KV gives each block a slice of the key range instead of all of it, so the
online softmax runs several times per output row and a second pass folds the
partials together. That is a different summation order for the same math, which
makes it exactly the kind of change that can pass a loose tolerance while being
subtly wrong -- so this compares the two paths directly, not just each against a
reference, and it asserts that the split path actually fired.

Both paths are exercised in one process, flipped with tile_set_split_kv(). That
is deliberate: run-to-run variance on this card is larger than most of the
effects being measured here, so a rebuild-and-compare would prove nothing about
the timings printed alongside.

Covers the three mask modes separately, since each takes a different branch
through the kernel: unmasked, causal (where each block splits its own
triangular range rather than the dense one), and an explicit mask. Plus two
cases that exist only to break it -- a row with no valid keys at all, and a
causal block with fewer key tiles than there are splits.

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_split_kv.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_ext  # noqa: E402
from verify_kernel import IMPLS, build_case, reference_attention, timed  # noqa: E402

# Shapes where the launcher chooses to split, taken from a sweep of
# tile_workspace_bytes rather than guessed -- split-KV only fires when the grid
# is too small to fill the device, which is a narrower set than it looks.
# Which impls split on which shape differs, because each math mode has its own
# tuned block shape and the decision keys off the resulting grid. So a case that
# does not split for one impl is not a failure -- but a *mask mode* that never
# splits for some impl is, since it would mean this file had stopped covering
# that branch. That is the assertion at the bottom.
#
# (label, batch, heads, seq_len, head_dim, causal, padded)
CASES = [
    ("dense hd64",      2, 2,  512, 64, False, False),
    ("dense hd32",      2, 2, 1024, 32, False, False),
    ("dense hd16",      2, 2,  512, 16, False, False),
    ("dense hd8",       1, 8,  512,  8, False, False),
    ("dense long",      2, 2, 2048, 64, False, False),
    ("causal hd32",     2, 2, 1024, 32, True,  False),
    ("causal hd8",      2, 2, 1024,  8, True,  False),
    ("causal hd64",     2, 2,  512, 64, True,  False),
    ("padded hd64",     2, 2,  512, 64, False, True),
    ("padded hd32",     2, 2, 1024, 32, False, True),
    ("caus+pad hd32",   2, 2, 1024, 32, True,  True),
]

# How closely the two paths must agree with each other.
#
# The strict check is the fp32 one. There the arithmetic is exact, so splitting
# changes only the summation order and the two paths land within a few ULP --
# measured 1.2e-7 to 2.4e-7 across every case here, 20x inside this bound. That
# is the number that would move if the recombination were wrong.
#
# tf32 and bf16 cannot be held there, and it would be dishonest to pretend
# otherwise: their operands are rounded to 10 and 8 significand bits before
# every multiply, so re-associating the sum re-rounds it, and the per-element
# noise floor is already ~1e-4 and ~1e-3. Their bound is their own reference
# tolerance -- the two paths must not disagree by more than the precision the
# operands carry. The fp32 row is what actually guards the math; these two guard
# against a gross error only.
PATH_AGREEMENT = {
    "tile": 5e-6,
    "tile-tf32": 3e-3,
    "tile-bf16": 3e-2,
}

TILE_IMPLS = [(code, name, tol) for code, name, tol in IMPLS
              if name in PATH_AGREEMENT]

# Which (impl, mask mode) pairs the launcher is expected to split at all, so
# this file notices when one silently stops being covered. It is not "all nine":
# min_tiles_per_split() in tile_attention.cu deliberately holds bf16 out of the
# dense and explicit paths, because the combine pass costs the same bytes in
# every math mode while bf16's main kernel is the fastest of the three, and
# dense bf16 measured 0.83x. Causal is the one place bf16 still splits.
#
# If a retune changes that, change it here too -- but only alongside a
# measurement, not to make this file go green.
EXPECTED_COVERAGE = {
    "tile": {"none", "causal", "explicit"},
    "tile-tf32": {"none", "causal", "explicit"},
    "tile-bf16": {"causal"},
}


def run(kernels, impl, q, k, v, attn_mask, is_causal, scale, split):
    kernels.tile_set_split_kv(enabled=split)
    return kernels.fused_attention_forward(
        q, k, v, attn_mask, is_causal, scale, impl
    )


def dead_row_case(device):
    """A mask with query rows that may attend to nothing at all.

    Every split of such a row reports the sentinel max, so the combine pass has
    to recognise that and emit zeros. Get it wrong and the row is 0/0 -- nan,
    which then spreads. The single-pass kernel has always had this guard; this
    checks that splitting did not lose it.
    """
    b, h, s, d = 2, 2, 512, 64
    g = torch.Generator(device=device).manual_seed(11)
    q = torch.randn(b, h, s, d, generator=g, device=device)
    k = torch.randn(b, h, s, d, generator=g, device=device)
    v = torch.randn(b, h, s, d, generator=g, device=device)
    mask = torch.ones(b, 1, s, s, dtype=torch.bool, device=device)
    mask[:, :, s // 2:, :] = False          # second half of the queries: no keys
    return q, k, v, mask.expand(b, 1, s, s).contiguous(), False, d ** -0.5


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1

    kernels = kernel_ext.get_kernels(verbose=False)
    if kernels is None:
        print(f"custom kernel failed to build: {kernel_ext.load_error()}")
        print("run through scripts/devenv.bat so cl.exe is on PATH")
        return 1
    if not hasattr(kernels, "tile_set_split_kv"):
        print("this build predates split-KV; rebuild the extension")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")

    row = "{:<16} {:>10} {:>7} {:>11} {:>11} {:>9} {:>9} {:>8}"
    print(row.format("case", "impl", "splits", "split_vs_1", "split_vs_ref",
                     "1pass_ms", "split_ms", "speedup"))
    print("-" * 92)

    failures = []
    covered: set[tuple[str, str]] = set()
    for label, b, h, s, d, causal, padded in CASES:
        # What the kernel will actually branch on, which is not the same as the
        # case flags: a padded causal case folds both into one explicit mask.
        mask_mode = "explicit" if padded else ("causal" if causal else "none")
        q, k, v, attn_mask, is_causal = build_case(
            b, h, s, d, causal, padded, device, torch.float32
        )
        scale = d ** -0.5

        with torch.inference_mode():
            ref = reference_attention(q, k, v, attn_mask, is_causal, scale)

            for impl, name, ref_tol in TILE_IMPLS:
                ws = kernels.tile_workspace_bytes(
                    B=b, H=h, S=s, head_dim=d, is_causal=is_causal, impl=impl
                )
                if ws == 0:
                    # This impl's block shape leaves the grid full enough that
                    # the launcher declines to split here. Another impl covers
                    # the same mask mode; the coverage check at the bottom is
                    # what makes sure of it.
                    print(row.format(label, name, "n/a", "-", "-",
                                     "-", "-", "-"))
                    continue
                covered.add((name, mask_mode))

                try:
                    single = run(kernels, impl, q, k, v, attn_mask,
                                 is_causal, scale, split=False)
                    split = run(kernels, impl, q, k, v, attn_mask,
                                is_causal, scale, split=True)
                except RuntimeError as exc:
                    print(row.format(label, name, "RAISED", str(exc)[:11],
                                     "-", "-", "-", "-"))
                    failures.append(f"{label}/{name}")
                    continue

                d_path = (split.float() - single.float()).abs().max().item()
                d_ref = (split.float() - ref.float()).abs().max().item()

                t_one = timed(lambda: run(kernels, impl, q, k, v, attn_mask,
                                          is_causal, scale, split=False))
                t_spl = timed(lambda: run(kernels, impl, q, k, v, attn_mask,
                                          is_causal, scale, split=True))

                bad_path = d_path > PATH_AGREEMENT[name]
                bad_ref = d_ref > ref_tol or not torch.isfinite(split).all()
                if bad_path or bad_ref:
                    failures.append(f"{label}/{name}")
                print(row.format(
                    label, name, f"{ws // 1024}K",
                    f"{d_path:.2e}" + ("!" if bad_path else ""),
                    f"{d_ref:.2e}" + ("!" if bad_ref else ""),
                    f"{t_one:.3f}", f"{t_spl:.3f}",
                    f"{t_one / t_spl:.2f}x",
                ))

    # --- the two cases that exist to break it -------------------------------
    print("-" * 92)
    q, k, v, mask, is_causal, scale = dead_row_case(device)
    with torch.inference_mode():
        for impl, name, ref_tol in TILE_IMPLS:
            split = run(kernels, impl, q, k, v, mask, is_causal, scale, True)
            dead = split[:, :, split.shape[2] // 2:, :]
            ok = torch.isfinite(split).all().item() and (dead == 0).all().item()
            print(f"dead rows / {name:<10} "
                  f"{'zeros, no nan' if ok else 'FAILED'}")
            if not ok:
                failures.append(f"dead-rows/{name}")

    # A causal block with fewer key tiles than there are splits: block 0 walks
    # one key tile, so the later splits get an empty range and must contribute
    # nothing rather than a sentinel.
    with torch.inference_mode():
        q, k, v, attn_mask, is_causal = build_case(
            2, 2, 1024, 8, True, False, device, torch.float32
        )
        ref = reference_attention(q, k, v, attn_mask, is_causal, 8 ** -0.5)
        for impl, name, ref_tol in TILE_IMPLS:
            split = run(kernels, impl, q, k, v, attn_mask, is_causal,
                        8 ** -0.5, True)
            first = (split[:, :, :16, :].float()
                     - ref[:, :, :16, :].float()).abs().max().item()
            ok = first <= ref_tol and torch.isfinite(split).all().item()
            print(f"empty splits / {name:<10} first 16 rows vs ref "
                  f"{first:.2e} {'ok' if ok else 'FAILED'}")
            if not ok:
                failures.append(f"empty-splits/{name}")

    # Every (impl, mask mode) pair the launcher is supposed to split must have
    # been exercised above. This is what catches a retune that quietly stops
    # splitting a whole branch -- the per-case "n/a" rows are allowed precisely
    # because this check is not. It is two-sided: a pair that starts splitting
    # when it was not expected to is also reported, since that is a silent
    # behaviour change in the opposite direction.
    print("-" * 92)
    for _, name, _ in TILE_IMPLS:
        for mode in ("none", "causal", "explicit"):
            want = mode in EXPECTED_COVERAGE[name]
            got = (name, mode) in covered
            if want and not got:
                failures.append(f"coverage/{name}/{mode}")
                print(f"coverage: {name} never split a {mode}-mask case")
            elif got and not want:
                failures.append(f"coverage/{name}/{mode}: unexpected")
                print(f"coverage: {name} split a {mode}-mask case, which "
                      f"min_tiles_per_split() is meant to prevent")

    kernels.tile_set_split_kv(enabled=True)
    print("-" * 92)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("split-KV matches the single-pass kernel on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
