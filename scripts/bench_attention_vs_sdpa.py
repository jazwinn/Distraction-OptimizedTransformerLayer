"""
Four attentions on the attention op alone: SDPA, scalar, wmma fp16, tile fp16.

This is `bench_attention.py`'s table with a different column set, and with the
one column that script deliberately dropped put back. The reason it was dropped
there was that a prebuilt attention is not reachable from the model any more, so
it could not be a column of a table about which kernel to dispatch to. That is
still true. This table is asking a different question -- "how do these kernels
compare to torch's own" -- and for that question SDPA is the point, not a
violation: nothing here is the model. `optimized/` and `csrc/` are untouched by
this file, and it is not imported by anything they use.

**This times the op, not the model.** That is the whole reason to run it: in the
benchmark harness the attention kernel shares a forward pass with a fused FFN, a
fused add+LayerNorm and (on small shapes) a captured CUDA graph, so a 2x
attention win shows up diluted by everything else in the layer. Here there is
nothing else in the layer.

    cmd.exe /c scripts\\devenv.bat python scripts\\bench_attention_vs_sdpa.py

Reading it:

* **ms** columns are min-of-N interleaved, in milliseconds. Lower is faster.
* **vs sdpa** is sdpa_ms / kernel_ms, so >1.00x means the kernel beats torch.
* **control** is wmma fp16 timed a second time under another name. Its true
  ratio is 1.00x, so however far off 1.00x it lands is this table's noise floor
  at that shape -- and a difference smaller than the floor is not a difference.
  It is here because interleaved min-of-N has faked double-digit wins in this
  repo before; see csrc/TUNING.md and scripts/ab_common.py.
* **err** is max abs difference from the reference, so a speed win bought by
  losing precision is visible rather than hidden.
* **n/a** means that impl declined the shape and raised, which the kernels do
  by design rather than falling back.

The reference has TF32 **on**, matching `bench_attention.py`: that is the
arithmetic the harness baseline actually runs, and the harness compares against
the baseline's output rather than against ground truth. An exact-fp32 reference
would flatter the scalar kernel.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Above torch on purpose: importing kernel_ext preloads the driver's GPU
# compiler, which stops a cuTile run from exiting 0xC0000005 only if it happens
# before torch pulls the NVIDIA DLLs in. See kernel_ext.preload_tile_compiler().
import kernel_ext  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from verify_kernel import CASES, build_case, reference_attention  # noqa: E402


ROUNDS = 10
INNER = 5

# optimized/config.py: _IMPL_CODE and _PRECISION_CODE.
IMPL_SCALAR, IMPL_WMMA, IMPL_TILE = 1, 2, 3
PREC_AUTO, PREC_FP16 = 0, 3

# The four asked for, in the order they are printed. wmma and tile are pinned to
# fp16 rather than left on "auto" so the column name is the truth: auto happens
# to choose fp16 for wmma today and fp32 for tile, so an auto tile column would
# be labelled fp16 and be running on the CUDA cores.
#
# scalar takes PREC_AUTO because it has exactly one arithmetic -- fp32 -- and
# asking it for fp16 raises rather than being ignored.
COLUMNS = (
    ("scalar",    IMPL_SCALAR, PREC_AUTO),
    ("wmma fp16", IMPL_WMMA,   PREC_FP16),
    ("tile fp16", IMPL_TILE,   PREC_FP16),
)

# Which column the control repeats. Timed twice, under two names, so the pair
# measures nothing but the timer.
CONTROL_OF = "wmma fp16"


def bench(fns):
    """Interleaved min-of-N timing, in ms.

    Candidates are timed round-robin rather than one after another, and the
    minimum is reported rather than the median: on a power-capped part the clock
    sags over a long run, which would otherwise penalise whichever candidate
    happened to be measured last. The control column is what says whether the
    remaining bias is small enough to read anything into.
    """
    for fn in fns.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    best = {name: float("inf") for name in fns}
    for _ in range(ROUNDS):
        for name, fn in fns.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(INNER):
                fn()
            end.record()
            torch.cuda.synchronize()
            best[name] = min(best[name], start.elapsed_time(end) / INNER)
    return best


def main() -> int:
    kernels = kernel_ext.get_kernels()
    if kernels is None:
        print(f"extension unavailable: {kernel_ext.load_error()}")
        return 1

    # What the harness baseline runs, so it is what the kernels have to match.
    torch.backends.cuda.matmul.allow_tf32 = True

    names = [name for name, _, _ in COLUMNS]
    head = (f"{'case':<18}{'sdpa':>10}"
            + "".join(f"{name:>11}" for name in names)
            + "".join(f"{name + '/sdpa':>17}" for name in names)
            + f"{'control':>9}"
            + f"{'sdpa err':>10}"
            + "".join(f"{name + ' err':>15}" for name in names))
    print(head)
    print("-" * len(head))

    for label, b, h, s, d, causal, padded in CASES:
        q, k, v, am, ic = build_case(b, h, s, d, causal, padded,
                                     torch.device("cuda"), torch.float32)
        scale = d ** -0.5

        def run(impl, prec):
            return kernels.fused_attention_forward(
                q, k, v, am, ic, scale, impl, 0, prec)

        # build_case never sets attn_mask and is_causal at once -- a causal
        # padded case folds the triangle into the mask and clears the flag -- so
        # this pair is already in the form SDPA accepts, and True already means
        # "may attend" on both sides. Nothing to convert, and nothing to get
        # backwards.
        assert not (am is not None and ic), f"{label}: mask and flag both set"

        def run_sdpa():
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=am, is_causal=ic, scale=scale)

        with torch.inference_mode():
            ref = reference_attention(q, k, v, am, ic, scale).float()

            timed = {"sdpa": run_sdpa}
            errs = {"sdpa": (run_sdpa().float() - ref).abs().max().item()}

            for name, impl, prec in COLUMNS:
                try:
                    errs[name] = (run(impl, prec).float() - ref).abs().max().item()
                except RuntimeError:
                    errs[name] = None
                    continue
                # Bind impl/prec per iteration; a bare closure over the loop
                # variables would time the last column under every name.
                timed[name] = (lambda i, p: lambda: run(i, p))(impl, prec)

            if CONTROL_OF in timed:
                timed["control"] = timed[CONTROL_OF]
            t = bench(timed)

        def ms(name):
            return f"{t[name]:.3f}" if errs.get(name) is not None else "n/a"

        def ratio(name):
            if errs.get(name) is None:
                return "n/a"
            return f"{t['sdpa'] / t[name]:.2f}x"

        def err(name):
            return f"{errs[name]:.1e}" if errs.get(name) is not None else "n/a"

        control = (f"{t['control'] / t[CONTROL_OF]:.3f}x"
                   if "control" in t else "n/a")

        print(f"{label:<18}{t['sdpa']:>10.3f}"
              + "".join(f"{ms(n):>11}" for n in names)
              + "".join(f"{ratio(n):>17}" for n in names)
              + f"{control:>9}"
              + f"{errs['sdpa']:>10.1e}"
              + "".join(f"{err(n):>15}" for n in names))

    print("-" * len(head))
    print("ms is min-of-%d interleaved. /sdpa above 1.00x means the kernel beats"
          % ROUNDS)
    print("torch on that shape. control is wmma fp16 against a second timing of")
    print("itself: its true value is 1.000x, so the gap to 1.000x is this row's")
    print("noise floor and a /sdpa ratio inside it is not a result.")
    print()
    print("This is the OP, not the layer. In the harness the same kernel shares a")
    print("forward with the fused FFN, the fused add+LayerNorm and a captured")
    print("graph, so these ratios are the ceiling that end-to-end speedup is a")
    print("fraction of -- not a prediction of it.")
    print()
    print("SDPA is a dispatcher and picks its own backend per shape: where no")
    print("fused one accepts the case it runs a math fallback that builds the")
    print("whole [B,H,S,S] score matrix, which is a different opponent from")
    print("flash. head_dim not a multiple of 8 is the usual trigger.")
    print()
    print("err is against a TF32-ON reference, the arithmetic the harness")
    print("baseline runs. n/a is an impl that declined the shape and raised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
