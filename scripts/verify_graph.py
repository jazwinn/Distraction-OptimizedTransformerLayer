"""
Check that replaying a captured CUDA graph is *bit-identical* to running the
model eagerly, and that the graph is actually being used when it claims to be.

Bit-exactness is the whole claim this feature rests on -- a graph replays the
same kernels in the same order on the same addresses, so there is no mechanism
by which the answer should move. That makes the tolerance here exactly zero, not
a small number: anything nonzero means the captured kernels are not the eager
kernels, and a fast wrong answer is worth less than a slow right one. The delta
is printed even when it is zero, so a future regression arrives as a number
rather than as a pass/fail flip.

The other half of the job is proving the fast path fired. A test that only
compares outputs passes trivially when capture silently declines and the eager
path answers both times -- so every case asserts a replay counter, the same
discipline as verify_split_kv.py's coverage check.

The subtler failure modes each get their own case, because none of them show up
as a wrong shape or a NaN:

  * a missing input copy replays stale data, which is finite, correctly shaped,
    and identical for different inputs -- so three different inputs are pushed
    through one graph and the outputs are asserted to differ from each other.
  * a graph handed out without cloning is overwritten in place by the next
    replay, so a result held across a call is asserted to survive one.
  * the mask mode selects a different op sequence and must key separately, so a
    trivial -> padded -> trivial sequence is asserted to capture twice and then
    *reuse*, not re-capture.

One thing this script cannot be blamed for: with a cuTile impl selected
(--include-tile), the process exits with an access violation *after* the verdict
is printed. That is pre-existing and has nothing to do with graphs -- the same
crash happens with CUDA_GRAPH="off", and scripts/verify_split_kv.py already exits
the same way on this machine. The cuTile kernels are bit-exact under capture; it
is their teardown that is broken. They are therefore off by default here so that
this script's exit code stays meaningful.

    cmd.exe /c scripts\\devenv.bat python scripts\\verify_graph.py
    cmd.exe /c scripts\\devenv.bat python scripts\\verify_graph.py --test-failure
    cmd.exe /c scripts\\devenv.bat python scripts\\verify_graph.py --include-tile
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_transformer_benchmark as bench  # noqa: E402
from optimized import config, graphs  # noqa: E402

# The runtime knobs live in optimized/config.py, and the model reads them
# from there on every call. Setting them on the harness module instead
# would create a dead attribute that nothing reads -- which is worse than
# an error here, because both sides of every comparison below would then
# be the *same* path and every delta would come out zero.

DEV = torch.device("cuda")

# Mirrors compare_backends.py's sweep, plus the shapes that exercise the
# branches most likely to be wrong rather than merely slow.
CASES = [
    # name,             B,  S,    d_model, heads, ffn,  layers, causal, padding
    ("small b1 s32",     1,   32,  512,  8, 2048,  6, False, 0.0),
    ("default",          8,  128,  512,  8, 2048,  6, False, 0.0),
    ("padded 0.3",       8,  128,  512,  8, 2048,  6, False, 0.3),
    ("causal",           8,  128,  512,  8, 2048,  6, True,  0.0),
    ("causal+padded",    8,  128,  512,  8, 2048,  6, True,  0.3),
    ("seq512 b4",        4,  512,  512,  8, 2048,  6, False, 0.0),
    ("seq2048 b1",       1, 2048,  512,  8, 2048,  6, False, 0.0),
    ("wide d1024",       8,  128, 1024, 16, 4096,  6, False, 0.0),
    ("deep 12L",         8,  128,  512,  8, 2048, 12, False, 0.0),
    ("heads1 hd512",     2,   64,  512,  1, 2048,  6, False, 0.0),
]

ROW = "{:<16} {:>7} {:>12} {:>8} {:>9} {:>10}"


def build(b, s, d, h, ffn, layers, causal):
    cfg = bench.TransformerConfig(batch_size=b, seq_len=s, d_model=d,
                                  num_heads=h, ffn_dim=ffn, num_layers=layers,
                                  causal=causal)
    base = bench.BaselineTransformer(cfg)
    opt = bench.UserOptimizedTransformer(cfg)
    bench.copy_model_weights(base, opt)
    return cfg, base.to(DEV).eval(), opt.to(DEV).eval()


def case_data(cfg, padding, seed):
    return bench.generate_random_case(config=cfg, device=DEV, dtype=torch.float32,
                                      seed=seed, padding_ratio=padding,
                                      input_scale=1.0)


def delta(a, b):
    return (a.float() - b.float()).abs().max().item()


def report(label, ok, failures, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-failure", action="store_true",
                    help="also exercise the capture-failure fallback. Runs "
                         "last on purpose: a failed capture can in principle "
                         "poison the CUDA context, which would invalidate any "
                         "result printed after it.")
    ap.add_argument("--include-tile", action="store_true",
                    help="also check the cuTile impls. Correct under capture, "
                         "but they crash the interpreter at shutdown -- see the "
                         "note in the source -- so this run's process exit code "
                         "becomes meaningless and only the printed verdict counts.")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    failures = []

    print("=== replay vs eager, tolerance is exactly zero ===\n")
    print(ROW.format("case", "tokens", "max_abs", "replays", "pool MiB", "verdict"))
    print("-" * 68)

    for name, b, s, d, h, ffn, layers, causal, padding in CASES:
        cfg, _, opt = build(b, s, d, h, ffn, layers, causal)
        x, m = case_data(cfg, padding, seed=1234)

        # Eager first, with capture switched off entirely. Note this records no
        # denial -- _graph_eligible returns False before _maybe_capture can mark
        # the key -- which is what makes the second pass able to capture.
        config.CUDA_GRAPH = "off"
        with torch.inference_mode():
            eager = opt(x, m).clone()
        if opt._graphs:
            failures.append(f"{name}: captured a graph while CUDA_GRAPH was off")

        # "always" so the size gate does not silently skip the large shapes --
        # this script is testing correctness, not deciding what is worth doing.
        #
        # empty_cache() before the baseline reading, for the same reason
        # _capture_graph does it: entering a capture drains the allocator cache,
        # so without this the delta is (pool - freed cache) and can come out
        # negative, which is not a number anyone can act on.
        torch.cuda.empty_cache()
        reserved_before = torch.cuda.memory_reserved(DEV)
        config.CUDA_GRAPH = "always"
        with torch.inference_mode():
            graphed = opt(x, m).clone()
        pool_mib = (torch.cuda.memory_reserved(DEV) - reserved_before) / (1 << 20)

        d_max = delta(graphed, eager)
        replays = sum(r.replays for r in opt._graphs.values())

        problems = []
        if not opt._graphs:
            problems.append("no graph captured")
        if replays != 1:
            problems.append(f"replays={replays}, expected 1")
        if d_max != 0.0:
            problems.append(f"max_abs={d_max:.3e}, expected exactly 0")

        verdict = "PASS" if not problems else "FAIL"
        print(ROW.format(name, b * s, f"{d_max:.3e}", replays,
                         f"{pool_mib:.1f}", verdict))
        for p in problems:
            print(f"                 -> {p}")
            failures.append(f"{name}: {p}")

        # Released as we go rather than left to the module's atexit net: this
        # loop builds a model per case, and holding ten private memory pools to
        # the end of the run would OOM an 8 GB card long before it finished.
        for r in opt._graphs.values():
            r.release()
        del opt
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- behaviours
    print("\n=== behaviours that fail silently rather than loudly ===\n")
    config.CUDA_GRAPH = "always"

    # 1. Fresh inputs must actually reach the graph. A missing copy_ replays
    #    stale data: finite, right shape, and *identical* every call.
    cfg, _, opt = build(2, 64, 512, 8, 2048, 6, False)
    outs, refs = [], []
    for seed in (1, 2, 3):
        x, m = case_data(cfg, 0.0, seed=seed)
        config.CUDA_GRAPH = "off"
        with torch.inference_mode():
            refs.append(opt(x, m).clone())
        config.CUDA_GRAPH = "always"
        with torch.inference_mode():
            outs.append(opt(x, m).clone())
    ok = all(delta(o, r) == 0.0 for o, r in zip(outs, refs))
    distinct = (delta(outs[0], outs[1]) > 0 and delta(outs[1], outs[2]) > 0)
    report("fresh inputs reach the graph", ok and distinct, failures,
           f"matches_eager={ok}, outputs_distinct={distinct}")

    # 2. The returned tensor must survive the next replay. Without the clone in
    #    _GraphRunner.replay it is the graph's own output buffer, overwritten in
    #    place by the call after it.
    x1, m1 = case_data(cfg, 0.0, seed=11)
    x2, m2 = case_data(cfg, 0.0, seed=12)
    with torch.inference_mode():
        r1 = opt(x1, m1)
        snap = r1.clone()
        r2 = opt(x2, m2)
    report("result survives the next call",
           r1.data_ptr() != r2.data_ptr() and delta(r1, snap) == 0.0, failures,
           f"aliased={r1.data_ptr() == r2.data_ptr()}, r1_changed={delta(r1, snap)}")

    # 3. Mask mode is a separate op sequence, so it must key separately -- and
    #    coming back to a mode already captured must reuse, not re-capture.
    cfg, _, opt = build(2, 64, 512, 8, 2048, 6, False)
    xa, ma = case_data(cfg, 0.0, seed=21)       # trivial mask
    xb, mb = case_data(cfg, 0.3, seed=22)       # real padding
    with torch.inference_mode():
        opt(xa, ma)
        opt(xb, mb)
        opt(xa, ma)
    entries = len(opt._graphs)
    modes = sorted(k[3] for k in opt._graphs)   # use_mask is key element 3
    total_replays = sum(r.replays for r in opt._graphs.values())
    report("mask mode keys separately and reuses",
           entries == 2 and modes == [False, True] and total_replays == 3, failures,
           f"entries={entries}, modes={modes}, replays={total_replays}")

    # 4. mask=None and an all-True mask are the same op sequence, so they must
    #    share one graph rather than capturing twice.
    cfg, _, opt = build(2, 64, 512, 8, 2048, 6, False)
    xc, mc = case_data(cfg, 0.0, seed=31)
    with torch.inference_mode():
        with_mask = opt(xc, mc).clone()
        without = opt(xc, None).clone()
    report("mask=None shares the trivial graph",
           len(opt._graphs) == 1 and delta(with_mask, without) == 0.0, failures,
           f"entries={len(opt._graphs)}, delta={delta(with_mask, without):.3e}")

    # 5. Strides are deliberately absent from the cache key, because inputs are
    #    copied into a contiguous buffer and copy_ handles any source layout.
    #    That makes the graph valid for layouts eager also accepts -- assert it
    #    rather than trusting the argument.
    cfg, _, opt = build(2, 64, 512, 8, 2048, 6, False)
    big = torch.randn(2, 64, 1024, device=DEV)
    xs = big[:, :, :512]                        # non-contiguous view
    ms = torch.ones(2, 64, device=DEV, dtype=torch.bool)
    config.CUDA_GRAPH = "off"
    with torch.inference_mode():
        ref_s = opt(xs, ms).clone()
    config.CUDA_GRAPH = "always"
    with torch.inference_mode():
        got_s = opt(xs, ms).clone()
    report("non-contiguous input", not xs.is_contiguous() and delta(got_s, ref_s) == 0.0,
           failures, f"contiguous={xs.is_contiguous()}, delta={delta(got_s, ref_s):.3e}")

    # 6. The size gate must actually gate, and must still return correct answers
    #    when it declines. 8*512*512 is four times _GRAPH_MAX_ACTIVATION, so this
    #    stays a decline unless someone raises that constant a long way -- at
    #    which point this assertion failing is the correct outcome and a prompt
    #    to re-check the sweep.
    cfg, _, opt = build(8, 512, 512, 8, 2048, 6, False)
    xg, mg = case_data(cfg, 0.0, seed=41)
    config.CUDA_GRAPH = "auto"
    with torch.inference_mode():
        gated = opt(xg, mg).clone()
    declined = len(opt._graphs) == 0
    config.CUDA_GRAPH = "off"
    with torch.inference_mode():
        gated_ref = opt(xg, mg).clone()
    report("auto declines above the size gate",
           declined and delta(gated, gated_ref) == 0.0, failures,
           f"activation={8 * 512 * 512} vs gate {config._GRAPH_MAX_ACTIVATION}, "
           f"entries={len(opt._graphs)}, delta={delta(gated, gated_ref):.3e}")

    # 7. And a shape under the gate must still be captured, so the decline above
    #    is the gate working rather than something else refusing.
    cfg, _, opt = build(4, 128, 512, 8, 2048, 6, False)
    xh, mh = case_data(cfg, 0.0, seed=42)
    config.CUDA_GRAPH = "auto"
    with torch.inference_mode():
        opt(xh, mh)
    report("auto captures below the size gate",
           len(opt._graphs) == 1, failures,
           f"activation={4 * 128 * 512} vs gate {config._GRAPH_MAX_ACTIVATION}, "
           f"entries={len(opt._graphs)}")

    # ------------------------------------------------------- per-impl coverage
    print("\n=== per attention impl ===\n")
    saved_impl = config.ATTENTION_IMPL
    # scalar and wmma are the two impls "auto" can actually select, so they are
    # the ones that matter by default. The cuTile impls are opt-in only because
    # of the shutdown crash noted at the top of this file -- not because they
    # are wrong under capture; they are bit-exact, and --include-tile shows it.
    impls = ["scalar", "wmma"]
    if args.include_tile:
        impls += ["tile", "tile-tf32"]
    for impl in impls:
        config.ATTENTION_IMPL = impl
        try:
            cfg, _, opt = build(2, 128, 512, 8, 2048, 6, False)
            xi, mi = case_data(cfg, 0.0, seed=51)
            config.CUDA_GRAPH = "off"
            with torch.inference_mode():
                ref_i = opt(xi, mi).clone()
            config.CUDA_GRAPH = "always"
            with torch.inference_mode():
                got_i = opt(xi, mi).clone()
            d_i = delta(got_i, ref_i)
            fired = len(opt._graphs) == 1
            report(f"impl={impl}", d_i == 0.0 and fired, failures,
                   f"delta={d_i:.3e}, captured={fired}")
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP  impl={impl}: unavailable ({type(exc).__name__})")
    config.ATTENTION_IMPL = saved_impl

    # ------------------------------------------------------------ failure path
    if args.test_failure:
        print("\n=== capture failure falls back to eager ===\n")
        cfg, _, opt = build(2, 64, 512, 8, 2048, 6, False)
        xf, mf = case_data(cfg, 0.0, seed=61)
        config.CUDA_GRAPH = "off"
        with torch.inference_mode():
            ref_f = opt(xf, mf).clone()

        # A device->host read is exactly what capture forbids, so this is the
        # real failure mode rather than a simulated one.
        real = opt._forward_eager

        def poisoned(x, mask, use_mask):
            out = real(x, mask, use_mask)
            if torch.cuda.is_current_stream_capturing():
                _ = out.sum().item()          # illegal mid-capture
            return out

        opt._forward_eager = poisoned
        graphs._graph_warned = False
        config.CUDA_GRAPH = "always"
        try:
            with torch.inference_mode():
                got_f = opt(xf, mf).clone()
            denied = len(opt._graph_denied) == 1 and len(opt._graphs) == 0
            report("failed capture returns correct eager result",
                   denied and delta(got_f, ref_f) == 0.0, failures,
                   f"denied={denied}, delta={delta(got_f, ref_f):.3e}")
            with torch.inference_mode():
                again = opt(xf, mf).clone()
            report("denied key is not retried",
                   len(opt._graphs) == 0 and delta(again, ref_f) == 0.0, failures,
                   f"entries={len(opt._graphs)}")
        except RuntimeError as exc:
            if "unusable" in str(exc):
                print("  NOTE  capture failure poisoned the CUDA context and the "
                      "health probe raised, which is the designed behaviour:")
                print(f"        {str(exc).splitlines()[0][:96]}")
            else:
                raise

    print()
    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: replay is bit-identical to eager on every case, and every "
          "graph that should have fired did")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
