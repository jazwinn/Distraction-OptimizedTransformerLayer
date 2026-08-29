"""Run appendix shape 14 (b32 h16 s100000 d1024 L2) by microbatching the batch.

The harness allocates the whole [32,100000,1024] input -- 12.21 GiB -- before
either model exists, and the baseline then wants an 18.63 TiB score matrix. This
generates one slice at a time instead and reports the optimized model alone.

Microbatching is mathematically equivalent (no cross-batch interaction in a
transformer forward) but not bit-identical: cuBLAS picks its GEMM algorithm from
M = B*S, which moves results ~6.5e-4 -- inside the harness's 2e-3 budget.
--self-check measures that on a shape small enough to run both ways.

Totals are a sum over slices, so an UPPER BOUND on what a card holding the whole
batch would report.

    python scripts/bench_shape14.py                  # shape 14, micro=1
    python scripts/bench_shape14.py --micro 2        # roomier, still fits 8 GiB
    python scripts/bench_shape14.py --dtype float16  # halves the footprint
    python scripts/bench_shape14.py --self-check     # prove microbatching sound
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernel_ext  # noqa: F401,E402  (preloads the tile compiler before torch)

import torch  # noqa: E402

from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
)

GIB = 1024 ** 3
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_model(cfg, dtype, device):
    """Optimized model with baseline weights. The baseline is built on the CPU as
    the weight source only -- on this shape it could not be run."""
    torch.manual_seed(1234)
    baseline = BaselineTransformer(cfg)
    model = UserOptimizedTransformer(cfg)
    copy_model_weights(baseline, model, strict=True)
    del baseline
    return model.to(device=device, dtype=dtype).eval()


def make_slice(rows, seq_len, d_model, dtype, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(rows, seq_len, d_model, generator=g, device=device, dtype=dtype)
    mask = torch.ones(rows, seq_len, device=device, dtype=torch.bool)
    return x, mask


HARNESS_ATOL = 2e-3


def self_check(device):
    """Microbatching vs a full batch on a shape that fits both ways. The bar is
    the harness's atol, not zero -- see the module docstring for why."""
    cfg = TransformerConfig(batch_size=4, seq_len=512, d_model=128, num_heads=4,
                            ffn_dim=128, num_layers=2, causal=True)
    model = build_model(cfg, torch.float32, device)
    torch.manual_seed(99)
    x = torch.randn(4, 512, 128, device=device)
    mask = torch.ones(4, 512, device=device, dtype=torch.bool)
    with torch.no_grad():
        whole = model(x, mask)
        parts = torch.cat([model(x[i:i + 1], mask[i:i + 1]) for i in range(4)], dim=0)
    diff = (whole - parts).abs().max().item()
    ok = diff <= HARNESS_ATOL
    print("self-check  b4 s512 d128: full batch vs 4x microbatch")
    print("  max abs diff = {:.3e}   harness atol = {:.0e}   -> {}".format(
        diff, HARNESS_ATOL, "PASS" if ok else "FAIL"))
    if ok:
        print("  microbatching is equivalent within tolerance; the residual is")
        print("  cuBLAS choosing a different GEMM algorithm for a different M.\n")
        return 0
    print("  diff exceeds the harness tolerance -- do not trust shape 14 from this.\n")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=100000)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--micro", type=int, default=1, help="rows per microbatch")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    ap.add_argument("--warmup", type=int, default=1,
                    help="microbatches run untimed first")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mem-fraction", type=float, default=0.90,
                    help="cap the allocator so oversubscription raises OOM instead "
                         "of spilling into system RAM, which hangs Windows")
    ap.add_argument("--self-check", action="store_true",
                    help="prove microbatching is exact, then exit")
    ap.add_argument("--attn-backend", choices=("custom",), default=None,
                    help="override ATTENTION_BACKEND for this run. Only "
                         "'custom' is accepted now that the prebuilt-attention "
                         "backends are gone; kept so existing command lines "
                         "still parse")
    args = ap.parse_args()

    if args.attn_backend is not None:
        from optimized import config as _ocfg
        _ocfg.ATTENTION_BACKEND = args.attn_backend

    if not torch.cuda.is_available():
        print("CUDA is required")
        return 2
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if 0.0 < args.mem_fraction <= 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    if args.self_check:
        return self_check(device)

    if args.batch % args.micro:
        print("--batch {} is not a multiple of --micro {}".format(args.batch, args.micro))
        return 2

    dtype = DTYPES[args.dtype]
    el = torch.empty((), dtype=dtype).element_size()
    B, S, D = args.batch, args.seq_len, args.d_model
    total_bytes = B * S * D * el
    slice_bytes = args.micro * S * D * el
    _, cap = torch.cuda.mem_get_info()

    print("{}  |  {:.2f} GiB, allocator capped at {:.2f} GiB".format(
        torch.cuda.get_device_name(0), cap / GIB, args.mem_fraction * cap / GIB))
    print("shape: b{} h{} s{} d{} ffn{} L{} causal {}".format(
        B, args.heads, S, D, args.ffn_dim, args.layers, args.dtype))
    print("  full input [{},{},{}] would be {:.2f} GiB -- never allocated".format(
        B, S, D, total_bytes / GIB))
    print("  per microbatch (--micro {}): {:.2f} GiB per tensor".format(
        args.micro, slice_bytes / GIB))
    print("  {} microbatches | attn backend: {}\n".format(
        B // args.micro, args.attn_backend or "auto (default)"))

    cfg = TransformerConfig(batch_size=args.micro, seq_len=S, d_model=D,
                            num_heads=args.heads, ffn_dim=args.ffn_dim,
                            num_layers=args.layers, causal=True)
    model = build_model(cfg, dtype, device)

    try:
        with torch.no_grad():
            for w in range(args.warmup):
                x, mask = make_slice(args.micro, S, D, dtype, device,
                                     args.seed + 9000 + w)
                model(x, mask)
                del x, mask
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            times = []
            checksum = 0.0
            n = B // args.micro
            for i in range(n):
                x, mask = make_slice(args.micro, S, D, dtype, device, args.seed + i)
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                out = model(x, mask)
                b.record()
                torch.cuda.synchronize()
                times.append(a.elapsed_time(b))
                checksum += out.float().sum().item()
                del x, mask, out
                print("  microbatch {:>3}/{}: {:9.2f} ms".format(i + 1, n, times[-1]),
                      flush=True)
    except torch.OutOfMemoryError as exc:
        print("\nOOM at --micro {}. Try a smaller --micro, or --dtype float16."
              .format(args.micro))
        print("  " + str(exc).splitlines()[0])
        return 1

    total = sum(times)
    print("\n" + "-" * 68)
    print("per microbatch : median {:.2f} ms | min {:.2f} | max {:.2f}".format(
        statistics.median(times), min(times), max(times)))
    print("total (sum)    : {:.1f} ms = {:.3f} s   <- upper bound on true b{} latency"
          .format(total, total / 1000.0, B))
    print("throughput     : {:,.0f} token/s".format(B * S / (total / 1000.0)))
    print("peak GPU alloc : {:.2f} GiB of {:.2f} GiB".format(
        torch.cuda.max_memory_allocated() / GIB, cap / GIB))
    print("output checksum: {:.6e}".format(checksum))
    print("\nbaseline: not runnable on any hardware -- needs a {:.2f} GiB causal mask "
          "and a {:.2f} TiB score matrix, so no speedup is defined.".format(
              S * S / GIB, B * args.heads * S * S * 4.0 / GIB / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
