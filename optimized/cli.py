"""Command-line overrides for the knobs in optimized/config.py.

Three hooks, so the harness carries three lines rather than the whole block:

    parse_args()     -> add_arguments(parser)
    validate_args()  -> validate_args(args, device, dtype)
    main()           -> apply_overrides(args)

Every override is for one run only; the defaults live in config.py.
"""

from __future__ import annotations

import argparse

import torch

from . import config


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the optimization flags to the harness's own parser."""
    parser.add_argument(
        "--attn-backend",
        choices=("custom",),
        default=None,
        help="override ATTENTION_BACKEND for this run only. Only one value is "
             "accepted now: the 'auto' and 'sdpa' choices were routes to a "
             "prebuilt attention and have been removed. Kept as a flag so "
             "existing command lines that pass --attn-backend custom still "
             "work (default: use the value set in optimized/config.py)",
    )
    parser.add_argument(
        "--attn-impl",
        choices=("auto", "scalar", "wmma", "tile", "tile-bf16", "tile-tf32",
                 "tile-fp16"),
        default=None,
        help="override ATTENTION_IMPL for this run only: which kernel inside "
             "the custom extension runs attention",
    )
    parser.add_argument(
        "--attn-fp16",
        choices=("auto", "tf32"),
        default=None,
        help="override ATTENTION_FP16 for this run only: whether the wmma "
             "attention kernel contracts fp32 q/k/v in fp16 fragments (auto) "
             "or tf32. Same 10-bit mantissa either way; fp16 is faster",
    )
    parser.add_argument(
        "--linear-gelu",
        choices=("auto", "tf32", "off"),
        default=None,
        help="override LINEAR_GELU for this run only: fuse the FFN's first "
             "Linear with its GELU into one kernel. 'auto' uses fp16 "
             "fragments, 'tf32' the same kernel at half the tensor-core rate "
             "(for measurement), 'off' cuBLAS plus a separate GELU",
    )
    parser.add_argument(
        "--cuda-graph",
        choices=("off", "auto", "always"),
        default=None,
        help="override CUDA_GRAPH for this run only: capture the optimized "
             "model's forward pass into a CUDA graph and replay it. 'always' "
             "ignores the size gate, for measurement only",
    )


def apply_overrides(args: argparse.Namespace) -> None:
    """Write the requested overrides into config, before any model is built."""
    if args.attn_backend is not None:
        config.ATTENTION_BACKEND = args.attn_backend
    if args.attn_impl is not None:
        config.ATTENTION_IMPL = args.attn_impl
    if args.attn_fp16 is not None:
        config.ATTENTION_FP16 = args.attn_fp16
    if args.linear_gelu is not None:
        config.LINEAR_GELU = args.linear_gelu
    if args.cuda_graph is not None:
        config.CUDA_GRAPH = args.cuda_graph

    # --compile-user with reduce-overhead already puts the model behind
    # Inductor's own CUDA graphs, so hand-rolled capture on top is two
    # mechanisms for one job. Asymmetric on purpose: silently stand down when
    # graphs were only the file's default, so --compile-user keeps working
    # unchanged; refuse when both were asked for explicitly, because then the
    # command line is contradictory and picking one silently would hide it.
    if args.compile_user and config.CUDA_GRAPH != "off":
        if args.cuda_graph is None:
            config.CUDA_GRAPH = "off"
            print("[info] --compile-user: CUDA_GRAPH forced off, since "
                  "--compile-mode reduce-overhead already captures CUDA graphs")
        else:
            raise ValueError(
                "--compile-user and --cuda-graph are two implementations of the "
                "same optimization; reduce-overhead already captures graphs. "
                "Pick one."
            )


def validate_args(args: argparse.Namespace, device: torch.device,
                  dtype: torch.dtype) -> None:
    """Warn about combinations that do nothing. Never raises.

    The runtime gate already handles both of these; raising would make an
    otherwise reasonable command line fail for nothing.
    """
    if args.cuda_graph in ("auto", "always") and device.type != "cuda":
        print("[warning] --cuda-graph has no effect on a non-CUDA device")
    activation = args.batch_size * args.seq_len * args.d_model
    if args.cuda_graph == "always" and activation > config._GRAPH_MAX_ACTIVATION:
        print(f"[warning] --cuda-graph always at {activation} activation elements "
              f"is past the point where replay measured any gain "
              f"({config._GRAPH_MAX_ACTIVATION}); it pins the whole working set "
              f"in the graph's private pool for no speedup")
