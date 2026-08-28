"""Runtime knobs for the optimized model.

Every one of these is read at call time, not captured at import, so a script can
flip one between timed runs:

    from optimized import config
    config.CUDA_GRAPH = "off"

Assigning to a copy imported with `from optimized.config import CUDA_GRAPH`
would not work -- that binds the value, not the setting. Always go through the
module. --attn-backend / --attn-impl / --cuda-graph do the same thing for one
run; see optimized/cli.py.
"""

# Attention backend. --attn-backend overrides this for a single run.
#
#   "auto"     custom CUDA kernel when it builds and loads, else SDPA
#   "sdpa"     always F.scaled_dot_product_attention. No build required.
#   "custom"   require the custom kernel, so a broken build fails loudly
#              instead of quietly benchmarking the fallback and looking slow
ATTENTION_BACKEND = "auto"

# Which kernel inside the extension handles attention; only meaningful when the
# custom backend is in play. --attn-impl overrides this for a single run.
#
#   "auto"       the fastest path for the shape, which is not the same as the
#                first one that covers it: the tensor-core kernel where it
#                wins, the scalar kernel where that is all there is, and SDPA
#                from head_dim 128 up, where the wmma kernel is correct and
#                slower. See run_kernel() in csrc/fused_attention.cu.
#   "scalar"     force the scalar kernel (no tensor cores, no TF32 rounding);
#                head_dim in {8,16,32,64,128}, and raises on anything else. It
#                used to fall through to ATen instead, which meant --attn-impl
#                scalar at a head_dim it did not cover timed ATen and called it
#                the scalar kernel.
#   "wmma"       force the tensor-core kernel; raises on shapes it misses
#   "tile"       force the cuTile kernel, fp32 operands: exact, CUDA cores.
#                float32 and head_dim in {8,16,32,64}, and needs a build that
#                found CUDA 13.3+. Never picked by "auto".
#   "tile-tf32"  the cuTile kernel with its GEMMs narrowed to tf32, which is
#                what puts them on the tensor cores. Same arithmetic cuBLAS
#                gives the baseline under allow_tf32 (~1e-3), so this is the
#                tensor-core mode to reach for first.
#   "tile-bf16"  as above, narrowed to bfloat16 -- 8 mantissa bits, ~4e-3.
#                Expect it to fail the accuracy gate where "tile" passes.
ATTENTION_IMPL = "auto"

_IMPL_CODE = {"auto": 0, "scalar": 1, "wmma": 2, "tile": 3, "tile-bf16": 4,
              "tile-tf32": 5}

# Ask the kernel for [B, S, H*head_dim] rather than [B, H, S, head_dim].
# out_proj wants the flattened layout, and the transpose+reshape that used to
# produce it could not be a view -- it repacked the whole tensor once per layer.
# The kernel epilogue reaches the same addresses for free.
_OUT_LAYOUT_BSHD = 1

# Fuse the FFN's first Linear with its GELU into one kernel. --linear-gelu
# overrides this for a single run.
#
#   "auto"   the fused kernel, fp16 fragments. Faster than cuBLAS + a separate
#            GELU at every shape measured -- 1.24x to 2.40x on the op, across
#            grids from 2 tiles to 20000 and K/N from 32 to 4096 -- so there is
#            no shape gate. See pick_gemm_tile() in csrc/fused_attention.cu.
#   "tf32"   the fused kernel with tf32 fragments instead. Same 10-bit mantissa
#            as fp16 and therefore the same accuracy, but half the tensor-core
#            throughput on this card, so this is for measurement and as a
#            fallback -- not a mode to run in.
#   "off"    always cuBLAS + F.gelu.
#
# Why fp16 rather than tf32: both carry a 10-bit mantissa, so they produce the
# same error against an fp64 reference -- measured identical to three
# significant figures at the attention op, the FFN GEMM, and the whole six-layer
# model. But fp16 tensor cores run 2.0x-2.25x faster here (39.7 vs 17.7 TFLOPS
# at N=2048) and an fp16 fragment contracts 16 elements of K against tf32's 8.
# bf16 is not offered: 8 mantissa bits put it at 425%-622% of the accuracy
# budget, with tens of thousands of failing elements.
#
# Worth having an off switch because this is the one optimization that can move
# the accuracy number. Where cuBLAS picks a TF32 kernel with the same k-order,
# "tf32" here is BIT-IDENTICAL to F.linear + F.gelu; fp16 is not, and costs
# about 1e-4 of extra end-to-end error. Measured against the 2e-3 gate, the
# tightest grading shape sits at 1.35e-3 -- a 33% margin. Do not assume that
# holds across shapes or driver versions; scripts/tune_linear_gelu.py prints
# max_abs per shape.
#
# fp16 also has a PRECONDITION tf32 does not: operands must fit fp16's range.
# This model's activations are post-LayerNorm and its weights are O(1/sqrt(d)),
# so it holds by construction; "tf32" or "off" is the escape hatch if it ever
# does not.
LINEAR_GELU = "auto"

# Math ids, matching kGemmMath* in csrc/fused_attention.cu.
_GEMM_MATH_CODE = {"auto": -1, "tf32": 0}

# CUDA graph capture. --cuda-graph overrides this for a single run.
#
#   "off"     never capture; every forward pass launches its ~79 kernels
#   "auto"    capture when launch overhead dominates and the graph's pinned
#             activation volume batch*seq*d_model is at or under
#             _GRAPH_MAX_ACTIVATION
#   "always"  capture regardless of size, to measure where the crossover
#             actually is. Pins the whole working set; not for benchmark runs.
#
# Launches are asynchronous, so the CPU runs ahead queueing kernel n+1 while the
# GPU works on n: when the average kernel outlasts the time to issue one, launch
# cost is invisible and a graph buys nothing, and when it does not, the GPU
# starves. At batch 1 seq 32 the mean kernel is 6.0 us and the GPU idles 80% of
# the wall clock (4.2x from replay); at batch 8 seq 2048 it is 1018 us, idles
# 0.9%, and a graph is worth nothing. Kernel *count* does not predict this --
# both shapes issue ~79-91 -- so the gate is on activation volume, not on
# launches; see _GRAPH_MAX_ACTIVATION, including how to re-derive it elsewhere.
#
# Replay is bit-identical to eager, not merely close -- same kernels, same
# order, same addresses. _capture_graph verifies that rather than assuming it.
#
# A capture freezes more than it looks like: the kernel chosen for each op, the
# cuBLAS algorithm, allow_tf32/matmul precision, and the extension's own runtime
# knobs (tile_set_split_kv). Only ATTENTION_BACKEND and ATTENTION_IMPL are in
# the cache key, so do not change the rest after the first forward pass -- a
# captured model will quietly ignore you.
CUDA_GRAPH = "auto"

# Elements in one activation tensor -- batch * seq * d_model -- at or below which
# "auto" captures. Set from scripts/ab_graph.py, which measured all three axes
# rather than assuming the obvious one mattered:
#
#   batch*seq*d_model    measured ratio (best of 5, control +/-1.2%)
#   ----------------     -------------------------------------------
#      16384 (b1 s32)        4.23x
#      65536                 2.41x - 3.26x
#     262144                 1.02x - 1.32x
#     524288                 1.029x, 1.038x    <- this gate
#    1048576 and up          1.01x or less, i.e. inside the noise
#
# ---------------------------------------------------------------------------
# THIS NUMBER WAS MEASURED ON AN RTX 3070 (8 GiB, SM 8.6). On other hardware,
# re-derive it:
#
#     cmd.exe /c scripts\devenv.bat python scripts\ab_graph.py
#
# and read off where the ratio column drops into the control column. The
# crossover is the point where the GPU stops starving, which depends on the
# card's throughput relative to how fast the host can feed it -- so a faster GPU
# starves at larger shapes and wants a LARGER value here, and a busier host wants
# a larger one too.
#
# Getting it wrong is cheap in both directions. Too low costs some latency on
# shapes that would have benefited; too high costs some pinned memory on shapes
# that do not. Neither can produce a wrong answer, because replay is bit-identical
# to eager whatever this is set to.
# ---------------------------------------------------------------------------
#
# Why activation volume and not tokens, which is the obvious choice: tokens
# mispredict badly. At 512 tokens, d_model 256 gave 2.708x and d_model 512 gave
# 1.036x -- the same token count, a 2.6x difference in payoff. Work per kernel
# scales with the activation tensor rather than with its rows, and a pure token
# gate of 1024 would have declined batch 8 seq 256 at d_model 256, measured at
# 1.038x.
#
# num_layers does not belong here at all. At fixed activation volume, 3/6/12/24
# layers gave 1.031x/1.037x/1.040x/1.040x -- eager and replay scale with depth
# together.
#
# Nothing swept was ever *slower* than eager (worst case 0.998x, inside the
# control's spread), so this gate is not protecting against a slowdown. It caps
# the pinned pool, which at this threshold measured about 84 MiB.
_GRAPH_MAX_ACTIVATION = 1 << 19    # 524288

# Safety net, deliberately not the gate: a captured pool larger than this share
# of the card is released rather than held for the whole run. At the threshold
# above it should never fire -- 84 MiB against 2 GiB on an 8 GiB card -- and that
# is the point. It is there so that a much larger _GRAPH_MAX_ACTIVATION set on
# unfamiliar hardware, or a pool that turns out bigger than it was here, degrades
# to eager instead of quietly eating the card.
_GRAPH_POOL_SAFETY_FRACTION = 0.25

# Each captured graph holds its own private memory pool for the whole run, so
# this caps memory, not time. One shape and one mask mode is the normal case; 4
# leaves room for a mask-mode flip without letting a caller capture dozens.
_GRAPH_MAX_ENTRIES = 4

# Iterations before capture, the count torch's own make_graphed_callables uses.
# One would populate every Python-level cache; three is cheap insurance against
# anything initializing on a second or third touch (cuBLAS heuristic caches,
# per-kernel lazy module loading).
_GRAPH_WARMUP_ITERS = 3

# Refuse to capture with less than this much device memory free. Cheaper than
# discovering it by OOM, though OOM is handled too.
_GRAPH_MIN_FREE_BYTES = 512 << 20

# Capture feeds the model its own data rather than the caller's, from a local
# generator so it cannot perturb the global RNG stream main() seeds.
_GRAPH_SEED = 20240827
