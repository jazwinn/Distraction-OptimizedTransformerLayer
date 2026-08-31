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
#   "custom"   the custom CUDA kernel, and nothing else. A broken build fails
#              loudly rather than quietly substituting something.
#
# There is no second choice. This used to offer "auto" (kernel if it loads,
# else SDPA) and "sdpa" (always F.scaled_dot_product_attention), and both were
# routes to a prebuilt attention living inside the submission. The whole point
# of the project is that this file implements attention; a switch that quietly
# stops doing so is worse than no switch. To compare against torch's own
# attention, use the benchmark scripts, which are not the model.
ATTENTION_BACKEND = "custom"

# WHICH kernel inside the extension handles attention. What arithmetic it uses
# is ATTENTION_PRECISION, below -- the two are separate axes, and used to not
# be: "tile-fp16" named a kernel and a precision at once, while wmma's
# precision lived in its own ATTENTION_FP16 setting that no other backend
# shared. --attn-impl overrides this for a single run.
#
#   "auto"     the first kernel that covers the shape: the tensor-core kernel
#              wherever it applies, the scalar kernel where that is all there
#              is. It used to prefer SDPA from head_dim 128 up, which is gone
#              -- see run_kernel() in csrc/attention_dispatch.cuh.
#   "scalar"   force the scalar kernel: no tensor cores, everything accumulated
#              in fp32, 5e-6 against an exact reference. Six tuned head_dims
#              plus a generic kernel that takes any head_dim to 2048.
#   "wmma"     force the tensor-core kernel; raises on shapes it misses.
#   "tile"     force the cuTile kernel -- the same math written against the
#              CUDA tile programming model instead of per-thread. float32
#              tensors, head_dim in {8,16,32,64,128,256}, and a build that
#              found CUDA 13.3+. Never picked by "auto": it is a separate
#              programming model whose performance you should opt into.
ATTENTION_IMPL = "auto"

_IMPL_CODE = {"auto": 0, "scalar": 1, "wmma": 2, "tile": 3}

# The arithmetic that kernel contracts q/k/v in. Independent of the tensor
# dtype, which is --dtype: this is what the multiply-accumulate is done in
# after the operands are staged, and narrowing it does not change what is
# stored. --attn-precision overrides this for a single run.
#
#   "auto"   each kernel's own preference. wmma picks fp16, scalar and tile
#            pick fp32 -- which reproduces exactly what the old "wmma" and
#            plain "tile" spellings did, so the default is unchanged.
#   "fp32"   true single precision on the CUDA cores. ~1e-6. Under "auto" this
#            also rules wmma out, since it has no fp32 arithmetic to give, so
#            `--attn-precision fp32` means "the exact one" without having to
#            know which kernel that is.
#   "tf32"   tensor cores, 10 mantissa bits, ~1e-3. The same arithmetic cuBLAS
#            gives the baseline under allow_tf32.
#   "fp16"   tensor cores, also 10 mantissa bits, so the same ~1e-3 -- but
#            16-bit operands, so the fragments contract twice the K per step
#            and run 2.0x-2.25x tf32 on this card. The default for wmma.
#   "bf16"   tensor cores, 8 mantissa bits, ~4e-3. Measurement only: it ran
#            425%-622% of the harness's 2e-3 budget. Exposed so the comparison
#            can be re-run, not because anything should ship on it.
#
# Not every pair exists. scalar is fp32 only, wmma has no fp32, tile has all
# four. A forced impl asking for one it does not have raises and names what it
# does have; "auto" treats the precision as a preference instead.
ATTENTION_PRECISION = "auto"

_PRECISION_CODE = {"auto": 0, "fp32": 1, "tf32": 2, "fp16": 3, "bf16": 4}

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
#            no shape gate. See pick_gemm_tile() in csrc/linear_gelu.cuh.
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

# cp.async for the attention kernel's K/V staging. --cp-async overrides.
#
#   "auto"  cp.async, double buffered where the shared budget allows a second
#           K/V stage -- everything but head_dim 256
#   "sync"  cp.async, single buffered: the copy is committed and waited on
#           before the tile is read, so it bypasses registers but does not
#           overlap the MMAs. This is the intermediate link, kept so the two
#           halves of the chain stay separately measurable
#   "off"   the scalar global->register->shared path
#
# Only has an effect on the fp16 path, which is why it could not exist before
# QKV_FP16: cp.async moves bytes global->shared without passing through
# registers and so cannot convert. While the kernel was handed fp32 tensors and
# contracted them as __half, the narrowing on the staging path ruled it out.
CP_ASYNC = "auto"

_CP_ASYNC_CODE = {"off": 0, "sync": 1, "auto": 2}

# The model's entry LayerNorm -- the one norm with no residual add before it,
# and so the only one _add_layernorm cannot absorb. --layernorm overrides this
# for a single run.
#
#   "auto"  the custom warp-per-row kernel with the residual add compiled out
#   "off"   F.layer_norm, which is ATen
#
# It was the last op in the forward pass still served by a prebuilt function.
# ATen ran it at 198 GB/s on [64,128,128]; the same kernel reaches 350 GB/s
# doing twice the traffic in fused_add_layernorm.
LAYERNORM = "auto"

# The activation-free projections -- QKV, out_proj, ffn_out -- on the same
# custom GEMM as LINEAR_GELU, with the activation compiled out.
#
#   "auto"  fp16 fragments (the fast path)
#   "tf32"  tf32 fragments, bit-comparable with cuBLAS, for A/B
#   "off"   F.linear, which is cuBLAS
#
# Same fp16 PRECONDITION as LINEAR_GELU: operands must fit fp16's range. The
# QKV input is post-LayerNorm and the weights are O(1/sqrt(d)), so it holds by
# construction here. See _linear_bias in optimized/kernels.py for the measured
# per-shape ratios against cuBLAS.
LINEAR_BIAS = "auto"

# Above this many rows, "auto" declines and F.linear runs instead. The kernel
# COVERS every M -- this is a preference, like _FFN_BLOCK_MAX_D, and it lives
# here for the same reason: the extension answers "can I serve this", the
# caller decides "should I".
#
# Measured against cuBLAS on the QKV shape (K=128, N=384), interleaved with a
# self-control, 2026-08-31:
#
#   M          cuBLAS     custom     ratio
#   32768      261.4us    203.5us    1.284x
#   65536      427.5us    373.4us    1.145x     <- shape 13, the largest that wins
#   131072     797.1us    734.5us    1.085x
#   262144     1548us     1614us     0.959x     <- crossover
#   524288     3161us     3415us     0.926x
#   1280000    7973us     8165us     0.977x     <- shape 6
#
# End to end the loss at shape 6 was far worse than that 0.977x suggests --
# 0.902x against a 0.999x self-control. Shape 6 runs its whole forward pass at
# ~86 GB/s against the 350 GB/s the same kernels reach at shape 1, because its
# working set does not fit the 3070's 8 GB; in that regime the epilogue's
# scalar stores (16 consecutive floats per half-warp, a partial 128-byte
# sector) cost much more than they do when the card is not paging. cuBLAS
# vectorizes its epilogue and degrades less.
#
# THE EPILOGUE IS NOW VECTORIZED, so the threshold above has been lifted. The
# store takes 4 elements per lane through `store_out4` (csrc/linear_gelu.cuh),
# which puts 4 lanes on a 64-byte row instead of 16 and cuts store instructions
# fourfold. Re-measuring the same ladder in one process, against a 0.975..1.012
# self-control floor:
#
#   M          cuBLAS     custom     was        now
#   32768      274.0us    211.1us    1.284x     1.298x
#   65536      492.0us    419.7us    1.145x     1.172x
#   131072     926.3us    844.1us    1.085x     1.097x
#   262144     1797us     1685us     0.959x     1.066x   <- was the crossover
#   524288     3527us     3346us     0.926x     1.054x
#   1280000    8584us     8194us     0.977x     1.048x   <- shape 6
#
# Both losses that set the old threshold are now wins, and -- the number that
# actually decides it, because an op probe does not predict this shape -- shape 6
# measured IN THE MODEL reads 1.309x (764.5ms -> 584.0ms) against a 1.002x
# self-control. Lifting the gate also satisfies the fp16 predicates below, which
# both bound themselves by this constant, so shape 6 simultaneously gains the
# fp16 qkv and fp16 normed paths it had been excluded from.
#
# Why a finite bound at all, rather than removing it: 1280000 rows is the
# largest M measured, standalone and in the model. The bound is a documented
# edge of tested ground, not a claim that the kernel loses beyond it. Shape 14
# (the 100k-sequence outlier) is the shape that would push past it and is not
# run in the loop.
#
# This constant is deliberately NOT in graphs.py::_graph_key: every shape that
# captures a graph is at most 8192 rows (shape 7), so both settings dispatch
# identically there and no captured graph can depend on it.
_LINEAR_BIAS_MAX_ROWS = 1 << 21   # 2097152; covers shape 6's 1280000 rows

# Hand the attention kernel its q/k/v already in fp16, instead of fp32 that it
# narrows itself. --qkv-fp16 overrides this for a single run.
#
#   "auto"  fp16 whenever the attention path really does contract in fp16
#   "off"   fp32 qkv, as before
#
# This is not a precision trade. The wmma kernel stages q/k/v into shared memory
# through dev_from_float, which applies __float2half with RNE -- so an fp32 qkv
# buffer is written wide, read wide, and then discarded one mantissa at a time.
# Doing the same rounding in the GEMM epilogue is bit-identical (verified as raw
# int16 bits) and halves both the write and the read: 12 MB -> 6 MB each way at
# grading shape 1.
#
# Only valid when the consumer narrows. An explicit fp32 or tf32 attention
# precision, or the scalar kernel, needs the wide operands -- see
# _attention_wants_fp16 in optimized/kernels.py, which is what gates this.
QKV_FP16 = "auto"

# Below this many rows "auto" declines and qkv stays fp32. Halving a tensor's
# traffic is worth nothing on a shape that never waited on bandwidth, and it is
# not free: the fp16 read path widens four halves per lane with __half2float
# where the fp32 one issues a single float4 load, so out_proj pays ALU for a
# bandwidth saving it cannot use.
#
# Measured in-process against a 0.994x self-control whose worst shape read
# 0.970x:
#
#   M      shape                ratio
#   128    2 - batch 1          0.988x   <- no gain, inside the noise floor
#   512    3 - batch 4          0.943x   <- a real loss, below the floor
#   2048   4 - batch 16         1.051x
#   2048   12 - seq 32          1.047x
#   8192   1 - base             1.101x
#
# The crossover is between 512 and 2048, so the gate sits at 1024. Shapes 2 and
# 3 are the only grading shapes under it.
_QKV_FP16_MIN_ROWS = 1024

# Store a LayerNorm's normalised output as fp16 where its consumer is the QKV
# projection, which narrows it anyway. --normed-fp16 overrides for one run.
#
#   "auto"  fp16 on the edges that feed the QKV GEMM
#   "off"   fp32 everywhere, as before
#
# Same argument as QKV_FP16 and the same bit-identity: the GEMM stages A into
# shared memory through __float2half, so doing it in the LayerNorm epilogue
# moves the rounding one global round trip earlier and changes no value.
#
# Deliberately narrower than it could be. The residual stream x_new is NEVER
# narrowed -- every later layer adds to it and the error would compound. The
# model's final norm is never narrowed either, because that tensor is the
# model's output. And the norm2 edge, which feeds linear_gelu rather than the
# QKV GEMM, is left fp32: linear_gelu has no fp16-input instantiation and
# adding one costs 6 more for a second-order gain.
NORMED_FP16 = "auto"

# Above this d_model, "auto" declines. d_model is the GEMM's K, so it sets how
# many times the k-loop runs -- and the fp16 A path pays four __half2float per
# lane per iteration where the fp32 one issues a single float4 load.
#
# At d_model 128 that is 4 iterations of BK=32 and the halved A read wins. At
# 1024 it is 32 iterations, and that GEMM is compute-bound near the card's TF32
# roofline, so there is no bandwidth to reclaim and the conversion is pure
# overhead. Measured in-process, grading shape 8 (d_model 1024): **0.850x** --
# a 15% loss, against 1.001x-1.040x on every d_model 128 and 32 shape.
#
# Worth being clear about why QKV_FP16 (row 6) does NOT need this gate and won
# 1.02x on the very same shape: there the fp16 was the GEMM's OUTPUT, saving a
# 96 MB -> 48 MB C write at N = 3*1024, which dwarfs the same added ALU. Here
# only the A read changes. Same dtype, opposite verdict, because the tensor it
# applies to is on the other side of the GEMM.
_NORMED_FP16_MAX_K = 128

# Fused post-attention block: add+LayerNorm, Linear+GELU, Linear, add+LayerNorm
# in one kernel. --ffn-block overrides this for a single run.
#
#   "auto"  use it where it is faster, which is d_model <= _FFN_BLOCK_MAX_D
#   "off"   always run the unfused chain
#   "force" use it wherever the kernel covers the shape, ignoring the width
#           gate -- for A/B only, since it loses above the gate
#
# The kernel itself covers d_model and ffn_dim up to 128; this is a *preference*,
# not a coverage rule, and it lives here for the same reason attention's does --
# the extension answers "can I serve this", the caller decides "should I".
FFN_BLOCK = "auto"

# Above this d_model the fused block loses to the unfused chain, so "auto"
# declines it. Measured on the RTX 3070 against the real chain -- the tuned
# linear_gelu kernel, not F.linear:
#
#   d_model 32 (grading shape 7):  5.60x
#   d_model 128 shape 1:  0.979x   shape 5: 0.959x   shape 13: 0.919x
#   d_model 128 shape 6:  0.897x
#
# The crossover is structural. The second GEMM reduces over ffn_dim, so a block
# must own every column of the intermediate, which pins the row tile to 16 -- one
# wmma m-tile. linear_gelu is free to pick a much larger BM and gets better
# arithmetic intensity per weight load. Below d_model 64 the GEMMs are small
# enough that collapsing four launches into one outweighs the worse tiling;
# above it, it does not.
_FFN_BLOCK_MAX_D = 64

# Math ids, matching kGemmMath* in csrc/linear_gelu.cuh.
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

# Split the batch and retry when a forward runs out of memory. Reactive: the
# whole batch is always tried first, so shapes that fit pay nothing. It cannot
# help when the *caller's* input does not fit -- that fails before forward() --
# and results shift ~6.5e-4 when it fires, since cuBLAS picks its GEMM algorithm
# from M = B*S. The chosen chunk size is cached per shape to keep that stable.
MICROBATCH_FALLBACK = True

# Floor on the retry ladder: at one row there is nothing left to split.
_MICROBATCH_MIN_ROWS = 1

# Predictive half of the gate, and on Windows the half that matters: an
# oversubscribed allocation there does not raise, it spills to system RAM and
# crawls, so nothing catches it after the fact. peak =~ FACTOR * one [B,S,D]
# activation; measured 9.03-9.33 on the shapes this fires for. Deliberately not
# 14 (shape 8's ratio): that would predict 8.54 GiB for shape 6, whose real peak
# is 5.51, and split a batch that fits. False positives cost numerics and speed;
# false negatives just fall through to the ladder.
_MICROBATCH_PEAK_FACTOR = 10

# Share of total device memory the prediction may claim before splitting up
# front. 0.85 leaves shape 6 (6.10 GiB predicted of 6.80) running whole.
_MICROBATCH_BUDGET_FRACTION = 0.85
