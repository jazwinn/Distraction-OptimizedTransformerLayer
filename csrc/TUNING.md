# Kernel tuning notes

Measurements behind the constants in `tile_attention.cu` and `fused_attention.cu`.
All numbers are from an RTX 3070 (SM 8.6, 46 SMs, 448 GB/s), CUDA 13.3.

The source files point here rather than carrying these tables inline. If you
change a block shape or a threshold, change it here too — this file is the only
record of why the current value is the current value.

## Two rules for every measurement below

**Never compare timings across runs.** Run-to-run variance was large enough to
invert a ranking outright: a cross-run comparison "showed" 128x128 beating
128x32 by 1.58x at head_dim 8, when timed interleaved in one process it is 4th
of 5. Rank candidates only within a single run, and re-measure the incumbent
alongside any challenger. This is also why `set_split_kv()` is settable at run
time — on causal long-sequence cases the +/-10-15% variance is larger than the
effect being measured.

**Score short and long sequences together.** Summing raw milliseconds weights
seq_len 2048 about 10x over seq_len 128, so a shape that tanks short sequences
can still win the sum. An early pass scored only 512/2048 and regressed
seq_len 128 by ~20%.

## Which units a math mode actually reached

```
cuobjdump -sass build/tile_attention.cuda.o | grep HMMA
```

Fp32 kernels contain no HMMA; Tf32 emits `HMMA.1688.F32.TF32` and Bf16
`HMMA.16816.F32.BF16`.

## Tile kernel: block shapes

The kernel holds Q, O, K, V and the score tile live at once, so the footprint
goes as `BLOCK_M*HEAD_DIM*2 + BLOCK_N*HEAD_DIM*2 + BLOCK_M*BLOCK_N`. Past a
threshold the compiler spills and the cost jumps an order of magnitude rather
than degrading smoothly — at head_dim 64 in Fp32, BLOCK_N 16 runs at 1.5 ms
where BLOCK_N 32 runs at 10.9 ms.

A narrow mode halves the operand width, which *moves* that cliff rather than
shifting the curve: the head_dim 64 that wants 32x16 in Fp32 runs best at 64x64
in Bf16 — the shape that was worst in Fp32. So shapes are per math mode, not
merely per head_dim.

tf32 is the mode where the two pressures pull apart. It occupies the same 32
bits as fp32, so the spill cliff sits where fp32's does — but it runs on the MMA
units, whose 16x8x8 shape the narrow fp32 tiles (BLOCK_N 16) starve: at
head_dim 16 the inherited 32x16 emitted 4 HMMA where bf16's 64x64 emitted 16.
Inheriting either mode's shapes is wrong, so tf32 was swept separately.

### Dense, tf32

Best-of-5 interleaved rounds over six shapes spanning seq_len 128/512/2048,
causal and dense. Swept by `scripts/tune_tile_tf32.py`.

| head_dim | best   | ms    | runner-up            |
|---------:|:-------|------:|:---------------------|
| 8        | 128x64 | 0.585 | 64x64 0.587 (noise)  |
| 16       | 128x32 | 0.751 | 64x64 0.779 (1.04x)  |
| 32       | 128x64 | 1.419 | 128x32 1.469 (1.04x) |
| 64       | 128x32 | 2.855 | 64x64 3.151 (1.10x)  |

BLOCK_M is 128 in all four, which is exactly what FlashAttention-2 does:
`tile_size_fwd_sm8x` returns `kBlockM=128` unconditionally for every head_dim,
arch and dtype. Only BLOCK_N moves. Reaching for that table first would have
been cheaper than searching a 16x16..128x128 grid blind.

Every extent must be a power of two — cuTile enforces `is_pow2` per dimension
(`crt/cuda_tile.h:749`). FA2's `kBlockN` for the headdim<=64 bucket all four of
these fall into is 112, which cannot be expressed here at all; 64 and 128 are
its legal neighbours, and 128 loses badly at head_dim 64 (11.6 ms).

### Causal, tf32

Causal masking makes a block's cost depend on where it sits: block m walks m+1
key tiles, not S/BLOCK_N of them. A 128-row block therefore does two bad things
at once — it halves the number of blocks available to fill 46 SMs, and it
doubles the *spread* between the cheapest and most expensive block, so the tail
of the grid is longer. Both push the optimum towards a smaller BLOCK_M than the
dense kernel wants. At head_dim 64: seq_len 2048 causal runs 0.867 ms at 64x64
where the dense winner 128x32 runs 0.981; at seq_len 128 causal it is 0.082
against 0.110. Tuning one shape across both mask modes gave up that much.

Swept by `scripts/tune_tile_tf32.py --causal`, which scores only causal cases.
Only head_dim 64 differs from dense (64x64 rather than 128x32).

### Short dense grid, tf32

BLOCK_M sets how many blocks there are: `ceil(S/BLOCK_M) * H * B`. At batch 8,
heads 8, seq_len 128 and BLOCK_M 128 that is 64 blocks on a 46-SM card — 1.4
waves, so the second wave runs 18 blocks against 46 slots and a third of the
device idles for half the kernel. Halving BLOCK_M doubles the block count and
fills it. At head_dim 64, seq_len 128 dense, one interleaved run: 64x32 = 0.096
ms against the long-sequence winner 128x32 = 0.109. The same shape *loses* at
seq_len 512 (0.474 vs 0.376), which is why this is a launch-time choice rather
than a retune. Two waves (`blocks < 2 * sm_count`) is the threshold: 64 blocks
takes the short shape, 128 blocks (seq_len 512 and up) does not, which is
exactly where the two shapes swapped places.

Only head_dim 64 has been measured. The others default to their dense shape,
which turns the short-grid path off for them entirely.

### What no block shape can fix

At seq_len 128 a 128-row block leaves only `batch*heads` blocks in the grid, and
no choice here fills 46 SMs. FlashAttention keeps `kBlockM` at 128 regardless
and solves it by splitting the key dimension instead — see split-KV below.

## Tile kernel: causal block-index reversal

Blocks are dispatched in roughly increasing linear index with grid.x varying
fastest, so the natural mapping hands out the cheapest causal blocks first and
leaves the most expensive for the final wave, where they alone set the makespan.
Reversing it (`m_tile = n_m - 1 - lane`) is longest-processing-time-first.

A/B'd against the identity mapping, both compiled and timed interleaved in one
process:

| case                        | identity | reversed | ratio |
|:----------------------------|---------:|---------:|------:|
| seq 2048 causal head_dim 32 | 0.411    | 0.387    | 1.06x |
| seq 2048 causal head_dim 64 | 0.999    | 0.960    | 1.04x |
| seq 2048 causal head_dim 16 | 0.223    | 0.221    | 1.01x |
| seq 128  causal head_dim 64 | 0.083    | 0.083    | 1.00x |
| seq 512  causal head_dim 64 | 0.301    | 0.317    | 0.95x |

About 1% on the geometric mean, positive on absolute milliseconds because the
wins land on the long cases. Kept for that, not for the theory; three lines to
drop if the seq 512 regression ever matters more. The dense kernel has no such
spread and keeps the identity mapping.

## Tile kernel: split-KV (Flash-Decoding)

Block shape decides how the *query* dimension is cut up, and it runs out at
short sequences: at batch 1, heads 8, seq 512 with BLOCK_M 128 the grid is 32
blocks on a 46-SM card. Shared-memory use pins this kernel at one block per SM,
so 14 SMs do nothing, and no choice of BLOCK_M fixes it — halving BLOCK_M
doubles the block count but also doubles the K/V stream, since every block walks
the whole key range regardless.

Split-KV cuts the *key* dimension instead: each block takes a slice of the keys,
runs the same online softmax over it, and writes an unnormalised partial; a
second pass folds the partials together with one more rescale. That multiplies
the grid by `splits` without touching BLOCK_M, and the extra blocks read
disjoint K/V rather than re-reading the same rows. What it costs is a full pass
over `[B,H,SPLITS,S,HEAD_DIM]` plus the scratch to hold it.

Measured with `scripts/ab_split_kv.py` — both paths compiled into one binary,
toggled at run time, timed round-robin over 5 rounds, best round kept. Geometric
mean of the per-case ratio over the cases that actually split, split-KV against
single-pass:

|      | causal      | dense        | explicit    |
|:-----|:------------|:-------------|:------------|
| fp32 | 1.47x (n=3) | 0.98x (n=7)  | 1.02x (n=2) |
| tf32 | 1.19x (n=5) | 0.98x (n=11) | 1.04x (n=4) |
| bf16 | 1.01x (n=4) | 0.83x (n=8)  | 1.01x (n=3) |

`min_tiles_per_split()` is that table written down:

- **Causal, 4 tiles.** Causal is not buying grid occupancy at all — it is buying
  load balance. Block m walks m+1 key tiles, so the makespan is set by the
  largest block no matter how full the grid is, and cutting every block's own
  range into equal pieces evens that out. It pays at two splits over a short
  range: worst causal case measured 0.90x, best 1.65x.
- **Dense/explicit, 8 tiles.** No imbalance to fix, so all a split buys is idle
  SMs, against a full extra pass. At four tiles per split that trade is a loss
  (0.57x, 0.65x, 0.72x, 0.74x were all four-tile cases); at eight and above it
  turns over (1.05x–1.19x).
- **Dense/explicit bf16, 16 tiles.** The combine pass costs the same bytes in
  every math mode, but bf16's main kernel is the fastest of the three — 0.44 ms
  where fp32 takes 1.39 on b2 h2 s2048 d64 — so the same fixed pass is a much
  larger fraction of it. bf16 dense measured 0.83x overall with nothing above
  1.07x. Sixteen puts it past every dense shape that reaches this code, which is
  the intended effect: dense split-KV is off for bf16 until something measures
  otherwise.

Split-KV and the short-grid block shape attack the same starved grid, so they
are alternatives rather than a stack. Split-KV is the better lever when K/V
traffic dominates Q — long sequences with few heads — because it does not
replicate the Q tile per block. At the short sequences where the grid is
actually starved, Q and K/V are the same size and the two degenerate to the same
trade, so the cheap one (the short shape) wins.

`kMaxSplits = 8`: past a handful of splits the combine pass, which reads
`splits*B*H*S*(head_dim+2)` floats, costs more than the idle SMs it buys back.
`kMaxWorkspace = 96 MB` is a backstop against a pathological shape rather than a
real constraint — batch 8, heads 8, seq 512, head_dim 64 at 4 splits is 34 MB.

## GEMM with fused bias + GELU: measured, and NOT wired into the model

`gemm_bias_gelu_kernel` in `fused_attention.cu` is correct, exported as
`linear_gelu`, and never called by the harness. It is kept as the evidence for
the conclusion below.

On [1024,512]x[512,2048] fp32/tf32:

|                             | ms    | TFLOPS |                |
|:----------------------------|------:|-------:|:---------------|
| cuBLAS GEMM alone           | 0.119 | 18.1   |                |
| cuBLAS GEMM + separate GELU | 0.178 | 12.1   | what to beat   |
| this kernel, GELU fused     | 0.202 | 10.6   | 0.88x — a loss |

The fused activation is worth what it was predicted to be worth; the GEMM
underneath is not competitive. Unfused, GELU reads all M*N of C back out of
global memory, applies one cheap function and writes it again — at [1024, 2048]
that is 16 MB of traffic for 2M flops, and it measured 0.046 ms against the
GEMM's 0.133 ms. Folding it into the accumulator makes it free, which removes
~25% of the unfused pair's cost — and that cannot cover a 1.7x deficit on the
multiply itself. The same holds at every shape the model uses (0.48x to 0.82x of
cuBLAS), including `out_proj`, where cuBLAS is at its weakest (9.4 TFLOPS) and
this kernel still only reaches 7.7.

Block tile is 128x128, forced rather than chosen. Counting only DRAM traffic, a
BMxBN tile does `BM*BN*K` MACs per `(BM+BN)*K` floats loaded, so 64x64 sustains
32 MAC/float — against 448 GB/s that caps the kernel at ~7 TFLOPS, less than
half of what cuBLAS gets. 128x128 doubles the ratio to 64 and lifts the cap to
~14 TFLOPS; L2 reuse between blocks sharing a row or column of tiles has to make
up the rest. 128x256 would be better still and does not fit — its two staging
tiles want 55 KB against the 48 KB budget.

Register prefetching of the next K tile took it from 9.6 to 10.6 TFLOPS, so the
memory pipeline was a real bottleneck but not the binding one. At 10.6 the kernel
sits at 74% of its own tile's DRAM-bandwidth ceiling (~14.3 TFLOPS for 128x128,
ignoring L2 reuse), which says the remaining gap is not one missing trick.
Closing it would need, in rough order of expected value: `cp.async` staging so
the pipeline is not bounded by register pressure; a 4x4 warp tile to lift the
mma-per-fragment-load ratio from 1.33 to 2.0 (needs 128 accumulator registers,
so it may spill); and swizzled block scheduling for L2 reuse. That is a
GEMM-tuning project, not a step in this one.

## CUDA graph capture gate

See REPORT.md — "The gate is on activation volume, and it is a measured
constant" — for the sweep behind `_GRAPH_MAX_ACTIVATION`.

To re-derive it on another machine, `scripts/ab_graph.py --recommend` does the
sweep and the interpreting; README.md, Setup step 4 walks through reading its
output. It obeys both rules at the top of this file: candidates are timed
interleaved in one process, and it carries an eager-vs-eager control row so the
noise floor is measured rather than assumed.
