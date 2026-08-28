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

## Scalar kernel: the register ceiling, and head_dim 128

A thread holds `q_reg[HEAD_DIM]` and `acc[HEAD_DIM]` for its query row, which is
what makes the key loop a register FMA instead of a reload. At head_dim 128 that
is 256 registers per thread against a hardware ceiling of 255 — which is why the
kernel stopped at 64, and why any measurement labelled "scalar at head_dim 128"
before this was ATen (see REPORT.md).

`ScalarCfg` fits it by capping `DIMS` — the dims one thread owns — at 64 and
splitting the row across `TPR = HEAD_DIM / DIMS` adjacent lanes. Measured with
`cuobjdump -res-usage build/fused_attention.cuda.o`:

| dtype, head_dim | threads/block | shared | REG | LOCAL |
|:----------------|--------------:|-------:|----:|------:|
| float32, 8      | 64            | 8 KB   | 48  | 0     |
| float32, 16     | 64            | 16 KB  | 63  | 0     |
| float32, 32     | 64            | 32 KB  | 96  | 0     |
| float32, 64     | 64            | 32 KB  | 165 | 0     |
| float32, 128    | 128           | 32 KB  | 168 | 0     |

`LOCAL:0` at head_dim 128 is the whole point: the naive shape does not compile
to a slow kernel, it compiles to a spilling one.

`BLOCK_N` is chosen to hold `k_s + v_s` at 32 KB from head_dim 64 up — 128, 64
and 32 keys per tile at head_dim 32, 64 and 128 — which keeps two blocks
resident against the 48 KB budget. At head_dim 128 float32 that is 3 blocks per
SM by both shared memory and registers, so 12 warps against the 2 the
one-thread-per-row shape would have managed.

Costs one `__shfl_xor_sync` per key when `TPR == 2`, and nothing at all when
`TPR == 1` — `if constexpr` deletes it, so the existing head_dims compile
unchanged. Shared-memory traffic is identical either way: the two partners read
disjoint halves of the key row.

`SUPPORTED` is `SMEM <= 48 KB`, and the launcher returns false rather than
launching when it fails. float64 past head_dim 16 wants 64 KB and used to launch
and fail; it now reports a coverage gap like every other impl does.

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
causal and dense. Swept by `scripts/tune_block_shapes.py --backend tile-tf32`.

| head_dim | best   | ms    | runner-up            |
|---------:|:-------|------:|:---------------------|
| 8        | 128x64 | 0.585 | 64x64 0.587 (noise)  |  <!-- superseded: see re-sweep below -->
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

Swept by `scripts/tune_block_shapes.py --backend tile-tf32 --causal`, which
scores only causal cases.
Only head_dim 64 differed from dense (64x64 rather than 128x32) when this
was written; the re-sweep below moved head_dim 8 and 16 as well.

### 2026-08-28 re-sweep: what changed and what did not

`scripts/tune_block_shapes.py` replaced `tune_tile_tf32.py` and now covers wmma
and all three tile math modes. Re-running every cell against the current kernels
moved five of them. Each margin below is candidate against incumbent **within one
interleaved run**, which is the only comparison this file's first rule allows:

| backend | mask | head_dim | was    | now    | margin |
|:--------|:-----|---------:|:-------|:-------|:-------|
| tile-fp32 | dense  | 8  | 64x64  | 64x32  | 1.34x  |
| tile-tf32 | causal | 8  | 128x64 | 64x64  | 1.31x  |
| tile-tf32 | dense  | 8  | 128x64 | 128x16 | 1.13x  |
| tile-fp32 | causal | 32 | 64x16  | 32x16  | 1.17x  |
| tile-tf32 | causal | 16 | 128x32 | 64x64  | 1.12x  |
| wmma      | both   | 64 | 64x32  | 64x16  | 1.07x  |

tile-fp32 causal head_dim 32 is the one cell where the two mask modes genuinely
want different fp32 shapes: dense keeps 64x16, causal wants 32x16. Before the
`FP32_CM_*` macros existed that was not expressible -- fp32 and bf16 had no
causal shape axis at all, and the causal kernel silently ran the dense shape.
The other three fp32 causal cells measured out identical to their dense values
and still inherit them.

bf16 (both mask modes) and wmma at float16/bfloat16 were **not swept** -- the run
was stopped before them. bf16 still carries 64x64 at every head_dim, which given
how fp32 and tf32 behaved at head_dim 8 and 16 is where the next win most likely
sits.

Every other cell held, or moved by less than the noise floor and was left alone.
Four of the five sit at head_dim 8 or 16 -- the cells this file records as
*inherited* rather than measured. The two that got real attention at the time
(tf32 causal 32 and 64) both survived re-measurement unchanged. Inheriting a
shape is what cost the 12-31% here, exactly as the tf32-vs-fp32 argument above
predicts; the fix is to measure the small head_dims, not to find a better rule.

**The noise floor is about 4.3%.** Two runs of the same shape (wmma head_dim
128, 32x16) on an idle machine read 9.397 and 9.010 ms. Treat anything under
that as a tie and keep the incumbent -- several cells nominally changed winner
at 1.00-1.03x and were deliberately not touched.

**The absolute milliseconds in the tables above no longer reproduce.** The same
script, same case filter, same iters and rounds, measures roughly 1.9x faster
across every tf32 cell than the numbers recorded here. The rankings are mostly
intact and every qualitative claim still holds -- 128x128 is still worst at
head_dim 64, the fp32 BLOCK_N=16 cliff is still there -- so this is a systematic
shift, not a reordering. It is unexplained. Do not use the recorded ms as a
baseline for anything; re-measure the incumbent alongside the challenger, which
rule one already requires.

**wmma carries one shape for all three dtypes.** `WmmaShape` is not parameterised
on `scalar_t`, so the head_dim 64 change above was chosen on float32 -- the dtype
the benchmark defaults to -- and applies to float16 and bfloat16 whether or not
it suits them. A float16 sweep is the outstanding check; if it disagrees, that is
the evidence for adding the parameter.

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

## wmma kernel: causal block-index reversal, gated on one wave

The same idea as the tile kernel's reversal above, ported to the hand-written
kernel and measured by `scripts/ab_causal_reverse.py`. It does not port
unconditionally: ungated it is a wash, and the reason it is a wash is the useful
part.

Ungated, over 13 causal shapes: geometric mean **1.002x**, with individual
shapes from 0.933x to 1.101x. Both extremes reproduce to within 1% across runs,
so that spread is not noise -- the dense control rows read 0.2%, and a
`--self-control` run (the reversed mapping timed against itself, so every true
ratio is exactly 1.000) puts the causal noise floor at 1.2%. Sorting by how many
waves the grid fills explains it:

| blocks | vs ~138 resident | ratio |
|:-------|:-----------------|------:|
| 64     | 0.5 wave         | 0.988x |
| 128    | 0.9 wave         | 0.933x - 1.005x |
| 256    | 1.9 waves        | **1.101x** |
| 512    | 3.7 waves        | 1.018x - 1.029x |
| 1024   | 7.4 waves        | 0.995x |

Reversal reorders a *queue*. Below one wave there is no queue -- every block is
already resident, dispatch order decides nothing, and all the reordering can do
is scatter K/V locality between the blocks running side by side. Above one wave
LPT bites, and it bites hardest at about two waves, where the tail is a large
fraction of the makespan. By eight waves the tail is amortised and it washes out
again.

So the mapping is gated on `blocks > resident`. `resident` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than from dividing the
SM's shared memory by `Cfg::SMEM`: registers can bind before shared memory does,
and only the driver knows which did. It is a property of (kernel, threads,
smem), all compile-time constants here, so it is queried once per instantiation
rather than per launch.

**head_dim 128 is excluded outright.** It runs a 32x16 block of two warps at
~36 KB, so an SM holds two blocks and 128 threads. With that little in flight
the kernel is bound by K/V locality rather than by makespan, and the reordering
costs more L2 reuse than it saves tail. Five shapes, none of them positive:

| case                    | blocks | ratio |
|:------------------------|-------:|------:|
| b1 h8 s2048 d128 causal |    512 | 0.889x |
| b1 h4 s2048 d128 causal |    256 | 0.903x |
| b4 h8 s1024 d128 causal |   1024 | 0.948x |
| b2 h8 s1024 d128 causal |    512 | 0.952x |
| b2 h4 s1024 d128 causal |    256 | 0.966x |

Gated and with head_dim 128 out, over 17 causal shapes: geometric mean
**1.009x**, worst case 0.996x against a 0.3% control -- no shape regresses, and
`b1 h8 s2048 d64 causal` keeps its 1.096x. End to end at that shape
(`--seq-len 2048 --batch-size 1 --causal`) the harness min goes 9.24 -> 8.96 ms,
about 3%, reproduced over two process pairs.

Two limits worth knowing:

- **The padded-causal path used not to reach it.** `MySelfAttention` folded
  causal into an explicit `[B,1,S,S]` mask whenever there was real padding to
  combine with, which arrived as `is_causal=false`; the kernel could not tell
  that mask was triangular, so the imbalance was there and the fix was not, and
  those rows appear as controls in the A/B table above rather than as
  measurements. Fixed since — see "Causal and an explicit mask, together". The
  A/B was not re-run afterwards, so those control rows are still controls in
  the table; the reversal now applies to them.
- **`WMMA_CAUSAL_REVERSE=0` disables it**, and `wmma_set_causal_reverse()` flips
  it at run time so both mappings can be timed in one process. Like
  `tile_set_split_kv`, it is invisible to the CUDA-graph cache key -- flipping
  it after capture does nothing to a captured model.

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

## Auto's kernel preference: where coverage and speed come apart

`run_kernel(Impl::Auto)` used to mean "the first kernel that can do this",
which is only the right rule while covering a case and being the fastest way to
serve it are the same thing. At head_dim 128 they are not.

wmma against SDPA, fp32 causal, interleaved in one process. Ratio > 1 means the
wmma kernel is slower:

| head_dim 128 | batch 8 | batch 1 |
|:-------------|--------:|--------:|
| seq 32       |  0.42x  |  0.59x  |
| seq 64       |  0.98x  |  0.86x  |
| seq 128      |  1.47x  |  1.22x  |
| seq 256      |  1.46x  |  1.20x  |
| seq 512      |  1.52x  |  1.44x  |

The reason is in `cuobjdump -res-usage`: `fused_attention_wmma_kernel<float,
128>` reports **REG:255**, one under the hardware ceiling. head_dim 128 gives
each warp `q_frag[PDIM/WK]` = 16 fragments, 128 registers of query before a
single accumulator exists, and that is what forces `WmmaShape<128>` down to
32x16 — two warps, ~36 KB, two blocks and 128 threads per SM. There is not
enough in flight to cover memory latency, and no block shape fixes it while Q is
register-resident. (head_dim 64 reports REG:168, head_dim 32 REG:128.)

So Auto now takes wmma at `head_dim <= 64`, or at head_dim 128 when
`seq_len < 128`, and otherwise falls through. The crossover is where SDPA's
fixed per-launch cost stops dominating: seq 64 is a tie at the noise floor and
seq 128 is not. `--attn-impl wmma` still reaches the kernel at every head_dim it
covers, because a forced impl means that kernel or nothing.

float16 crosses earlier and harder (2.85x at seq 128, 3.30x at seq 512, where
SDPA reaches its flash backend), so a gate on head_dim and length alone is
conservative for the narrow dtypes rather than wrong for them.

End to end, `--causal --d-model 1024 --ffn-dim 1024` (head_dim 128), auto
against a forced wmma, two process pairs: **1.361x / 1.345x** against
**1.306x / 1.310x**. Accuracy improves alongside — `max_abs` 1.21e-3 against
1.35e-3.

## The uncovered-shape fallback is SDPA, not the baseline's own matmul

The path taken when no kernel covers a case used to mirror
`BaselineSelfAttention.forward` exactly: `matmul`, mask, `softmax`, `matmul`.
That read as the safe choice and was the opposite of one — it materializes the
whole [B, H, S, S] score matrix, so the one path this file takes when it has
nothing better ran the *baseline's* algorithm and inherited its memory traffic.

head_dim 256 is the only head_dim in the grading set no kernel covers. The old
fallback against SDPA, interleaved, causal fp32:

| head_dim 256 | batch 8 | batch 1 |
|:-------------|--------:|--------:|
| seq 32       |  3.91x  |  6.29x  |
| seq 128      |  1.37x  |  5.20x  |
| seq 256      |  1.29x  |  2.02x  |
| seq 512      |  1.30x  |  1.08x  |

Nothing measured was slower, and the largest wins are where the score matrix is
largest relative to the useful work. End to end at `--causal --d-model 2048
--ffn-dim 2048` the harness reports 1.170x, against the 1.19x recorded for that
shape before — the same, because at d_model 2048 the four projection GEMMs
dominate what attention costs either way.

## Causal and an explicit mask, together

`fused_attention_forward` used to reject `is_causal` alongside `attn_mask`,
copying SDPA's restriction. SDPA has to have it; this ABI did not. Both kernels
already apply the two independently — a causal `break`/predicate and a separate
mask lookup — so allowing the pair needed no kernel change at all.

What it buys is not one less tensor. `is_causal` in these kernels is not
masking, it is *skipping*: `key_limit` stops the key loop at the block's own
last row instead of walking S. Folding the triangle into a [B, 1, S, S] mask
hides that from the kernel, which then computes the upper half of the score
matrix and discards it. On the attention op alone, folded against
causal + a [B, 1, 1, S] key mask, interleaved:

| case                | gain  |
|:--------------------|------:|
| B8 H8 seq 128 d64   | 1.18x |
| B8 H8 seq 512 d64   | 1.69x |
| B8 H8 seq 1024 d64  | 1.85x |
| B64 H8 seq 128 d32  | 1.40x |
| B8 H8 seq 128 d8    | 1.24x |
| B3 H5 seq 37 d32    | 0.97x |

converging on the 2x that is exactly the half of the matrix the early exit
skips. Error against the reference is identical to seven digits either way —
this is the same arithmetic on fewer keys, not an approximation. It also
restores the causal block-index reversal to the padded path, which the section
above records as unreachable while the fold was in the way.

Whole model, in-process interleaved with a control row (the fused path timed
against itself, true value 1.000x), `--padding-ratio 0.3 --causal`:

| shape                  | gain   | control |
|:-----------------------|-------:|--------:|
| B8 seq 1024 d_model 512 | 1.285x | 1.001x |
| B8 seq 512  d_model 512 | 1.158x | 1.004x |
| B64 seq 128 d_model 256 | 1.063x | 0.998x |
| B8 seq 128  d_model 512 | 1.009x | 0.990x |
| B1 seq 128  d_model 256 | 0.984x | 1.005x |

The two short-sequence rows are ties against a ~1% control. The win is the
early exit, so it grows with what the early exit skips.

**The tile kernels are excluded.** `MaskMode` is a template parameter with
None/Causal/Explicit and no combined mode, so `launch_tile` declines the pair
rather than silently dropping half of it. Adding a fourth mode is the work, if
the tile kernels ever join Auto.

## Two things measured and NOT done

**cuBLASLt's fused GELU epilogue.** `torch._addmm_activation(bias, x, w.t(),
use_gelu=True)` reaches `CUBLASLT_EPILOGUE_GELU_BIAS`, which would remove the
separate GELU kernel — 4.6% of GPU time in a profile of the FFN-heavy grading
shape. It fails on both counts. It is the **tanh** approximation, not erf: it
matches `F.gelu(approximate="tanh")` to 5.6e-6 and differs from
`approximate="none"` by 4.7e-4, which is a quarter of the whole atol budget
spent on one op. And on [8192, 256] x [256, 256] it measures **0.842x** —
0.1754 ms against 0.1477 for the unfused pair, because the Lt heuristic picks a
worse kernel than plain cuBLAS does. Fused is not automatically faster.

**cp.async double-buffered K/V staging in the wmma kernel.** The obvious next
step from the priority list, and the arithmetic says no. At head_dim 64 the tile
footprint leaves room to double the K/V buffers (32.0 KB to 40.7 KB) but that
drops the SM from three resident blocks to two, and there is nothing to buy with
the occupancy. `b64 h8 s128 d64 causal` moves about 84 MB (Q and O in full, K/V
once per m-tile) in 0.226 ms — **371 GB/s, 83% of the card's 448**, before
counting the L2 reuse between the two m-tiles that share a (b, h). The kernel is
already bandwidth-saturated at the large shapes and latency-bound only where the
grid is too small to fill the card, which is what split-KV and the short-grid
shape address. Overlapping the load with the compute has almost nothing left to
recover.

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

## fused_add_layernorm: two block reductions instead of three

The block-per-row kernel reduced `sum(c)` and `sum(c*c)` separately, though both
are produced in one loop and consumed at the same point -- so it walked the whole
two-stage block reduction twice. `block_reduce_sum2` does both at once: same
shuffle count (two values still have to move), one shared round trip instead of
two, and **two `__syncthreads()` instead of four**.

Bit-identical to the split form -- same values summed in the same order -- and
verified as such rather than to a tolerance, across fp32/fp16/bf16, D from 1 to
4096, and the large-mean cases the corrected two-pass form exists for.

### The prize was bounded before the code was written

Since the warp kernel took D <= 256, this only serves D > 256. The bound comes
from the warp kernel itself, which has *no* shared reduction and *no* barriers at
all -- so at matched traffic its margin over the block kernel is everything
reduction restructuring could ever buy:

| rows x 256 | block | warp (zero reductions) | warp ahead by |
|---|---|---|---|
| 4096 | 50.00 us | 49.10 | 1.8% |
| 16384 | 195.24 | 193.48 | 0.9% |
| 65536 | 772.48 | 776.29 | -0.5% |

At the row counts the model runs, removing *all* reduction overhead is worth
0-2%. This change captures part of that.

### Measured, control +/-1.6%

| shape | warps/blk | split | fused | ratio | control |
|---|---|---|---|---|---|
| 1024 x 64 | 2 | 4.18 us | **3.67** | **1.139x** | 1.002 |
| 64 x 512 | 8 | 2.66 | **2.47** | **1.073x** | 0.994 |
| 1024 x 320 | 8 | 14.70 | **14.06** | **1.045x** | 0.995 |
| 1024 x 512 | 8 | 22.97 | 22.62 | 1.015x | 1.014 |
| 1024 x 2048 | 8 | 88.80 | 87.40 | 1.016x | 1.016 |
| 1024 x 1024 | 8 | 45.00 | 44.96 | 1.001x | 1.009 |
| 4096-16384 rows | 8 | - | - | 0.998-1.009x | ~1.00 |

The pattern is clean: **it wins where the kernel is latency-bound and warps are
few, and ties where DRAM saturates.** 1024 x 64 at two warps is the largest win
because the barrier is proportionally largest there -- and it is also the least
useful, since D=64 goes to the warp kernel in a default run.

### End to end: no measurable effect

| shape | split | fused | ratio |
|---|---|---|---|
| hd64 B8 S128 d512 | 2.1849 ms | 2.1832 | 1.0008x |
| hd128 B8 S128 d1024 | 6.2242 | 6.0279 | 1.0326x (unresolved) |
| hd256 B8 S128 d2048 | 18.4931 | 18.2236 | 1.0148x (unresolved) |

d512 is stable to 0.1% across runs and says **tie**. d1024 and d2048 carry 5-11%
run-to-run spread within a single variant, which cannot resolve a 1% effect --
their ratios are noise, not results.

**So this is kept for being strictly less work at identical output, not for a
number.** The honest summary is: a real 4-14% win in the latency-bound regime, a
tie everywhere the model actually operates.

### Measurement notes

Two things moved the control from +/-12% to +/-1.6%, and both were method, not
hardware:

- **Round-robin, not batched.** Timing every round of A and then every round of B
  lets drift over the run appear as a difference between them. Racing
  pre-captured graphs alternately within each round cancels it: +/-12% -> +/-4.7%.
- **A quiet machine.** A game running in the background cost ~15% absolute (1024
  x 512: 26.15 -> 22.97 us) and tripled the control: +/-4.7% -> +/-1.6%. Nothing
  in this file was measured against a busy GPU on purpose; check before trusting
  a marginal row.

The control here is the split form raced against *itself* on the same shape at
the same moment -- identical code, so its ratio is that shape's noise, rather
than a noise figure borrowed from a separate pass.

Knob: `layernorm_set_fused_reduce(bool)` / `LAYERNORM_FUSED_REDUCE`. Both forms
stay compiled so the table above can be re-derived on other hardware.

## fused_add_layernorm: warp per row, several rows per block

### What was measured first, and what it got wrong

`scripts/bench_layernorm_occupancy.py`, occupancy straight from the driver:

| D | threads | warps/blk | blocks/SM | warps/SM | of max | limiter | vs roofline |
|---|---|---|---|---|---|---|---|
| **32** | 32 | 1 | **16** | 16 | **33%** | blocks/SM cap | **1.48x off** |
| 64 | 64 | 2 | 16 | 32 | 67% | blocks/SM cap | 0.98x |
| 128 | 128 | 4 | 12 | 48 | 100% | warps | 0.97x |
| 256+ | 256 | 8 | 6 | 48 | 100% | warps | 0.99-1.00x |

Read literally this says: D=32 is starved because one row needs one warp and
sm_86 caps resident blocks at 16 however small they are; D=64 already reaches
67% and that is enough to saturate; so this is a D=32 problem and the fix is
more rows per block. **Every one of those statements is true and the conclusion
from them was wrong.**

The rows-per-block sweep is what exposed it. At D=32, 524288 rows:

| rows/block | warps/SM | us |
|---|---|---|
| 1 | 16 | 702 |
| 2 | 32 | 679 |
| 4 | 48 | 719 |
| 8 | 48 | 689 |
| 16 | 48 | 694 |

One row per block has **identical occupancy to the block-per-row kernel** -- 16
warps of 48 -- and already gets 702 us against its 1016. And the rest of the
sweep is flat inside a +/-9% control. If occupancy were the constraint neither
could be true.

What the warp kernel actually removes is the shared-memory round trip, the
cross-warp reduction stage, and all six `__syncthreads()`. That is a **latency**
win, not a bandwidth one, so it pays wherever the kernel is latency-bound --
which is any moderate row count at any width, not just narrow rows.

### Results

Graph replay with 20 calls captured per graph (see the trap below):

| shape | block | warp | ratio |
|---|---|---|---|
| 1024 x 32 | 3.58 us | **1.66** | **2.16x** |
| 1024 x 64 | 4.17 us | **1.86** | **2.24x** |
| 1024 x 256 | 8.60 us | **3.99** | **2.15x** |
| 4096 x 32 | 9.53 us | **3.24** | **2.94x** |
| 16384 x 32 | 33.13 us | **22.51** | 1.47x |
| 8192 x 256 | 88.11 us | 86.95 | 1.01x |
| 1024 x 512 | 23.01 us | 22.98 | 1.00x |

The two ties are the shapes big enough to saturate DRAM -- nothing left to win,
nothing lost either. **D=64 and D=256 are 2.2x wins and the occupancy reading
said both were fine**, which is why the width threshold is 256 and not the 32
the first pass set.

Bandwidth regime (17M elements at every D), where only D=32 moves:

| D | block | warp | ratio | block GB/s | warp GB/s |
|---|---|---|---|---|---|
| 32 | 1015.7 us | **672.4** | **1.51x** | 264 | **399** |
| 64 | 680.0 | 678.2 | 1.003x | 395 | 396 |
| 128+ | - | - | 0.998-1.001x | ~397 | ~397 |

D=32 lands exactly on the measured 395 GB/s roofline.

### End to end

Alternating runs, three per variant, `--layers 6 --causal`:

| shape | block | warp | ratio | speedup vs baseline |
|---|---|---|---|---|
| hd8 B8 S128 d32 | 0.2683 ms | **0.2427** | **1.105x** | 20.11x -> 22.17x |
| hd16 B8 S128 d256 | 0.7861 | **0.7461** | **1.054x** | 6.72x -> 7.13x |
| hd32 B8 S32 d256 | 0.3267 | **0.3174** | 1.029x | 17.13x -> 17.54x |
| hd32 B16 S128 d256 | 1.7842 | **1.7371** | 1.027x | 2.88x -> 2.94x |
| hd32 B1 S128 d256 | 0.3079 | **0.3031** | 1.016x | 18.88x -> 19.12x |
| hd32 B64 S128 d256 | 5.1331 | 5.1084 | 1.005x | bandwidth-bound |
| hd64 B8 S128 d512 | 2.1917 | 2.1924 | 1.000x | control (block kernel) |
| hd128 B8 S128 d1024 | 6.0228 | 6.0238 | 1.000x | control |

Cumulative on the hd8 shape, all three variants in one thermal state:

| | ms | speedup |
|---|---|---|
| original, flat 256 threads | 0.3164 | 17.08x |
| width-scaled block size | 0.2683 | 20.15x |
| **warp per row** | **0.2430** | **22.37x** |

### The threshold

`layernorm_warp_width()` defaults to 256, which is the widest
`ELEMS_PER_LANE` the launcher instantiates (8 * 32). It stops there because
1024 x 512 measures 23.0 us against a 21.2 us traffic floor -- already
bandwidth-bound, so a wider instantiation would buy nothing.

`layernorm_warp_rows()` is 4, and the sweep above says it barely matters; 4 is
the middle of a flat region.

Knobs: `layernorm_set_warp_width(n)` (0 disables, -1 restores the default) and
`layernorm_set_warp_rows(n)`, plus `LAYERNORM_WARP_WIDTH` / `LAYERNORM_WARP_ROWS`.

### Two measurement traps, both of which produced a wrong number first

**Capturing one call per CUDA graph measures the replay, not the kernel.**
Replaying a graph costs several microseconds whatever it holds. A first pass
captured a single call and reported 8.02 vs 8.95 us at 1024x32 -- a 0.90x
*regression* on the shape this was built for. The tell was 256x32, a quarter of
the work, coming out at 7.99 vs 8.84: two shapes differing 4x in work and
0.03 us in time are not being measured. Capturing 20 calls per graph turned the
same comparison into 3.58 vs 1.66 us, 2.16x.

**The first run in a fresh process is not comparable to the rest.** An
end-to-end sweep reported 1.321x on hd8; re-run in a settled state it was
1.105x, and a separate three-way sweep of the same shape agreed at 1.104x. Every
*other* shape in that sweep matched to within 0.002x across both runs -- only the
first one was wrong. Run a throwaway shape first, or discard run one.

## fused_add_layernorm: the block size has to scale with d_model

`threads` was a flat 256 in the launcher. That is right from d_model 256 up --
every thread owns at least one element and the loops stride for the rest -- and
it collapses below it. At d_model 32 only 32 of the 256 threads load anything,
while all eight warps still run three `block_reduce_sum` calls, six
`__syncthreads()` and a shared-memory round trip over what is mostly zeros.

Fixed by `layernorm_block_threads()`: one thread per element, rounded up to a
whole warp, capped at 256.

### Bandwidth regime -- ~16M elements at every width, so nothing is dispatch-bound

Floor is analytic: 4N floats (read x, read sub, write x_new, write normed) at
the card's **measured** 384 GB/s copy bandwidth, which is 698.68 us at this
element count for every row. It is not a `torch.add` row -- `torch.add` is
itself dispatch-bound at these sizes and reported a "floor" *above* the kernel
it was meant to bound, which is what made the first pass of this sweep useless.

| rows x D | old (t256) | new | speedup | new vs roofline |
|---|---|---|---|---|
| 524288 x 32   | 3197.74 | **1075.44** | **2.97x** | 1.54x off |
| 262144 x 64   | 1647.27 | **711.30**  | **2.32x** | 1.02x |
| 131072 x 128  | 941.70  | **674.68**  | **1.40x** | 0.97x |
| 65536 x 256   | 719.50  | 704.92 | 1.00x (same launch) | 1.01x |
| 32768 x 512   | 708.88  | 716.70 | 1.00x | 1.03x |
| 16384 x 1024  | 692.45  | 706.32 | 1.00x | 1.01x |
| 8192 x 2048   | 710.72  | 711.24 | 1.00x | 1.02x |

So D <= 128 went from 1.3x-4.6x off the roofline to on it, and D >= 256 is
untouched by construction -- the rule returns 256 there, so it is the identical
launch, not a re-tuned one.

### End to end, the shape this targets

`--batch-size 8 --seq-len 128 --d-model 32 --heads 4 --ffn-dim 32 --layers 6
--causal` (head_dim 8, 2 of the 14 appendix shapes), where add+LayerNorm was
37.6% of the forward:

| | median | speedup vs baseline |
|---|---|---|
| flat 256 | 0.3174 ms | 17.130x |
| width-scaled | **0.2693 ms** | **20.298x** |

**1.179x on the whole forward.** Accuracy is not merely close but identical --
`max_abs=0.0011037`, same digits both ways, because at D=32 the extra 224
threads were only ever summing zeros into the reduction tree.

Control on a shape the rule leaves at 256 (`b64 s128 d256 ffn256 causal`):
3.078x vs 3.068x, i.e. a tie inside noise, as it must be.

### Why the cap and the floor are where they are

**Capped at 256, not raised to D.** At D >= 512 more threads is not better --
t512 and t1024 are inside noise of t256 at every width, and t1024 is 4x worse at
D=32. Striding is already what the loops do.

**Floored at one warp, not at D.** A partial warp occupies the same issue slots
as a full one, and `block_reduce_sum`'s `__shfl_down_sync` assumes all 32 lanes
are present -- a 24-thread block would fold in undefined values from the
missing lanes.

### What is left at D=32

Still **1.54x off roofline**, the only width that is not on it. One warp per
block means one *block* per row, and SM 8.6 caps residency at 16 blocks/SM, so
the kernel gets 16 warps of the 48 an SM can hold -- 33% occupancy, not enough
memory-level parallelism to saturate. Fixing that means several rows per block
(warp-per-row, 8 warps, 8 rows), which also drops the shared-memory staging and
every `__syncthreads()` for narrow rows. That is a kernel rewrite, not a launch
parameter, and it is worth at most the remaining 1.54x on 37.6% of one shape.

### The knob

`layernorm_set_block_threads(n)` forces the block size, `0` restores the rule;
`LAYERNORM_BLOCK_THREADS` supplies the initial value. Same contract as
`tile_set_split_kv` and `wmma_set_causal_reverse` -- it exists so candidates are
timed interleaved in one process, per the first rule at the top of this file,
and the A/B above was run through it rather than through two builds.

## CUDA graph capture gate

See REPORT.md — "The gate is on activation volume, and it is a measured
constant" — for the sweep behind `_GRAPH_MAX_ACTIVATION`.

To re-derive it on another machine, `scripts/ab_graph.py --recommend` does the
sweep and the interpreting; README.md, Setup step 4 walks through reading its
output. It obeys both rules at the top of this file: candidates are timed
interleaved in one process, and it carries an eager-vs-eager control row so the
noise floor is measured rather than assumed.
