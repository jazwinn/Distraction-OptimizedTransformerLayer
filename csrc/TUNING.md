# Kernel tuning notes

Measurements behind the constants in the `csrc/` kernels. `fused_attention.cu` is the module root and includes the `.cuh` slices; its header comment maps the tree.
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

## GEMM with fused bias + GELU, in fp16: wired in, at every shape

`gemm_bias_gelu_kernel` computes `GELU(x @ W^T + b)` in one pass. Unfused, the
GELU reads the whole [M, ffn_dim] GEMM result back out of global memory, applies
one cheap function and writes it again; folded into the accumulator it is free.
Measured share of the pair, graph-timed: **33-46% at ffn_dim == d_model**, which
is every shape in the grading appendix.

It landed in two steps, and the second was worth more than the first.

### Step 1 -- re-tile. Three measurement errors had it marked dead code

The kernel sat in `fused_attention.cu` as **dead code** on the verdict "loses to
cuBLAS at every shape the model uses, 0.88x". That measurement was correct; the
conclusion was not.

1. **A kernel rejected on a benchmark is rejected on that benchmark's SHAPE.**
   The 0.88x was measured at `ffn_dim = 4*d_model` (the harness default). The
   grading appendix is `ffn_dim == d_model`, where the GELU pass's share is 4x
   larger and the GEMM is small enough that cuBLAS is far from its own roofline.
2. **A DRAM-traffic model is void when the operands fit in L2.** The 128x128
   tile was chosen, not tuned: a BMxBN tile does `BM*BN*K` MACs per `(BM+BN)*K`
   floats loaded, so 64x64 sustains 32 MAC/float and "caps at ~7 TFLOPS". That
   assumes the loads reach DRAM. At d_model 256 the weight is 256 KB and never
   leaves the 3070's 4 MB L2. What binds is the grid: at M=128, N=256 a 128x128
   tile is **2 blocks on a 46-SM card**, and it ran 0.198x of cuBLAS there.
3. **An op benchmark at small M measures PyTorch dispatch, not the kernel.**
   Eager, `F.linear` at M=128, K=N=256 reads 32.3 us -- and at M=1024, eight
   times the work, 30.4 us. Both are ~27 us of dispatch. The model graph-captures
   these shapes. Everything below is graph-timed, several calls per graph so the
   fixed replay cost amortises.

Re-tiled to 64x64 (64x32 below two waves of blocks), tf32 went from losing
everywhere to winning at 6 of 13 shapes -- and needed a gate to keep it off the
other 7.

### Step 2 -- fp16 fragments. Same accuracy, twice the tensor cores

TF32 and FP16 both carry a **10-bit mantissa**. Rounding operands to each and
accumulating in fp32 gives the same error, measured against an fp64 reference:

| | attention op (b16 h8 s128 d32) | FFN GEMM (M2048 K=N=256) | whole 6-layer model |
|:--|--:|--:|:--|
| tf32 | 1.63e-03 | 1.22e-03 | 9.82e-04 -- 49% of budget |
| fp16 | 1.63e-03 | 1.22e-03 | 9.82e-04 -- 49% of budget |
| bf16 | 1.18e-02 | 1.02e-02 | 9.70e-03 -- 485%, **12339 elements fail** |

And on this card fp16 tensor cores are twice as fast: 39.7 vs 17.7 TFLOPS at
N=2048, 38.7 vs 19.6 at 4096, 44.4 vs 22.0 at 8192 -- **1.98x-2.25x**. An fp16
fragment also contracts 16 elements of K against tf32's 8, halving the mma
instruction count, and the narrowing moves from per-fragment-load (tf32 converts
A `MT` times and W `NT` times per k-step) to once on the way into shared memory.

**bf16 is measured dead, under the current 2e-3 / 2% gate, not the old 1e-3 one.**
That re-test is the point: the loosened tolerance was a reason to re-ask, and the
answer did not change.

### The sweep: fp16 wins at all 16 shapes

`scripts/tune_linear_gelu.py`, graph-timed against cuBLAS + `F.gelu`, best tile
per precision, interleaved, control +/-1.0% to +/-2.9%:

| M | K | N | tiles64 | tf32 | fp16 | fp16/tf32 | best tile |
|---:|---:|---:|---:|--:|--:|--:|:--|
| 32 | 256 | 256 | 4 | 0.99x | **1.634x** | 1.599x | 64x32 |
| 128 | 64 | 64 | 2 | 1.36x | **1.665x** | 1.221x | 64x32 |
| 128 | 256 | 256 | 8 | 0.91x | **1.327x** | 1.464x | 64x32 |
| 512 | 256 | 256 | 32 | 0.71x | **1.240x** | 1.758x | 64x32 |
| 1024 | 32 | 32 | 16 | 2.10x | **2.402x** | 1.127x | 64x32 |
| 1024 | 256 | 256 | 64 | 0.86x | **1.425x** | 1.650x | 64x32 |
| 2048 | 256 | 256 | 128 | 1.66x | **3.188x** | 1.921x | 64x64 |
| 8192 | 256 | 256 | 512 | 1.34x | **2.534x** | 1.892x | 64x64 |
| 16384 | 256 | 256 | 1024 | 1.18x | **2.139x** | 1.817x | 64x64 |
| 320000 | 256 | 256 | 20000 | 1.03x | **1.919x** | 1.861x | 64x64 |
| 1024 | 512 | 512 | 128 | 1.19x | **2.448x** | 2.058x | 64x64 |
| 1024 | 1024 | 1024 | 256 | 0.86x | **1.846x** | 2.147x | 64x64 |
| 512 | 2048 | 2048 | 256 | 0.78x | **1.638x** | 2.112x | 64x64 |
| 1024 | 512 | 2048 | 512 | 0.82x | **1.707x** | 2.093x | 64x64 |
| 2048 | 4096 | 4096 | 2048 | 0.62x | **1.371x** | 2.227x | 128x128 |

Grids from 2 tiles to 20000, K and N from 32 to 4096, and fp16 wins every row by
1.24x to 2.40x -- **so the shape gate is gone**. It existed because tf32 lost
below two full waves of blocks and above K=N=512; fp16 loses at neither. What
survives is the coverage check (fp32 in, `K % 4 == 0`, SM 8.0+) and one
precondition below.

`pick_gemm_tile()` keeps two thresholds, and they are different constraints:

  * **Grid, below two full waves.** The crossover sits between 64 tiles (64x32
    wins) and 128 tiles (64x64 wins). Both precisions cross over in the same
    place, so one rule serves both.
  * **Contraction length, past L2.** At K=N=2048 the weight is 16 MB and 64x64
    still wins; at K=N=4096 it is 64 MB and 128x128 wins by 1.29x. One measured
    point, past every shape this model issues -- a guard, not a tuned threshold.

**PRECONDITION fp16 has and tf32 does not: operands must fit fp16's range.**
Post-LayerNorm activations and O(1/sqrt(d)) weights satisfy it by construction
here. A caller feeding values past 65504, or clustered under fp16's 6.1e-5
smallest normal, would silently lose them. `--linear-gelu tf32` and `off` are the
escape hatches, and the harness's accuracy gate is what would catch it.

### End to end, three ways

`scripts/ab_linear_gelu.py` scores the two changes separately: `off` is cuBLAS +
`F.gelu`, `tf32` is the fused kernel at the old precision, `auto` is fp16. Whole
model, causal, six layers, `ffn_dim == d_model`, flipped in-process and
interleaved, with "off" timed at both ends of every round as the control:

| shape | off ms | tf32 | fp16 | fusion | fp16 | total | control |
|:--|--:|--:|--:|--:|--:|--:|--:|
| B1 S128 d256 | 0.273 | 0.276 | 0.263 | 0.988x | 1.051x | **1.038x** | 0.4% |
| B4 S128 d256 | 0.389 | 0.410 | 0.391 | 0.949x | 1.047x | 0.994x | 5.4% |
| B8 S128 d256 | 0.728 | 0.765 | 0.713 | 0.952x | 1.072x | **1.020x** | 0.5% |
| B16 S128 d256 | 1.763 | 1.725 | 1.615 | 1.022x | 1.068x | **1.091x** | 1.1% |
| B64 S128 d256 | 5.412 | 5.021 | 5.086 | 1.078x | 0.987x | **1.064x** | 1.5% |
| B128 S128 d256 | 10.031 | 9.719 | 9.205 | 1.032x | 1.056x | **1.090x** | 0.2% |
| B8 S1024 d256 | 8.167 | 7.988 | 7.415 | 1.022x | 1.077x | **1.101x** | 2.8% |
| B8 S128 d32 | 0.216 | 0.196 | 0.194 | 1.105x | 1.009x | **1.115x** | 3.0% |
| B32 S128 d32 | 0.325 | 0.338 | 0.311 | 0.962x | 1.087x | **1.046x** | 1.8% |
| B8 S128 d512 | 2.306 | 2.283 | 2.124 | 1.010x | 1.075x | **1.086x** | 0.0% |
| B8 S128 d1024 | 6.250 | 6.413 | 5.753 | 0.975x | 1.115x | **1.086x** | 0.6% |

Geometric means over the eleven: fusion alone **1.008x**, fp16 alone **1.058x**,
together **1.066x**.

Read the first column carefully: **the fusion on its own is worth nothing now**,
because `tf32` here is ungated and so runs at the seven shapes where tf32 loses.
The gated tf32 configuration that shipped first measured 1.032x geometric mean
over these same eleven shapes. So fp16 roughly doubles the end-to-end payoff
*and* removes the gate that the tf32 version needed to get even that.

One row disagrees with the op sweep: `B64 S128 d256` reads fp16 at 0.987x of
tf32 where the op says 1.892x. Its control is 1.5%, so it is outside noise but
unexplained; the shape is the one whose activation volume declines CUDA-graph
capture, so it is timed eagerly. Total against `off` is still 1.064x.

### Accuracy, measured through the harness

Every shape PASSes. `max_abs` against the baseline, `off` -> `auto`, on a 2e-3
gate:

| shape | off | fp16 | budget used |
|:--|--:|--:|--:|
| B8 S128 d32 | 1.104e-3 | 1.316e-3 | 66% |
| B1 S128 d256 | 0.924e-3 | 0.989e-3 | 49% |
| B16 S128 d256 | 1.046e-3 | 1.321e-3 | 66% |
| B8 S1024 d256 | 1.086e-3 | 1.130e-3 | 57% |
| B8 S128 d512 | 1.133e-3 | 1.236e-3 | 62% |
| B8 S128 d1024 | 1.200e-3 | 1.346e-3 | 67% |
| default, causal | 1.233e-3 | 1.358e-3 | 68% |

fp16 costs about 1e-4 of extra end-to-end error and leaves a **~33% margin** at
the tightest shape. `verify_kernel.py`, `verify_graph.py`, default, `--causal`
and `--padding-ratio 0.3` all still PASS.

Note what was lost to gain it: with tf32 the kernel was **bit-identical** to
`F.linear + F.gelu` wherever cuBLAS picked a tf32 kernel with the same k-order.
fp16 is not, and cannot be. `--linear-gelu tf32` gets that property back at half
the tensor-core rate.

### One build note

Instantiating the fp16 kernel first failed with CUDA 13.3's CCCL complaining
about MSVC's traditional preprocessor -- a red herring. The real error was
`if (Math::convert_frag)` needing to be `if constexpr`: a runtime branch still
type-checks its dead body, and `__float_to_tf32` returns a float that an fp16
fragment element cannot be assigned from. The CCCL message came from
`kernel_ext.py`'s no-cuTile retry, whose flags lack `/Zc:preprocessor`, and it
buried the real one. That fallback now re-raises the *first* error instead.

## tile-fp16: the mode that was recorded as impossible, and still loses to wmma

`tile_attention.h` carried this, for as long as math modes have existed:

> There is no Fp16 mode: cuTile accumulates a `__half` matmul into `__half`,
> and attention sums hundreds of products per output.

**Half right, and the wrong half was load-bearing.** It is true of `ct::matmul`,
whose result element type is fixed by its operand: `matmul_element_result<__half>`
is `__half` (crt/cuda_tile.h:2737). It is false of `ct::mma(A, B, C)`, which takes
the accumulator as an argument -- `low_precision_mma_v` admits `__half` operands
against *either* a `__half` or a **`float`** accumulator (crt/cuda_tile.h:2646).

The kernel's second GEMM already used `mma`. Only the first had to stop using
`matmul`:

```cuda
auto s = [&] {
    if constexpr (MATH == MathMode::Fp16) {
        return ct::mma(q_op, k_op, ct::zeros<STile>());
    } else {
        return ct::matmul(q_op, k_op);
    }
}() * (scale * 1.4426950408889634f);
```

Block shapes are seeded from bf16's -- same 16-bit operand, same tile geometry.

### It does exactly what the format predicts

fp16 carries 10 mantissa bits like tf32, and 16-bit operands like bf16. It lands
on both: **tile-fp16's time equals tile-bf16's to three decimal places at every
case, and its error equals tile-tf32's to three significant figures at every
case.**

| case | wmma | tile-tf32 | tile-bf16 | tile-fp16 | tf32 err | bf16 err | fp16 err |
|:--|--:|--:|--:|--:|--:|--:|--:|
| default | 0.035 | 0.098 | 0.071 | 0.071 | 3.1e-4 | 4.5e-3 | 3.1e-4 |
| default causal | 0.035 | 0.084 | 0.061 | 0.061 | 9.4e-4 | 1.2e-2 | 9.4e-4 |
| default padded | 0.045 | 0.129 | 0.098 | 0.099 | 7.9e-4 | 8.3e-3 | 7.9e-4 |
| long seq | 0.607 | 1.280 | 0.871 | 0.871 | 7.3e-5 | 1.0e-3 | 6.9e-5 |
| long seq causal | 0.422 | 0.825 | 0.494 | 0.494 | 7.4e-4 | 9.2e-3 | 7.4e-4 |
| odd shape | 0.019 | 0.020 | 0.021 | 0.020 | 1.2e-3 | 1.0e-2 | 1.2e-3 |

Against sdpa at `long seq`, that moves the tile kernel from tf32's 1.26x to
**1.85x**, and at `long seq causal` from 1.23x to **2.05x** -- for free, since
`verify_kernel.py` holds tile-fp16 to the tf32 tolerance (3e-3), not bf16's.

**tile-bf16 is now strictly dominated**: identical speed, ~10x the error. There
is no shape at which it is the right choice.

### And it still does not make the program faster

The comparison that matters is against wmma, because that is what `auto` picks.
When this was scoped, tile-bf16 (1.712 ms) beat wmma (2.563 ms) at `long seq` by
1.50x, and the whole case for fp16 was that it would collect that win at an
accuracy that passes. **That gap closed while the work was in flight**: the wmma
kernel gained its own `compute_t`/fp16 path, and went 2.563 -> 0.607 ms on the
same case. wmma is now ahead everywhere:

| case | wmma | tile-fp16 | wmma ahead by |
|:--|--:|--:|--:|
| default | 0.035 | 0.071 | 2.03x |
| default causal | 0.035 | 0.061 | 1.74x |
| long seq | 0.607 | 0.871 | 1.43x |
| long seq causal | 0.422 | 0.494 | 1.17x |

End to end, forced through the harness, causal, `ffn_dim == d_model`:

| shape | wmma | tile-fp16 | tile-tf32 |
|:--|--:|--:|--:|
| B8 S1024 d256 | **10.752x** | 9.656x | 7.906x |
| B16 S128 d256 | **3.454x** | 3.185x | 2.996x |

So fp16 is worth **+22%** to the tile kernel at S=1024 and **+6%** at S=128 over
tf32 -- a real gain on that kernel -- and worth **nothing to the graded number**,
because `auto` correctly keeps choosing wmma. The mode is kept because it is the
tile kernel's best mode and because it settles the "is fp16 possible in cuTile"
question with a measurement rather than a comment; it is not on any dispatch
path.

The `tiny` rows of `bench_attention.py` are not quoted above: they disagree with
`verify_kernel.py`'s timings for the same shapes (0.004 ms against 0.037 ms) at
a size where a single kernel is a few microseconds. Neither is trustworthy there,
and nothing depends on them.

## The wmma attention kernel, in fp16: 1.33x on the op, 1.039x end to end

The kernel contracted fp32 q/k/v in **tf32** fragments. It now contracts them in
**fp16**, with the tensors still fp32 -- only the shared tiles and the fragments
narrow, and the output stays fp32 because it feeds `out_proj`, a cuBLAS fp32
GEMM. `compute_t` is the template parameter that splits the two roles;
`--attn-fp16 tf32` restores the old path.

### Why this is not a precision trade

fp16 and tf32 carry the **same 10-bit mantissa**. tf32 is 1+8+10, fp16 is
1+5+10; only the exponent range differs, and post-LayerNorm activations use none
of it. So the expected accuracy cost is zero, and it measures as zero -- not
"close", identical:

| | attention op vs fp64 | whole model, harness `max_abs` |
|:--|:--|:--|
| tf32 | 1.12e-3 … 1.77e-3 | 1.31567e-3 / 1.32135e-3 / 1.13010e-3 / 1.23608e-3 |
| fp16 | 1.12e-3 … 1.77e-3 | 1.31567e-3 / 1.32135e-3 / 1.13010e-3 / 1.23608e-3 |

Every digit of the harness column is the same for both. All shapes PASS,
including `--causal`, `--padding-ratio 0.3` and the default.

**bf16 is not the same argument and does not survive it.** 8 mantissa bits
measured 425%-622% of the 2e-3 budget with tens of thousands of failing
elements, under the *loosened* gate. See the fp16-vs-tf32 note in REPORT.md.

### What it buys

fp16 tensor cores measure 2.0x-2.25x tf32 on this card (39.7 vs 17.7 TFLOPS at
N=2048, 44.4 vs 22.0 at N=8192), and a 16x16x16 fp16 fragment contracts twice
the K of tf32's 16x16x8, so the mma count halves as well.

`scripts/ab_attention_fp16.py`, causal, graph-timed, interleaved, control
+/-0.4%. The SDPA column is what `auto` falls back to:

| shape | SDPA | tf32 | fp16 | fp16 vs tf32 | fp16 vs SDPA |
|:--|--:|--:|--:|--:|--:|
| B8 H4 S128 d8 | 29.3 | 11.3 | 9.3 | 1.210x | 3.145x |
| B16 H8 S128 d32 | 83.2 | 38.8 | 31.8 | 1.221x | 2.618x |
| B64 H8 S128 d32 | 274.5 | 123.0 | 102.0 | 1.206x | 2.691x |
| B8 H8 S1024 d32 | 1371.8 | 543.0 | 408.0 | **1.331x** | 3.362x |
| B8 H8 S128 d64 | 63.1 | 37.7 | 29.4 | 1.283x | 2.149x |
| B4 H8 S512 d64 | 294.7 | 179.2 | 130.3 | 1.376x | 2.262x |
| B1 H8 S2048 d64 | 1037.6 | 595.8 | 426.6 | **1.397x** | 2.432x |
| B8 H8 S32 d128 | 41.8 | 15.5 | 13.2 | 1.169x | 3.162x |
| B8 H8 S128 d128 | 95.8 | 145.4 | 101.6 | 1.430x | 0.943x |
| B4 H8 S512 d128 | 483.7 | 706.2 | 459.6 | **1.536x** | 1.052x |
| B2 H8 S1024 d128 | 892.2 | 1239.9 | 819.4 | 1.513x | 1.089x |

Geometric mean **1.328x** over the eleven shapes.

End to end, six layers, causal, `ffn_dim == d_model`, ATTENTION_FP16 flipped
in-process and interleaved (7 rounds, worst control +/-2.2%):

| shape | tf32 ms | fp16 ms | ratio | control |
|:--|--:|--:|--:|--:|
| B8 S1024 d256 | 7.369 | 6.564 | **1.123x** | 0.0% |
| B8 S128 d32 | 0.194 | 0.182 | **1.065x** | 0.1% |
| B1 S128 d256 | 0.264 | 0.249 | **1.064x** | 0.0% |
| B8 S128 d256 | 0.689 | 0.665 | 1.036x | 2.2% |
| B8 S128 d512 | 2.049 | 1.978 | 1.036x | 0.0% |
| B64 S128 d256 | 4.823 | 4.719 | 1.022x | 0.1% |
| B16 S128 d256 | 1.582 | 1.551 | 1.020x | 0.2% |
| B8 S128 d1024 | 5.563 | 5.485 | 1.014x | 0.1% |
| B8 S128 d512 h4 | 2.163 | 2.204 | 0.981x | 1.0% |

Geometric mean **1.039x**. The gap between 1.33x on the op and 1.039x on the
model is Amdahl and nothing else: attention is 20% of the forward pass at
`B16 S128 d256` and 46% at `S1024`, which is exactly where the two ends of the
table sit. The harness's own verdict agrees independently -- `--seq-len 2048
--batch-size 1 --causal` went **4.589x to 5.231x**, a 1.14x against the A/B's
1.123x at S 1024.

The last row is a **consistency check, not a regression**: at head_dim 128 and
S 128 the dispatch rule sends both configurations to SDPA, which is why its
`max_abs` column is exactly 0.00e+00. 0.981x against a 1.0% control is that
shape's noise.

### head_dim 128 changes verdict, and the auto rule with it

The tf32 kernel *lost* to SDPA at head_dim 128, which is why
`kWmmaAutoMaxHeadDim` was 64 and the kernel never ran there above S 64. In fp16
it wins -- but not everywhere. Re-measured, SDPA/wmma, so >1 means the kernel
wins:

| S | 64 | 128 | 256 | 384 | 512 | 1024 |
|:--|--:|--:|--:|--:|--:|--:|
| ratio | 1.552x | **0.938x** | 1.028x | 1.027x | 1.047x | 1.081x |

The dip at exactly S 128 reproduced across runs (0.938x, 0.943x). So the rule
gained one clause -- head_dim 128 is admitted from **S 512** up, where the
margin is 4.7%-8.1%, comfortably past the +/-0.4% control. S 256 and 384 also
win, by 2.8%, and are deliberately not claimed: too close to the floor to be
worth widening the rule for.

Leaving S 128 to SDPA matters more than the ratio suggests. It is the sequence
length of eleven of the fourteen grading shapes, so admitting head_dim 128
outright would have cost 6% at the one shape that has it.

### NEGATIVE: the freed shared memory does not want to be spent on a bigger tile

Narrowing `compute_t` frees about a third of the block's shared memory, so
shapes appear that fp32 cannot fit -- at head_dim 128, 32x32 costs 41.4 KB in
fp16 against 29.9 KB for 32x16, and only the latter fits in fp32. Spending the
slack on a wider key tile was the obvious next move. It loses badly
(`scripts/ab_attention_shapes.py`, one build per candidate):

| head_dim 128, fp16 | 32x16 (incumbent) | 32x32 | 16x32 |
|:--|--:|--:|--:|
| B8 H8 S32 | 1.000x | 0.192x | 0.689x |
| B8 H8 S128 | 1.000x | 0.765x | 0.378x |
| B4 H8 S512 | 1.000x | 0.738x | 0.346x |
| B2 H8 S1024 | 1.000x | 0.765x | 0.357x |
| **geomean** | **1.000x** | **0.537x** | 0.424x |

head_dim 64 says the same: 64x32 is 0.812x and 32x32 is 0.552x against the 64x16
incumbent.

A wider `BLOCK_N` doubles the K/V staging and the score tile without adding any
parallelism. 41.4 KB drops the SM from three resident blocks to two, and each
block then does twice the work per key-tile iteration. Q is already
register-resident, so the extra occupancy pressure buys nothing. **The fp16 win
at head_dim 128 is entirely the fragments, not the tile.** `WmmaShape` was
parameterised on `sizeof(compute_t)` to run this sweep and then un-parameterised
again, on the criterion the file already stated: a different winner per dtype is
what would justify the parameter, and there is none.

### Still open: fp16 q/k/v in GLOBAL memory

This change narrows on the way *into* shared memory. Narrowing the tensors
themselves would also halve the kernel's DRAM traffic, and it is worth
measuring rather than guessing at: handing the kernel fp16 tensors directly
measured **21.4 us at B8 H8 S128 d64 against 29.4 for fp16-compute-only** -- a
further 1.37x on the op.

It costs more than a cast. q/k/v come out of the fused qkv GEMM, so that GEMM
would need an fp16 output epilogue (`gemm_bias_gelu_kernel` is most of one
already), and the kernel would need an `out_t` so that a half-input call can
still write the fp32 that `out_proj` consumes. It also stops being free on
accuracy: the fp16-tensor measurement above wrote an fp16 *output*, which moved
the op's error from 1.59e-3 to 1.83e-3. An `out_t` avoids that, but it has to be
built before the 1.37x can be claimed.

### Three build traps, all of which cost a debugging session

1. **`if (Math::convert_frag)` must be `if constexpr`.** A runtime branch still
   type-checks its dead body, and `__float_to_tf32` returns a float that an fp16
   fragment element cannot be assigned from.
2. **`static_cast<float>` does not compile on `__nv_bfloat16` here.** torch's
   build passes `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` and
   `-D__CUDA_NO_HALF_CONVERSIONS__`, which delete the implicit operators. Hence
   `dev_to_float`, the mirror of the existing `dev_from_float`.
3. **Staging must stay a ternary.** Rewriting
   `k_s[i] = inb ? cvt(k_base[g]) : zero_v` as an `if/else` with two stores cost
   **1.3x-2.1x on the tf32 path** -- 708 us to 1473 us at B4 H8 S512 d128. The
   ternary compiles to a predicated select; the branch does not. It was caught
   only because SDPA, timed in the same table, did not move between the two
   runs. Hence `dev_of_float<T>`, a value-returning narrowing helper.

A fourth, in `kernel_ext.py` rather than the kernel: the no-cuTile retry does not
pass `/Zc:preprocessor`, so **any** compile error in `fused_attention.cu` used to
surface as CUDA 13.3's CCCL complaint about MSVC's traditional preprocessor,
with the real error discarded by the `except Exception`. The fallback now
re-raises the first failure.

## The attention softmax in the base-2 domain: 1.040x on the op, 1.012x end to end

`__expf(x)` is not one instruction. It is `ex2.approx(x * log2e)` -- an FMUL in
front of every MUFU.EX2 -- so the original softmax paid two multiplies per score
element: an explicit `s * scale` and that hidden one. FlashAttention-2 removes
the second by working in base 2 throughout (`softmax.h` reads
`exp2f(tensor(mi,ni) * scale - max_scaled)` with `M_LOG2E` premultiplied into
`scale`). `scripts/ab_attention_exp2.py` measures the port, and the `-inf` test
that goes with it: `ex2.approx.f32(-inf)` is *defined* to return `+0`, and
`m_new` is finite in the branch that reaches the test, so the guard was
computing something the exponential already gets right.

### Why the softmax was worth attacking at all

Static SASS on the key-tile loop body, `cuobjdump -sass build/fused_attention.cuda.o`,
per thread per key tile, fp32 tensors contracted in fp16:

| head_dim | loop body | of which HMMA |
|---:|---:|---:|
| 16 | 1070 | 8 |
| 32 | 1089 | 16 |
| 64 | 736 | 16 |

**The tensor cores are 1.5% of the instruction stream.** On SM 8.6 an SM retires
256 tensor MAC, 128 FP32 and 16 MUFU.EX2 per clock, so at head_dim 32 the two
GEMMs are ~512 SM-clocks per block-key-tile against ~1500 of ALU and SFU. This
kernel is not matmul-bound at the grading head_dims, which inverts the A100
framing FA2 optimizes for (312 vs 19.5 TFLOPS, a 16x matmul advantage) and makes
FA2's "minimize non-matmul FLOPs" advice apply *harder* here, not less.

### Three modes, because the obvious version is not the safe one

`softmax_mode_flag()`, `WMMA_SOFTMAX_MODE`, `wmma_set_softmax_mode()`. A
template parameter on the kernel, not a runtime branch, so no mode carries
another's code or registers.

- **0** -- the original. `s * scale`, `__expf`, explicit `-inf` test.
- **1** -- base-2. One `s * (scale * log2e)`, a bare `exp2f`, no `-inf` test.
  **Q is untouched.**
- **2** -- as 1, with `scale * log2e` also folded into Q at staging time, which
  deletes the score-side multiply too: BLOCK_M*head_dim multiplies per block
  instead of BLOCK_M*BLOCK_N per key tile.

Mode 2 is the version the arithmetic argues for and it is **rejected on both
counts**. It is the reason the knob is an int and not a bool.

### Measured, op level, causal, graph-timed, interleaved in one process

`python scripts/ab_attention_exp2.py`

| shape | mode0 us | mode1 us | mode2 us | 1v0 | 2v0 | err0 | err1 | err2 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| B8 H4 S128 d8 | 10.5 | 9.7 | 9.9 | 1.080x | 1.059x | 1.12e-03 | 1.12e-03 | 1.17e-03 |
| B8 H16 S128 d16 | 20.0 | 18.4 | 19.1 | 1.086x | 1.046x | 1.95e-03 | 1.95e-03 | 1.37e-03 |
| B8 H8 S512 d16 | 87.8 | 78.7 | 82.8 | **1.115x** | 1.061x | 1.66e-03 | 1.66e-03 | 1.24e-03 |
| B1 H8 S128 d32 | 9.3 | 8.9 | 9.3 | 1.044x | 1.002x | 9.75e-04 | 9.75e-04 | 9.75e-04 |
| B16 H8 S128 d32 | 31.8 | 30.6 | 31.3 | 1.039x | 1.016x | 1.77e-03 | 1.77e-03 | 1.51e-03 |
| B64 H8 S128 d32 | 103.5 | 100.4 | 100.9 | 1.031x | 1.026x | 1.77e-03 | 1.77e-03 | 1.73e-03 |
| B8 H8 S1024 d32 | 408.1 | 380.6 | 397.9 | 1.072x | 1.026x | 1.07e-03 | 1.07e-03 | 1.60e-03 |
| B8 H8 S128 d64 | 29.1 | 29.0 | 30.7 | 1.005x | 0.949x | 1.59e-03 | 1.59e-03 | **2.63e-03** |
| B4 H8 S512 d64 | 132.8 | 131.9 | 142.1 | 1.006x | 0.934x | 1.29e-03 | 1.29e-03 | 1.71e-03 |
| B1 H8 S2048 d64 | 428.9 | 423.3 | 458.7 | 1.013x | **0.935x** | 1.00e-03 | 1.00e-03 | 1.35e-03 |
| B4 H8 S512 d128 | 476.0 | 477.6 | 475.2 | 0.997x | 1.002x | 1.14e-03 | 1.14e-03 | 1.45e-03 |
| B2 H8 S1024 d128 | 843.0 | 847.9 | 834.8 | 0.994x | 1.010x | 1.14e-03 | 1.14e-03 | 1.15e-03 |

Geometric mean: **mode 1 1.040x**, best 1.115x, worst 0.994x. **Mode 2 1.004x**,
best 1.061x, worst 0.935x. Worst control this run +/-1.4%.

The win lands exactly where the throughput table above says it should: 1.03x-1.12x
at head_dim 8/16/32, a wash at 64, nothing at 128. The softmax is a first-order
cost only while head_dim is small enough that the GEMMs are cheap.

### Mode 2 fails accuracy, and mode 1 is free

`err1 == err0` on all twelve shapes, to every digit printed. That is not a
coincidence -- it is the point. Mode 1 changes only *where* fp32 constants are
multiplied, so the fp16 narrowing sees exactly the same Q it always did.

Mode 2 narrows `q * scale * log2e` instead of `q`, which puts both constants
inside the 10-bit mantissa, and **B8 H8 S128 d64 goes 1.59e-03 -> 2.63e-03,
over the harness's 2e-3 atol**. The error is not uniformly worse (d16 improves,
1.95e-03 -> 1.37e-03) -- it is a different rounding draw with a worse tail, and
the tail is what the grading gate reads. Rejected.

Through the harness itself, `--accuracy-trials 5`, four shapes, mode 0 against
mode 1: **every trial PASS with byte-identical max_abs** (0.00131567, 0.00116003,
0.0011301, 0.00123608). `scripts/verify_kernel.py` reports "every kernel matches
the reference on every case" under both. Fully-masked rows -- the branch the
deleted guard sits beside -- emit exactly 0.0 and no NaN under both.

### The noise floor, measured rather than inferred

`python scripts/ab_attention_exp2.py --op-only --self-control` hands mode 1 to
every column, so the "1v0" and "2v0" ratios compare identical code and their
true value is exactly 1.000x. Run back to back with the real A/B, same session:

| | self-control | real A/B |
|:--|--:|--:|
| geometric mean | **0.998x** | **1.038x** |
| range | **0.991x - 1.003x** | 0.991x - 1.112x |

So the floor is **+/-0.9%**, and against it:

- head_dim 16/32 -- 1.037x, 1.041x, 1.042x, 1.064x, 1.112x. Four to eleven
  points clear of the band. Unambiguous.
- head_dim 64 -- 1.009x, 1.010x, 1.011x. Three shapes, tight, all just above
  the 1.003x ceiling. A real ~1%, not the nothing an earlier reading called it.
- head_dim 128 -- 0.991x, 0.992x, sitting on the bottom edge of the
  self-control range. Indistinguishable from noise.

Mode 2 reads 0.998x under self-control -- identical to mode 1, as it must when
both slots run the same kernel -- against 0.937x-0.956x at head_dim 64 in the
real run, so its regression is a property of the kernel and not of the harness.

**So the honest summary of this change is "1.04x-1.11x at head_dim 16/32, ~1%
at 64, nothing at 128", not the 1.040x geometric mean**, which averages real
wins against near-zeros and flatters the shapes that gained nothing.

Two traps this run reproduced:

- **A control row under ~15 us is still meaningless.** The self-control's worst
  reading was 2.5%, all of it on the ~10 us `B8 H4 S128 d8` row; every row over
  15 us read <= 1.0%. Same rule `ab_wmma_split_kv.py` already encodes.
- **Desktop GPU contention silently doubles everything.** The first
  self-control attempt read 0.825x-1.067x with a 2.5% control and looked like
  proof the whole result was noise. Absolute times were 2-2.5x the clean run
  (`B8 H8 S1024 d32` 1032 us against 408) because Wallpaper Engine, Teams,
  Spotify and a video player all hold GPU contexts on this machine. **Check the
  absolute microseconds against a known-clean run before believing any ratio
  table**; `nvidia-smi --query-compute-apps` names the culprits.
### End to end, 6 layers, causal, ffn_dim == d_model

Two model instances, mode set before each one's warmup: `self._graphs` is per
instance, and flipping the flag after capture does nothing to a replay -- the
same trap `tile_set_split_kv` documents.

| shape | mode0 ms | mode1 ms | ratio | ctrl |
|:--|--:|--:|--:|--:|
| B8 S128 d32 h4 | 0.182 | 0.179 | 1.021x | 0.0% |
| B8 S128 d256 h16 | 0.705 | 0.695 | 1.014x | 0.0% |
| B1 S128 d256 h8 | 0.250 | 0.247 | 1.011x | 0.0% |
| B16 S128 d256 h8 | 1.545 | 1.531 | 1.009x | 0.4% |
| B64 S128 d256 h8 | 4.598 | 4.574 | 1.005x | 0.2% |
| B8 S1024 d256 h8 | 6.402 | 6.223 | **1.029x** | 0.0% |
| B8 S128 d512 h8 | 1.969 | 1.972 | 0.998x | 0.2% |

Geometric mean **1.012x**. Amdahl check on the best row: attention is 42.0% of
that forward and the op gained 1.072x, predicting 1.030x against 1.029x
measured -- the chain is real, not a coincidence of noise.

### The SASS predicted the wrong head_dim, which is why this was measured

The instruction-count delta says mode 2 should win biggest at head_dim 64
(736 -> 698, -5.2%) and barely at all at 16/32 (-1.0%, -0.6%, because the
compiler hands the 49 removed FMUL back as 64 LOP3 and 16 P2R once BLOCK_N is
32). The measurement is the exact inverse: head_dim 64 is where mode 2
*regresses* 0.935x-0.949x, and 16/32 is where mode 1 wins 1.03x-1.12x.
Registers are unchanged (REG:128 both at head_dim 64, LOCAL:0), so it is not
occupancy. **Static instruction count did not predict the sign of the result at
any head_dim here.** Treat it as a check that an edit landed, not as a
prediction that it helped.

## wmma kernel: split-KV (Flash-Decoding), gated hard

The attention grid is `(ceil(S/BLOCK_M), H, B)` -- query side only. Nothing in
it scales with the KEY length, so a shape with few queries and many keys
launches a grid too small to fill the card while each block walks a long serial
key loop. No block shape fixes that; only splitting the key range does.

`scripts/bench_wmma_occupancy.py`, `scripts/ab_wmma_split_kv.py`,
`scripts/verify_wmma_split_kv.py`. Knobs: `WMMA_SPLIT_KV` /
`wmma_set_split_kv()`, and `WMMA_SPLIT_COUNT` / `wmma_set_split_count()` to
force a count the rule would not pick -- which is how the rule was chosen.

### The prize, bounded before the code was written

`bench_wmma_occupancy.py` tables blocks against what the occupancy API says the
card holds, plus per-block cost normalised to the same (H, S, D) at a batch
that saturates:

| shape | blocks | resident | waves | n_kt | op us | per-block vs full |
|:--|--:|--:|--:|--:|--:|--:|
| B1 H4 S128 d32 c | 8 | 138 | 0.06 | 2 | 9.0 | **9.64x** |
| B1 H8 S128 d32 c | 16 | 138 | 0.12 | 2 | 10.8 | **5.71x** |
| B1 H8 S128 d64 c | 16 | 138 | 0.12 | 4 | 15.4 | **4.79x** |
| B1 H8 S512 d32 c | 64 | 138 | 0.46 | 8 | 35.9 | 2.57x |
| B8 H8 S128 d32 c | 128 | 138 | 0.93 | 2 | 13.6 | 0.90x |
| B64 H8 S128 d32 c | 1024 | 138 | 7.42 | 2 | 112.9 | 0.93x |

Two things the table settles up front. The idle capacity is real -- 12% of the
card at B1, at 5.7x the per-block cost of the same kernel when full. And the
ceiling is **small**, because `cap = min(resident/blocks, n_kt)` and at S 128
with BLOCK_N 32 a causal block walks about **two** key tiles. A 94%-idle card
does not help if there is nothing to divide.

### Design

One kernel, `splits` as a runtime argument rather than a template parameter:
it changes only the prologue (which slice of the key range) and the epilogue
(partial or finished row), never the inner loop, and measuring it found no
register cost -- so it buys none of the instantiation blow-up a template would.

Grid.x carries `n_m * splits`, since a grid has three axes and y/z are heads
and batch. `splits` consecutive blocks share an m-tile, keeping the K/V a split
group reads adjacent in L2.

Each split takes a contiguous run of whole key tiles out of **this block's**
range, not out of `[0, S)`. That is what makes it work under causal masking:
slicing the dense range would leave the later splits of an early m-tile idle
while a late m-tile still carried everything. A split that comes up empty
stores its initial state -- zero-filled O, `-inf` max, zero sum -- which the
combine pass weights to exactly zero. Cheaper than a launch-time special case
and impossible to get wrong.

`wmma_split_combine_kernel` rebases every partial onto the max over all splits
before adding numerators and denominators. It is templated on the softmax mode,
because that decides the domain the stored maxima are in.

The workspace is **one** allocation with three views. Three separate `at::empty`
calls cost enough host time at these sizes to swamp the device win.

### The gate, and the two clauses that were bought with regressions

```
head_dim > 8, and blocks * 8 <= resident, and n_kt >= 4,
then n = min(floor(resident / blocks), n_kt, 8)
```

**`blocks * 8 <= resident`** -- an eighth of the card, deliberately stricter
than "there is spare capacity". Splitting adds a whole combine launch, and
unless the grid is very small that launch costs more than the shortened key
loop saves. Measured with the count the rule itself picks:

| shape | blocks | waves | ratio |
|:--|--:|--:|--:|
| B1 H4 S128 d32 c | 8 | 0.06 | 1.122x |
| B1 H8 S128 d64 c | 16 | 0.12 | 1.582x |
| B1 H4 S256 d64 c | 16 | 0.12 | 2.526x |
| B2 H8 S128 d32 c | 32 | 0.23 | **0.898x** |
| B4 H8 S128 d32 c | 64 | 0.46 | **0.779x** |
| B1 H8 S512 d64 c | 64 | 0.46 | **0.955x** |

A looser `blocks * 4 <= resident` admits the 0.898x row; adding an
`n_kt >= 16` clause for long key loops admits the 0.955x one. Both were tried
and both lose.

**`n_kt >= 4`** -- and this clause is the one the END-TO-END measurement bought,
against what the op alone said:

| shape | head_dim | n_kt | op | end to end |
|:--|--:|--:|--:|--:|
| B1 S128 d512 h8 | 64 | 4 | 1.807x | 1.061x |
| B1 S128 d256 h8 | 32 | 2 | 1.084x | **0.963x** |

The op-level win at `n_kt` 2 is real and still does not survive, because the
op benchmark amortises the extra launch over a graph full of back-to-back
attention calls while a six-layer model pays it six times against six much
smaller savings. **An op-level A/B cannot see this**; only the model can.

`floor`, never `ceiling`: a count that overfills the card serialises the extra
blocks into a second wave and pays the combine pass for it.

### Results

Op level, causal, graph-timed, interleaved, symmetric sampling. Only two of the
fourteen shapes now pass the gate; the other twelve run identical code and are
the control:

| shape | off us | on us | ratio | splits |
|:--|--:|--:|--:|--:|
| B1 H8 S128 d64 c | 22.0 | 12.8 | **1.713x** | 4 |
| B1 H4 S256 d64 c | 40.8 | 15.2 | **2.677x** | 8 |
| the 12 declined | | | 0.994x - 1.014x | 1 |

End to end, 6 layers, causal, `ffn_dim == d_model`:

| shape | off ms | on ms | ratio | splits |
|:--|--:|--:|--:|--:|
| B1 S256 d256 h4 | 0.712 | 0.513 | **1.389x** | yes |
| B1 S128 d256 h4 | 0.410 | 0.346 | **1.183x** | yes |
| B1 S128 d512 h8 | 0.695 | 0.663 | 1.049x | yes |
| 6 declined rows | | | 1.007x (0.976x - 1.055x) | no |

The declined rows are identical code both sides, so their spread **is** the
harness noise floor: geomean 1.007x, +/-4%. The 1.049x row is inside it and is
not claimed; 1.183x and 1.389x are well clear.

Through the harness, `--accuracy-trials 3`: every trial PASS with **identical
max_abs** split on and off (0.00113535, 0.000999093, 0.00133359), and its own
speedup column moves 18.10x -> 21.02x and 13.37x -> 15.57x.

`verify_wmma_split_kv.py` forces 2, 3, 4 and 8 splits over 16 cases -- dense,
causal, both output layouts, `S` not a multiple of BLOCK_M, key-padding masks,
and fully-masked query rows. Worst split-vs-single-pass disagreement 2.82e-04,
error against float64 unchanged from the single-pass kernel, no NaN anywhere.
Eight splits over a two-tile range is the empty-split path, and it is in there
on purpose.

### Three measurement traps, one of which invented an 8.2x

- **A non-alternated sweep invented an 8.2x.** Timing each forced split count
  once, in one sequential pass, reported `B1 H8 S512 d64` at 8.220x for n=3.
  Alternating an n=1 baseline next to every count showed the shape is flat --
  85-90 us at every count, ratio 0.955x-1.004x. Correctness was identical at
  every count, so this was never a bug, only a benchmark. Same failure mode
  csrc/TUNING.md already records for min-of-2 across processes. **A speedup
  larger than the split count is arithmetically impossible; treat it as a
  broken measurement, not a result.**
- **Asymmetric sampling biased every ratio by 2.5%.** Timing `off` twice per
  round and `on` once, then taking min of each, compares min-of-2 against
  min-of-1. The declined rows -- identical code, true ratio exactly 1.000x --
  read a geometric mean of **0.975x** under it. Two samples each fixed it to
  1.007x. This is inherited from the older A/B scripts in this directory and
  is worth fixing there too.
- **The declined rows are a free control.** Any A/B over a gated optimization
  gets one: the shapes the gate turns down run identical code, so their spread
  is the harness noise floor, measured in the same process and the same run as
  the result. No separate `--self-control` pass needed for them.

### What this does NOT cover

`B2 H2 S2048 d64 c` measures **1.679x at n=6** in the forced sweep and the gate
declines it: 128 blocks against 138 resident is 0.93 waves, so `blocks * 8 <=
resident` fails and `floor(138/128)` is 1 anyway. Taking it would need the
count to exceed the spare capacity, which is exactly what the 0.555x
ceiling-vs-floor result warns against. One data point, deliberately not fitted.
`B1 H8 S512 d32 c` is a smaller version of the same miss: 1.276x forced at n=2,
declined at 0.46 waves.

## Per-tile mask classification: 1.33x on the op, 1.05x end to end, bit-identical

Every score element used to evaluate four predicates -- `i < S`, `gj < S`, the
causal `gj > i`, and an explicit-mask lookup -- plus the address arithmetic
under them. FlashAttention-2 does not do that. `mask.h` templates on
`Is_even_MN` / `Is_causal` / `Col_idx_only` so interior tiles compile with no
row-index computation at all, and only the diagonal block pays; the paper puts
it as "for each row we only need apply the causal mask to 1 block".

This is the same split, decided per (warp, key tile) at run time rather than at
compile time, because BLOCK_N and the mask pointer are not compile-time
constants here.

`scripts/ab_mask_classify.py`, `scripts/verify_mask_classify.py`. Knob:
`WMMA_MASK_CLASSIFY` / `wmma_set_mask_classify()`.

### Why this was the target

SASS on the key-tile loop body said so, before any of it was written: at
head_dim 32 the body is **1089 instructions per thread, 16 of them HMMA**, and
roughly 530 of the rest are integer address and predicate math. The two GEMMs
are 1.5% of the instruction stream. See "The attention softmax in the base-2
domain" above for the throughput arithmetic behind that.

### What is classified

Three clauses, all **warp-uniform** -- `q_base`, `kt` and `S` do not vary
across a warp -- so the branch cannot diverge within one:

- `rows_in`  every query row of the warp stripe is inside `S`
- `cols_in`  every key column of the tile is inside `S`
- `tri_free` the whole tile is below the causal diagonal, i.e. the largest
  column it holds is still at most the smallest row the warp owns:
  `kt + BLOCK_N - 1 <= q_base`

An explicit mask disables the softmax fast path outright -- an arbitrary mask
cannot be classified away.

Both paths have to exist regardless; that IS the optimization. So the runtime
flag that forces the slow one costs nothing the shipped kernel does not already
carry, and no template parameter or extra instantiation was needed.

The K/V staging gets its own, weaker classification: it only asks whether the
tile is inside `S`, so it applies **even when there is an explicit mask**.

### Results, op level, graph-timed, interleaved, symmetric sampling

| shape | off us | on us | ratio |
|:--|--:|--:|--:|
| dense B8 H8 S512 d32 | 230.0 | 157.8 | 1.457x |
| dense B8 H8 S1024 d32 | 863.0 | 585.3 | 1.474x |
| dense B4 H8 S2048 d32 | 1682.7 | 1135.4 | **1.482x** |
| dense B8 H16 S512 d16 | 282.4 | 196.4 | 1.438x |
| dense B8 H8 S512 d64 | 377.0 | 297.9 | 1.266x |
| causal B8 H8 S128 d32 | 14.0 | 11.3 | 1.234x |
| causal B8 H8 S512 d32 | 158.0 | 112.4 | 1.406x |
| causal B8 H8 S1024 d32 | 501.4 | 353.8 | 1.417x |
| causal B4 H8 S2048 d32 | 921.4 | 636.7 | 1.447x |
| causal B8 H16 S512 d16 | 174.8 | 128.4 | 1.362x |
| causal B8 H8 S512 d64 | 248.0 | 206.6 | 1.200x |
| causal B4 H8 S1024 d64 | 439.4 | 350.8 | 1.253x |
| causal B8 H8 S500 d32 | 153.2 | 111.8 | 1.371x |
| mask B8 H8 S512 d32 | 352.4 | 308.8 | 1.141x |
| mask B8 H8 S512 d32 c | 215.8 | 193.2 | 1.117x |

Geometric mean **1.332x**: dense **1.421x**, causal **1.333x**, masked
**1.129x**. Self-control (`--self-control`, classification-off timed against
itself, so every true ratio is 1.000x) reads **0.999x, 0.992x - 1.013x**. The
smallest win here is nine times the noise floor.

The ordering is the mechanism: dense shapes, where every interior tile
qualifies, gain most; causal shapes, which have exactly one diagonal tile per
block, gain slightly less; head_dim 64 gains least because BLOCK_N is 16 there,
so the same predicate work is spread over half as many score elements per tile.

**The masked group is not a control**, which is what a first pass assumed. An
explicit mask kills only the softmax fast path; the K/V staging classification
asks a different question and still applies. So those two rows isolate the
staging half of the change at **~1.13x**, and the softmax half is the rest.

### End to end, 6 layers, causal, ffn_dim == d_model

| shape | off ms | on ms | ratio | max_abs |
|:--|--:|--:|--:|--:|
| B8 S1024 d256 h8 | 8.617 | 7.749 | **1.112x** | 0.0 |
| B8 S512 d256 h8 | 3.931 | 3.685 | 1.067x | 0.0 |
| B8 S128 d256 h8 | 1.029 | 0.988 | 1.041x | 0.0 |
| B8 S512 d512 h8 | 10.324 | 9.927 | 1.040x | 0.0 |
| B8 S128 d32 h4 | 0.214 | 0.206 | 1.037x | 0.0 |
| B16 S128 d256 h8 | 1.945 | 1.935 | 1.005x | 0.0 |

Geometric mean **1.050x**. Through the harness itself the speedup column moves
**6.478x -> 7.348x**, **9.880x -> 11.485x** and **29.363x -> 31.308x**.

### Bit-identical, and that is the right bar

This removes tests whose outcome was already "pass". An interior tile is by
definition one where every predicate would have been true, so the arithmetic is
not merely equivalent -- it is the same arithmetic in the same order.

`verify_mask_classify.py` checks **exact equality**, not a tolerance, over 85
cases built to fail each clause in turn: `S` not a multiple of BLOCK_M (ragged
query block), `S` not a multiple of BLOCK_N (ragged key tile), causal, dense,
key-padding and full-row masks, both output layouts, and head_dim 8 where the
operands pad 8 -> 16 so `DIM != PDIM`. All 85 bit-identical.

Through the harness, `max_abs` **and** `max_rel` are identical to every digit
with the flag on and off -- `max_rel` especially, since it is dominated by
near-zero denominators and would move under any reordering at all.

### Registers went down, not up

The obvious worry is that carrying two paths costs registers and therefore
occupancy. Measured with `cuobjdump -res-usage`, it does the opposite:

| head_dim | before | after |
|--:|--:|--:|
| 16 | 105 | 96 |
| 32 | 102 | 76 |
| 64 | 128 | 128 |

`LOCAL:0` throughout. The fast path has no predicates to keep alive across the
unrolled score loop, so the register pressure that dominated the old body is
simply gone on the tiles that take it.

### The measurement trap: a cross-process ablation reported the impossible

The prize was supposed to be bounded first by an ablation -- strip the
predicates entirely, keep the work identical, time it. Built and timed in a
separate process from the real kernel, it produced rows like:

| shape | ablated | real |
|:--|--:|--:|
| B8 H8 S1024 d32 dense | 1116.2 us | 828.6 us |
| B8 H16 S512 d16 causal | 289.2 us | 159.1 us |

The ablated build **cannot** be slower than the real one; it does strictly less
work for the same answer shape. Both rows are contamination, and the whole
table was discarded. A compile-time ablation cannot be timed against its
baseline in one process, which is exactly why every other measurement here is a
runtime flag. **Build the A/B as a runtime switch from the start; a
compile-time ablation is not worth the process it has to run in.**

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

## head_dim 256: coverage everywhere, a win nowhere

Grading shape 8 (batch 64, seq 128, d_model 1024, 4 heads, causal) is head_dim
256, and it was the one shape in the set no kernel covered: `--attn-impl wmma`
refused it, `--attn-impl scalar` refused it, the tile kernels refused it, and
Auto quietly served it from SDPA. Every impl covers it now. None of them should
be used for it.

### What each kernel had to change

**scalar.** Nothing structural. Past head_dim 64 a query row is already split
across threads so that `q_reg + acc` clears the 255-register ceiling; head_dim
256 just makes that four threads instead of two. The one real change is the
score exchange: a single `__shfl_xor_sync(mask, s, 1)` only completes a pair, so
it became a `log2(TPR)`-step butterfly over the group. Per-thread registers are
therefore identical at head_dim 64, 128 and 256 -- 128 floats -- and what grows
is the block (64 -> 128 -> 256 threads) and what shrinks is the key tile
(BLOCK_N 64 -> 32 -> 16, holding `k_s + v_s` at 32 KB).

**wmma.** Q and O are register-resident across the whole head, so head_dim 256
gives each warp a 16-fragment `q_frag` and a 16-fragment `o_frag`. Nothing about
that is tunable; what is tunable is the block shape, and at 256 no shape fits in
the 48 KB a block gets for free -- Q, O and both K/V tiles are all `BLOCK_* x
256`. So head_dim 256 is the one head_dim that opts into the larger dynamic
shared-memory carveout, via one `cudaFuncSetAttribute` per instantiation.

**tile.** Only a `BlockCfg<256, ...>` per math mode and a dispatch case. The
kernel itself is already generic in HEAD_DIM.

### The wmma block shape, swept

Once the carveout is being paid for anyway, the shape is a free choice. Against
SDPA, causal and dense, ratio > 1 meaning the kernel wins, one build per
candidate:

|                    | 16x16 | 32x16 | 16x32 |
|--------------------|-------|-------|-------|
| causal b8h8s32     | 0.690 | 0.796 | 0.393 |
| causal b8h8s128    | 0.346 | 0.536 | 0.192 |
| causal b8h8s512    | 0.357 | 0.607 | 0.213 |
| causal b1h8s128    | 0.479 | 0.716 | 0.295 |
| causal b2h8s1024   | 0.390 | 0.700 | 0.259 |
| dense  b8h8s32     | 0.575 | 0.764 | 0.377 |
| dense  b8h8s128    | 0.269 | 0.434 | 0.143 |
| dense  b8h8s512    | 0.309 | 0.524 | 0.196 |
| dense  b1h8s128    | 0.472 | 0.725 | 0.225 |
| dense  b2h8s1024   | 0.305 | 0.528 | 0.203 |

32x16 wins all ten by 1.15x-1.55x, and not through occupancy: at 55 KB an SM
holds one block of two warps, which is exactly the 64 threads 16x16 gets from
two blocks of one warp. What it buys is half the blocks and therefore half the
K/V passes over global memory.

16x32 loses everywhere, which is the head_dim 128 result again (see "NEGATIVE:
the freed shared memory does not want to be spent on a bigger tile"): a wider
key tile doubles the staging without adding parallelism, and Q is already in
registers, so there is nothing for it to buy.

### Why Auto still routes head_dim 256 to SDPA

Interleaved against SDPA, causal, with an A/A control row, at the shape that
actually matters and three neighbours:

| shape                   | sdpa_ms | control | scalar | wmma  | tile  | tile-fp16 |
|-------------------------|---------|---------|--------|-------|-------|-----------|
| b64 h4 s128 (shape 8)   |  0.6745 |   0.0%  | 0.262  | 0.503 | 0.067 | 0.096     |
| b16 h4 s128             |  0.1835 |   1.1%  | 0.250  | 0.498 | 0.071 | 0.101     |
| b2  h4 s128             |  0.0378 |   2.0%  | 0.140  | 0.706 | 0.099 | 0.126     |
| b2  h4 s64              |  0.0292 |   1.4%  | 0.242  | 0.884 | 0.203 | 0.241     |

The best of the four is half SDPA's speed at the grading shape, and the closest
any of them gets anywhere is 0.884x on a shape small enough that SDPA is mostly
launch overhead. That is not a threshold to find; it is a loss everywhere. So
`wmma_preferred_by_auto` declines above head_dim 128 outright
(`kWmmaAutoMaxCandidateHeadDim`), and the two head_dim 128 clauses above it are
left describing only the case they were measured on.

The reason is structural rather than a tuning miss. This family of kernels keeps
Q and the output accumulator in registers for the whole key loop, which is what
makes them fast at head_dim 64. At 256 that same choice forces a 32-row block,
one block per SM, 64 threads -- a sixteenth of what the SM can hold -- and no
block shape recovers it while Q stays register-resident. SDPA's flash backend
tiles the head dimension instead. Beating it at head_dim 256 means a kernel that
does the same, and that is a different kernel, not a different constant.

So this is coverage: `--attn-impl wmma` (or scalar, or any tile mode) now runs
at head_dim 256 and reports its own honest time instead of refusing, which is
what the A/B tooling needs. It is not a speedup, and Auto does not take it.

### The coverage gap that was left, since closed

The tile kernels covered `{8,16,32,64,256}` after this pass -- 128 was still
missing. It is covered now; see "Tile kernel: head_dim 128" below.


## Tile kernel: head_dim 128

The last head_dim the tile kernels did not cover. Nothing in the kernel is
head_dim-specific -- it is a template parameter, and the body reads `HEAD_DIM`
through `ct::extents` -- so this was `BlockCfg<128, MODE, CAUSAL>` for the four
math modes, a `case 128:` in `splits_for_head_dim` and `launch_mode`, and the
macro block the sweep drives. No kernel change.

The shapes are where the work was. head_dim 128 is the head_dim where the choice
matters most: best to worst across nine candidates is **12.6x**, against
1.1x-1.3x at head_dim 64. Four of the five live tiles (Q, the accumulator,
K^T, V) scale with head_dim and only the score tile does not, so the spill
cliff sits inside the candidate set instead of at its edge.

### One set of builds, eight scores

`tune_block_shapes.py` builds once per (backend, mask mode, candidate) -- 9
candidates x 4 math modes x 2 mask modes would be 72 builds at ~2.5 minutes
each. But every build already contains all four math modes and both mask modes;
only the macros pick the shape. Pinning `FP32_*`, `BF16_*`, `TF32_*` and their
`_CM_`/`_CN_` causal twins to the same (M, N) in one `-D` set makes one build
serve all eight scores. 9 builds, not 72.

Scored as a geometric mean of per-case ratios against the best candidate on that
case, not a sum of raw milliseconds: seq 2048 is 30x seq 128 here, and summing
lets a shape that tanks the short cases win on the long one (rule 2 at the top
of this file). Dense cases: b8h8s128, b4h8s512, b1h8s2048. Causal adds
b64h1s128, which is grading shape 9.

### The narrow modes want the widest block that fits

| geo mean, dense  | tf32 | bf16 | fp16 |
|:-----------------|-----:|-----:|-----:|
| **64x16**        | **1.00** | **1.01** | 1.04 |
| 64x32            | 1.07 | 1.03 | 1.04 |
| 64x64            | 1.08 | 1.01 | **1.01** |
| 32x16            | 1.27 | 1.52 | 1.55 |
| 16x32            | 1.92 | 1.83 | 1.83 |
| 16x16            | 8.54 | 12.61 | 12.48 |

| geo mean, causal | tf32 | bf16 | fp16 |
|:-----------------|-----:|-----:|-----:|
| **64x16**        | **1.02** | **1.00** | **1.00** |
| 64x32            | 1.10 | 1.02 | 1.03 |
| 32x16            | 1.07 | 1.26 | 1.25 |
| 16x32            | 1.56 | 1.38 | 1.40 |
| 16x16            | 6.14 | 8.13 | 7.79 |

64x16 for all three, both mask modes. The three shapes with BLOCK_M 64 finish
within noise of each other and everything below them loses by 25% or more, so
what this measures is a *row count*, not a shape: the block needs 64 query rows
to have enough work in flight, and N past 16 buys nothing because the key tile
is already `HEAD_DIM x BLOCK_N` = 128x16. fp16 dense nominally prefers 64x64 by
3%, inside the 4.3% floor, and it inherits bf16's macros anyway.

16x16 is not merely last, it is 8-12x last. That is the cliff, and it is why the
head_dim 256 coverage shape was a bad prior to start from.

### fp32 splits on the mask mode, and by more than anything else in this file

| geo mean | dense | causal |
|:---------|------:|-------:|
| 32x16    | **1.00** | 1.59 |
| 16x16    | 1.48 | **1.00** |
| 16x32    | 1.21 | 1.83 |
| 32x32    | 1.59 | 4.31 |
| 64x16    | (4.0x worse than 32x16) | 3.98 |

fp32 keeps 32-bit operands, so 64x16 -- every narrow mode's winner -- is 4x
slower here. It lands on the other side of the cliff, and dense and causal then
disagree about where to stand: dense wants 32 query rows, causal wants 16, and
each loses ~1.5x taking the other's. A causal block walks m+1 key tiles instead
of S/BLOCK_N, so halving BLOCK_M costs it much less work than it costs a dense
block, while buying back the same occupancy. This is the clearest case in the
file for `FP32_CM_*` existing separately from `FP32_M_*`.

The fp32 dense row was measured in a second run of the same sweep. Cross-run
numbers are not comparable (rule 1), so it is reported as its own ranking; the
causal ranking reproduced across both runs to within 2% (32x16 read 14.11 and
14.34 ms at b1h8s2048), which is what makes the two tables trustworthy side by
side rather than the ratios between them.

### Against wmma and SDPA

Interleaved, one process, min of rounds, fp32 in and out. `control` is SDPA
timed a second time under another name: its true ratio is exactly 1.000x, so
what it reports is this run's noise floor.

| case (head_dim 128)   | control | wmma | tile-fp16 | tile-tf32 | tile (fp32) |
|:----------------------|--------:|-----:|----------:|----------:|------------:|
| b64 h1 s128 causal (shape 9) | 1.00x | **1.11x** | 0.90x | 0.64x | 0.32x |
| b8 h8 s128 causal     | 0.94x | **1.04x** | 0.85x | 0.64x | 0.29x |
| b8 h8 s128 dense      | 0.96x | **1.06x** | 0.79x | 0.60x | 0.11x |
| b8 h8 s128 causal+pad | 0.91x | **1.01x** | 0.83x | 0.60x | 0.14x |
| b4 h8 s512 causal     | 1.01x | 1.25x | **1.28x** | 0.82x | 0.20x |
| b4 h8 s512 dense      | 1.07x | **1.31x** | 1.30x | 0.78x | 0.11x |
| b1 h8 s2048 causal    | 1.03x | 1.50x | **1.67x** | 0.96x | 0.18x |
| b1 h8 s2048 dense     | 0.98x | 1.19x | **1.50x** | 0.87x | 0.11x |

Ratios are against SDPA; >1 is faster. Three things follow:

* **The crossover is around seq 512.** Below it wmma wins and tile-fp16 is
  slower than SDPA; at 512 they tie; at 2048 tile-fp16 is 1.12x (causal) to
  1.26x (dense) faster than wmma. Same shape as the head_dim 64 story -- the
  tile kernel needs a long key loop before its scheduling pays for itself.
* **Auto is unchanged.** It routes head_dim 128 to wmma, and wmma wins the two
  shapes that matter for grading (seq 128). `wmma_preferred_by_auto` did not
  move; nothing here is a default-path change.
* **tile-tf32 is dead at this head_dim.** 0.60x-0.96x, never a win, and
  tile-fp16 matches its accuracy exactly (9.0e-4 against the same reference)
  while running 1.3x-1.7x faster. Same 10 mantissa bits, half the operand width
  -- the fp16-vs-tf32 result from head_dim 64, reproduced.

`tile` (fp32) is 0.11x-0.32x and is coverage, not a candidate: it is the only
mode that does not round its operands, which is what `--attn-impl tile` is for.
Its error column reads 2.2e-3 in this table only because the reference is
computed with `allow_tf32` on, matching the harness baseline -- that number is
the *reference's* rounding, not the kernel's. Against an exact reference
`verify_kernel.py` reads 7.2e-7.

### Where tile-fp16 overtakes wmma: seq ~512, and it is sequence length

The table above samples three sequence lengths and reads "crossover somewhere
around 512". Swept finely, at three grid shapes, `tile-fp16 / wmma` (>1 means
the tile kernel wins):

| seq | b8h8 caus | b8h8 dense | b1h8 caus | b1h8 dense | b64h1 caus | b64h1 dense |
|----:|----------:|-----------:|----------:|-----------:|-----------:|------------:|
| 128 | 0.81 | 0.74 | -- | -- | 0.84 | 0.75 |
| 192 | 0.86 | 0.78 | -- | -- | -- | -- |
| 256 | 0.92 | 0.88 | 0.70 | 0.85 | 0.90 | 0.89 |
| 320 | 0.93 | 0.98 | -- | -- | -- | -- |
| 384 | 0.94 | 0.97 | 1.05 | 0.81 | 0.97 | 0.98 |
| 448 | 0.99 | **1.03** | -- | -- | -- | -- |
| 512 | 0.97 | **1.03** | 0.98 | 0.73 | 0.97 | 0.95 |
| 640 | **1.07** | **1.14** | 0.98 | 1.16 | -- | -- |
| 768 | 1.01 | **1.15** | **1.14** | **1.07** | 1.00 | **1.08** |
| 896 | 0.98 | **1.06** | -- | -- | -- | -- |
| 1024 | **1.08** | **1.07** | **1.20** | **1.08** | 0.96 | **1.08** |
| 1536 | -- | -- | **1.16** | 1.01 | -- | -- |
| 2048 | -- | -- | **1.10** | **1.20** | -- | -- |

The control column deviated up to 4.6% in this run, so +-5% is a tie. That makes
it a band, not a point: wmma wins outright to seq 256 (0.70-0.92, far outside
noise), 384-512 is a tie at 0.93-1.03, and from 768 up tile-fp16 leads in 9 of
11 rows by 1.07x-1.20x.

**It is sequence length, not grid size**, which is the part worth measuring
rather than assuming. The two kernels run different BLOCK_M -- 64 in the tile
kernel, 32 in wmma (`WMMA_M_128`) -- so they never put the same number of blocks
on the device at a given seq, and a threshold quoted in seq alone could really
have been a threshold in occupancy. Two comparisons separate them:

* At an identical 128 tile-blocks, `b8h8 s128` reads 0.81 and `b1h8 s1024` reads
  1.20. Same occupancy, opposite verdict.
* At seq 512 the ratio is 0.97 / 0.97 / 0.98 across grids of 512, 512 and 64
  blocks. Same seq, 8x the block count, same answer.

So what the tile kernel needs is a long key loop to amortise its scheduling, not
a starved grid -- the opposite of the split-KV story, where a starved grid was
exactly the problem. Which also means this threshold is not a candidate for a
dispatch rule keyed on block count.

`b1h8` is the noisiest series: 64-128 blocks is under one wave on 46 SMs, so it
is latency-bound, and its 384 (1.05) and 512 (0.73) dense rows disagree with
their neighbours by more than the trend does. Past the threshold the win is
modest -- 1.1x typical, against the 1.5x tile-fp16 posts over SDPA in the same
rows, because wmma is beating SDPA by 1.2x-1.4x there too.

Auto is unaffected either way: every grading shape is seq 128 or seq 1024, and
the seq-1024 one is head_dim 32.

### Past the threshold: dense is a reliable win, causal is a tie on big grids

"tile-fp16 wins above seq 512" is true of the dense kernel and much weaker than
that of the causal one. 40 (shape, mask) combinations above seq 512, each timed
in THREE independent passes -- one number per shape cannot separate a 7% win
from run-to-run variance. Ratio is `wmma_ms / tile-fp16_ms`, so >1 means the
tile kernel wins; `spread` is max-min across the three passes, which is the
honest error bar on the row.

|          | shapes | median | range | verdict |
|:---------|-------:|-------:|:------|:--------|
| dense    | 20 | **1.090** | 1.054 - 1.184 | wins all 20, every one outside noise |
| causal   | 20 | 1.012 | 0.947 - 1.203 | 6 wins, 13 ties, 1 loss |

Dense, seq 640-4096, grids from 8 to 64 `b*h`: every median between 1.054 and
1.184, spread under 0.09 on all but one row, and the tightest repeat to three
digits (`b1 h8 s1024` read 1.075 / 1.077 / 1.077). There is nothing ambiguous
in the dense half of this table.

Causal sorts by grid size, not by sequence length:

| causal, by `b*h` | rows | ratios |
|:-----------------|-----:|:-------|
| 64  | 9  | 0.985 - 1.022, median 1.005 -- a tie, every row |
| <=32 | 11 | mostly 1.04 - 1.20; two exceptions, `b2 h8 s2048` 0.974 and `b1 h8 s4096` 0.947 |

`b1 h8 s4096` causal is the only real loss in 40 shapes.

The mechanism is wmma getting better, not the tile kernel getting worse:
`wmma/sdpa` under causal climbs from ~1.20x at 8 blocks-per-head to 1.42x-1.53x
at 64, while `tile-fp16/sdpa` sits flat near 1.40x throughout. wmma runs
`WMMA_M_128` = 32 against the tile kernel's 64, so it has twice the blocks with
which to fill a large causal grid -- and causal is the mask mode where block
cost varies most, so more, smaller blocks schedule better.

This refines the subsection above rather than contradicting it. "Sequence
length, not grid size" is about *where the crossover sits*, and that still holds
-- at seq 512 the ratio was 0.97-0.98 across grids of 64, 512 and 512 blocks.
What grid size decides is *how large the win is once past it*, and under causal
that is the difference between a tie and 1.1x-1.2x.

One caveat on reading the raw table: several causal rows carry a control
deviation of 9%-12% while their own three passes agree to within 0.01 (`b2 h8
s1024` read 1.103 / 1.103 / 1.097 against a 12.1% control). That is SDPA's
variance between two timings of itself, and the `wmma/fp16` ratio does not go
through SDPA at all -- which is why `spread` is the error bar to read here and
the control only gates the `*/sdpa` columns.

### Routing Auto to tile-fp16 at head_dim 128: measured, and NOT done

The op-level tables above say tile-fp16 beats wmma by 1.05x-1.18x on dense
attention above seq 512. That is a real measurement and it is not a reason to
route to it. End to end -- 4 layers, `ffn_dim = d_model`, one process, impls
interleaved, `--attn-impl` forcing exactly the path a routing rule would pick:

| head_dim 128, e2e | causal | dense |
|:------------------|-------:|------:|
| b8 h4 s512        | 0.961x | 0.936x |
| b8 h4 s1024       | 0.968x | 0.988x |
| b16 h4 s1024      | 0.958x | 0.981x |
| b4 h4 s2048       | 0.986x | **1.031x** |
| b2 h8 s2048       | 0.988x | **1.042x** |
| b1 h8 s4096       | 0.953x | **1.073x** |

Control column 0.987x-1.015x. Causal loses in all six cases: the op-level tie
becomes a 1.2%-4.7% regression. Dense survives only from seq 2048, at 3%-7%
rather than the 5%-18% the op promised.

Two things eat it, and neither amortises:

* **The repack.** The tile kernels cannot write `[B, S, H*head_dim]`, so
  `fused_attention_forward` allocates `[B,H,S,D]` and calls `to_bshd()` -- a
  full transpose + reshape + allocation, once per layer. wmma writes the flat
  layout from its epilogue for free. That cost scales with the same tensor the
  win does.
* **Amdahl.** Attention shares the layer with four projections, an FFN, two
  LayerNorms and the residuals. 1.09x on the op is ~1%-2% on the model -- and
  `ffn_dim = d_model` is already the attention-favourable setting. A 4x FFN
  would be worse.

There is also a trap for whoever tries this anyway. `tile_mode` is derived from
the REQUESTED impl (attention_dispatch.cuh, `const bool tile_mode =`), and it
decides two things *before* `run_kernel` is reached: the causal->mask fold, and
`kernel_writes_bshd`. Routing Auto to a tile kernel from inside `run_kernel`
would allocate a `[B,S,H*D]` output and then let the tile epilogue write
`[B,H,S,D]` into it -- silently scrambled, not an error. Auto's choice has to be
resolved before the allocation, which makes this a dispatch refactor rather than
a clause in `wmma_preferred_by_auto`.

So Auto keeps sending head_dim 128 to wmma. A workload that really is long dense
attention at this head_dim can ask for `--attn-impl tile-fp16` and get the 3%-7%
without putting a new way to mis-lay-out the output into the default path.

### Measurement note

Part of this pass was measured while a second process was using the same GPU.
It was caught by the control column (0.44x on one row, where the true value is
1.000x) and by SDPA itself reading 4x its quiet time, and those runs were
discarded and re-run. Every number above comes from a run whose control column
sits inside 0.91x-1.07x. This is what the control row is for; a table without
one would have shipped the contaminated ranking.
