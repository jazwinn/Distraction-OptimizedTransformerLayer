# Optimizing a Transformer's Attention Layer

A record of the work done on the fused attention kernels in [`csrc/`](csrc/): what was
built, what broke, and what each fix was actually worth.

It opens with a plain-language account for readers who don't work with GPUs, then goes into
the detail of how the kernel was moved onto the tensor cores.

- [What happened, in plain language](#what-happened-in-plain-language)
- [The starting point](#the-starting-point)
- [The change](#the-change)
- [What each step was worth](#what-each-step-was-worth)
- [Performance](#performance)
- [Why the harness number looks small](#why-the-harness-number-looks-small)
- [Accuracy](#accuracy)
- [Coverage and limits](#coverage-and-limits)
- [CUDA graphs](#cuda-graphs)
- [What's left](#whats-left)
- [Reproducing](#reproducing)

---

## What happened, in plain language

### What this project is

A transformer is the kind of model behind most modern AI. The slowest part of it is
**attention** — the step where every word in a sentence compares itself against every other
word. The goal is to rewrite that step so it runs faster on the graphics card without
changing the answers it produces.

The catch is the second half of that sentence. An automated grader checks the fast version
against a known-correct version and rejects it if the numbers drift too far. Most of the
problems below come from that tension: **faster usually means less precise, and less precise
eventually means wrong.**

### The problems, and what fixed them

**1. The script wouldn't run a second time.**
The first run worked; every run after it hung forever. To stop two builds colliding, the
build tool drops a marker file meaning "busy" and deletes it when done — but a run killed
hard (closing the terminal, Task Manager) dies before cleaning up. Later runs then wait on
something already dead, with no timeout. Like a "do not disturb" sign left on a meeting room
door after the meeting ended. A normal Ctrl+C is safe; only force-killing leaves the sign up.

**2. You had to type a long wrapper command every time.**
Building GPU code needs Microsoft's C++ compiler, which Windows only exposes inside a special
developer terminal. The project now performs that lookup itself, so a plain
`python torch_transformer_benchmark.py` works from any shell. A latent bug surfaced on the
way: the compiler path was being quoted twice, so Windows couldn't find it.

**3. Building takes 70 seconds.**
Not a bug. GPU code is translated ahead of time, once per supported configuration, and that
genuinely costs about a minute. It is cached afterwards; later runs start in about four
seconds.

**4. The new technique "couldn't run" on this card.**
A published paper states that NVIDIA's new tile programming style requires a Blackwell GPU —
far newer than an RTX 3070. Rather than take that on faith, we wrote the smallest possible
test and ran it. **It works on this card.** Getting it to compile took four separate
discoveries, each fatal alone: a newer toolkit than the default, a newer language version, a
switch that is silently ignored when missing, and a compatibility flag for MSVC.

**5. That new version was 30x slower.**
Correct answers, 16.8 ms against 0.53 for the existing kernel. The card has a small pool of
very fast scratch memory, and this kernel keeps five things in it at once; past a certain
total the card starts shuffling data to slow memory. The surprise was that this is a **cliff,
not a slope** — one setting ran in 1.5 ms and the very next size up took 10.9 ms, with no
warning. Measuring every sensible size instead of guessing took it from **16.8 ms to 1.3 ms**,
nearly 13x faster, without changing a line of the maths.

**6. It still can't reach the card's fastest hardware.**
Graphics cards have dedicated multiplication units — *tensor cores* — that are far faster but
only accept reduced-precision numbers. The precision format that would suit this best has a
name in the new toolkit but **no actual definition**; the feature is announced, not finished.
Worth correcting a common assumption: **a newer graphics card would not fix this.** No tensor
core on any card, Blackwell included, handles full precision. The data format decides, not
the hardware. The upside is that this kernel became the **most accurate** of the three —
about a thousand times closer to the true answer.

**7. The fast kernel refused two shapes outright.**
Two configurations were declined, for two unrelated reasons that produced the same symptom.
One was **too small**: tensor cores only take work in fixed 16-wide blocks, and it was 8 wide
— half a block. The other was **too big**: it needed 75.8 KB of scratch memory against a
48 KB budget, where that budget is not the hardware limit but the point below which the card
can run two jobs at once. Fixes: pad the small one to 16 with zeros (free — zeros add nothing
to a sum, and the extra results are discarded), and give the large one a smaller working set
(75.8 KB to 35.9 KB). The project's own documentation claimed the second needed a fundamental
redesign; it didn't, and that note was written when the setting in question was assumed
fixed. Both now run **faster**, not merely supported: the large case went from 0.142 ms to
**0.041 ms**, turning a loss against the standard library into a win.

**8. Tidying the code quietly broke it.**
The logic choosing between kernels had grown tangled, so it was rewritten to read as four
plain lines. But the tidy-up also made all three kernels behave consistently when asked for
something unsupported — and one had always behaved differently. The test suite caught it
immediately. Original behaviour was restored exactly, with a comment explaining the
inconsistency rather than silently "fixing" it. A tidy-up that changes behaviour isn't one.

**9. Reaching the tensor cores after all.**
Since the fastest hardware needs reduced-precision numbers, we supplied them — using a format
called *bfloat16*, the only compact format this toolkit accumulates safely. (The obvious
alternative, fp16, keeps its running total at reduced precision too; since attention sums
hundreds of values per result, that quietly destroys accuracy. It would compile and run, just
produce bad numbers.) The result is **2-3x faster**, and on some shapes faster than the
existing kernel. It costs about four orders of magnitude of accuracy, so it **fails the
grader on four of five configurations** — and passes the fifth, where it is the fastest
option available. It is opt-in only.

**10. The second wrong diagnosis — a missing `#include`.**
The tile kernel's fp32 path was documented, in three places, as unable to reach the tensor
cores because the toolkit "does not define" the TF32 number type. Asked to justify that
claim, we went looking for the source and found it was never verified — it had been written
once and then cited by everything downstream, including this report. The type *is* defined,
in a header nobody was including. The tile headers include nothing at all; they name each
special number type and leave it to the caller to supply the definition, which this kernel
already did for bfloat16 and simply never did for TF32.

One line of `#include` turned it on. The result is a mode with **the same accuracy as the
existing tensor-core kernel** — it is the identical arithmetic — at tensor-core speed, and
it clears the accuracy gate everywhere that kernel does. The lesson is not about CUDA: a
plausible explanation, written down once, had been treated as established fact for as long
as nobody asked where it came from.

### One diagnosis that was wrong

A configuration with 12 layers fails the accuracy check. The explanation given at the time —
that attention's precision was to blame — was **wrong**, and measuring properly disproved it.
All four implementations land within 9% of each other there, including the two that never
lose precision at all. The drift comes from the other two-thirds of the model. The failure is
also one bad value in 2.6 million, close enough to the line that which implementation
"passes" changes with the random seed. **No amount of work on the attention kernel will fix
it.** It was the wrong target.

### Where it ended up

| Implementation | Speed | Accuracy | Best for |
| --- | --- | --- | --- |
| **wmma** (tensor cores) | Fastest overall | ~1 in 1,000 | Everyday use; the default |
| **tile** (fp32) | Slowest | ~1 in 1,000,000 | When precision matters most |
| **tile-tf32** | Fast | ~1 in 1,000 | The tile kernel on tensor cores; same accuracy as wmma |
| **tile-bf16** | Near-fastest | ~1 in 250 | Opt-in; fails the grader on most configs |
| **scalar** | Middle | ~1 in 1,000,000 | Older cards; a reference to check against |

The headline is **3.48x on long masked sequences**, and that every configuration is now
handled by a purpose-built kernel rather than falling back to a general one.

### The pattern worth remembering

Five of the nine problems had the same shape: **a stated limit turned out to be an assumption
nobody had retested.** The paper said the technique needed newer hardware. The documentation
said a configuration needed a redesign. The slow kernel looked like it needed better maths.

In each case the fix was a setting, not a rewrite — found by measuring rather than reasoning.
The two genuine limits that remain, the missing number format and the accuracy budget, are
real, and were left alone rather than worked around.

---

## The starting point

The original kernel was FlashAttention-shaped and correct, but it did its arithmetic the
slow way: **one thread per query row**, plain fp32 FMA, no tensor cores at all.

```cuda
float q_reg[HEAD_DIM];    // 64 floats
float acc[HEAD_DIM];      // 64 more
for (int j = 0; j < n_keys; ++j) {
    float s = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d)
        s += q_reg[d] * k_s[j * HEAD_DIM + d];      // scalar dot product
    ...
}
```

At `head_dim=64` that is ~128 floats of live state per thread, in blocks of only 64 threads
(2 warps). It left the MMA units — the single biggest source of arithmetic throughput on
the card — completely idle, and it measured slower than PyTorch's SDPA on most shapes.

## The change

Both of attention's matrix multiplies now run through `nvcuda::wmma` fragments:

| | |
| --- | --- |
| `S = Q @ K^T` | `16x16x8` TF32 fragments (fp32 in), `16x16x16` for half/bfloat16 |
| `P @ V` | same, accumulating straight into the output fragments |
| accumulate | fp32 in both cases |

A block owns **64 query rows** across **4 warps**, one 16-row stripe each — 16 being the `M`
of a wmma tile — and walks the keys in tiles of **32**. Per key tile a warp:

1. computes its `16x32` score block on tensor cores, into shared memory
2. runs the online softmax over those 16 rows
3. multiplies `P @ V` straight into its `16 x head_dim` accumulator fragments

Q and O both stay in registers for the entire key loop, so the only inner-loop traffic is
the K/V tile, the score tile, and the fragment reads feeding the MMA units. The
`[B, H, S, S]` score matrix is still never written to global memory.

For fp32 the operands are rounded to TF32, which is *the same arithmetic cuBLAS gives the
baseline* whenever `torch.backends.cuda.matmul.allow_tf32` is on — which is the harness
default. This is not a precision concession; see [Accuracy](#accuracy).

The old scalar kernel is still there. It covers what wmma cannot (`head_dim=8`,
pre-Ampere cards) and stays selectable via `--attn-impl scalar`, which is what made the
measurements below possible.

## What each step was worth

Calling `mma_sync` was the easy part and, on its own, **a regression**. Three further
changes turned it into a win. Each row is the `seq2048` shape, measured against the scalar
kernel *in the same run* — the GPU is a power-capped laptop part whose absolute latencies
drift, so only same-run ratios are meaningful:

| Stage | vs. scalar kernel |
| --- | --- |
| wmma, naive | **0.75x** (slower) |
| \+ padded shared-memory leading dimensions | 0.90x |
| \+ Q and O held in registers | 1.13x |
| \+ one softmax lane per row | **1.94x** |

### 1. Padded leading dimensions

A fragment load walks a *column* of a shared-memory tile. With a row stride of 16, 32 or 64
floats, every row of the fragment lands in the same shared-memory bank and the load
serialises.

Every 2-D tile now carries a padded leading dimension — the smallest pad wmma permits
(`ldm` must be a multiple of 4 floats, or 8 halves), which is enough to rotate successive
rows off each other:

```cuda
static constexpr int KV_LD = HEAD_DIM + PAD;   // k_s, v_s, Q staging
static constexpr int O_LD  = HEAD_DIM + 4;     // o_s is always fp32
static constexpr int S_LD  = BLOCK_N + PAD;    // s_s and p_s
```

### 2. Q and O in registers

Q is read into fragments once per block and never re-read. O accumulates in accumulator
fragments across the whole key loop rather than being written back to shared memory each
tile — worth roughly 40% of the inner loop's shared-memory traffic.

That second one has a catch. The per-row softmax rescale (`O = O * corr`) has to be applied
to individual fragment *elements*, and the element-to-row mapping inside an accumulator
fragment is architecture-defined — CUDA deliberately does not document it.

Rather than hard-code a layout that could silently be wrong on another card, the kernel
**probes** it once per block: store a fragment whose elements are tagged with
`(lane, slot)`, read back which position each tag landed in, invert the mapping.

```cuda
acc_frag_t probe;
for (int t = 0; t < ACC_ELEMS; ++t)
    probe.x[t] = static_cast<float>(lane * ACC_ELEMS + t);
wm::store_matrix_sync(probe_out, probe, 16, wm::mem_row_major);
__syncwarp();
for (int t = 0; t < ACC_ELEMS; ++t) {
    const int pos = lane * ACC_ELEMS + t;
    tag_to_row[static_cast<int>(probe_out[pos])] = pos / 16;
}
```

One 16x16 tile per warp, once per block, exact by construction on any device the kernel
compiles for. The rescale then becomes `o_frag[n].x[t] *= corr_of[t]`.

### 3. One softmax lane per query row

The obvious lane mapping is `lane == key column`, which makes a row reduction a 5-step
butterfly — 16 rows deep, 160 shuffles per warp per key tile. That cost is **independent of
`head_dim`**, so at `head_dim=16` it swamped both GEMMs.

Giving each lane a whole row segment instead leaves exactly one shuffle, between the two
lanes that share a row. This was the single largest step, and it is why `head_dim=16` went
from 0.78x to 1.74x against the scalar kernel.

### 4. Hoisting P out of the output-tile loop

`P` does not depend on the output tile `n`, so it is loaded once per key tile instead of
once per `n`. Strictly fewer shared-memory loads, though it measured inside the noise —
the kernel was not P-load bound by that point.

## Performance

### The attention op

fp32, all nine shapes, candidates timed **interleaved** and minimum-of-N so they see the
same clock state:

| Shape | vs. scalar kernel | vs. PyTorch SDPA |
| --- | --- | --- |
| default (B8 H8 S128 D64) | 2.50x | 1.63x |
| default causal | 2.60x | 1.54x |
| default padded | 1.70x | 1.44x |
| seq 512 | 1.99x | 1.72x |
| seq 2048 | 1.85x | 1.79x |
| seq 2048 causal | 2.19x | 1.78x |
| head_dim 32 | 2.43x | 2.30x |
| head_dim 16 | 1.74x | 3.64x |
| wide (d_model 1024, 16 heads) | 2.40x | 1.57x |
| **total** | **2.00x** | **1.81x** |

Every shape improves, and the tensor-core kernel is faster than SDPA on all of them. An
earlier independent run of the same table gave 2.07x / 1.87x, which is the size of the
run-to-run error bar.

### Why the tile kernel loses to wmma, and where it doesn't

The table above ranks wmma first and tile-tf32 second, which invites the obvious question:
the tile kernel is written at a higher level, the compiler does the register allocation and
the load scheduling, so why is the hand-written one faster? The answer is worth writing down
because it is **not** a code-quality problem, and three rounds of trying to fix it as one
came back empty.

Timed interleaved in a single process, five rounds, best round kept — head_dim 64 unless
noted, fp32 in and out for every row. **Measured on an RTX 3070 (SM 8.6, 46 SMs), CUDA 13.3,
PyTorch 2.12.0+cu132** — not the RTX 4050 Laptop this document's footer names, so these
milliseconds are not comparable with the tables above. The ratios inside this section are
internally consistent; ones drawn across sections are not.

| Shape | wmma-tf32 | tile-tf32 | tile-bf16 | tile-fp32 |
| --- | --- | --- | --- | --- |
| B2 H8 S512 dense | **0.170** | 0.250 | 0.140 | 0.372 |
| B2 H8 S512 causal | **0.114** | 0.155 | 0.103 | 0.245 |
| B1 H8 S2048 dense | **0.973** | 1.289 | 0.865 | 2.746 |
| B1 H8 S2048 causal | **0.579** | 0.807 | 0.495 | 1.463 |
| B2 H8 S2048 dense | **1.915** | 2.572 | 1.705 | 5.317 |
| B2 H8 S2048 causal | **1.032** | 1.503 | 0.923 | 2.776 |
| B2 H8 S1024 head_dim 32 dense | **0.254** | 0.308 | 0.202 | 0.743 |
| B2 H8 S1024 head_dim 16 dense | 0.158 | **0.126** | 0.077 | 0.387 |

Geometric mean: **tile-tf32 is 1.28x slower than wmma; tile-bf16 is 1.25x faster.** Read the
right row before concluding anything — tile-bf16 does beat the hand-written kernel, but it is
8 significand bits against tf32's 10, so it is a cheaper computation and not a faster
implementation of the same one. Against the mode that does the *same* arithmetic, wmma wins
almost everywhere. head_dim 16 is the one exception.

Three things explain the gap, and none of them is the kernel source.

**The tf32 conversion is software in cuTile and free in wmma.** `ct::tile<__nv_tf32,...>(x)`,
the only way to reach the tf32 MMA units from tile code, lowers to a full IEEE
round-to-nearest-even written in integer operations — masks against `0x7f800000`, tie
detection on `0xfff` and `0x3000`, `+0x2000`, a NaN fixup, and a final `& 0xffffe000`. That is
about fourteen instructions per element. `wmma::__float_to_tf32` compiles to **nothing**:
ptxas deletes it outright, because `HMMA.1688.F32.TF32` reads ordinary fp32 registers and
rounds in hardware. Grep the SASS for `0xffffe000` to see which one you have.

**Occupancy.** tile-tf32 at its tuned 128x32 shape needs **95.5 KB** of static shared memory,
which on this card is **one block per SM** — four warps, with nothing else resident to hide a
memory stall behind. wmma asks for **43.8 KB** dynamically and gets **two**.

**It is not instruction-bound, so the usual levers do nothing.** This is the part that took
longest to accept. Counting SASS at head_dim 64, the tile kernel emits the *better* tensor-core
instruction — `HMMA.1688` (m16n8k8, 1024 MACs per warp-op) where wmma's `mma_sync` on tf32
fragments lowers to `HMMA.1684` (m16n8k4, 512), so wmma issues twice as many MMA instructions
for the same work. Normalised per MAC of attention, tile-tf32 executes **1.03** instructions
where wmma executes **1.38**. It is more instruction-efficient and still slower. Consistent
with that, removing 12% of the tile kernel's instructions once bought 3% of its time.

The clinching evidence that the programming model is not at fault is the bf16 column. Same
tile source, same compiler, same one-block-per-SM occupancy — the only thing that changes is
that cuTile's bf16 narrowing gets a hardware `F2F` instead of the software rounding sequence.
With the conversion free, the tile kernel beats the hand-written one.

So the honest summary of "tile programming is fast": it buys **portability and a great deal
less code** — the tile kernel is a few hundred lines against the wmma kernel's fragment
choreography, shared-memory padding and manual barriers — and on this hardware it costs about
28% on the tf32 path. cuTile is aimed at Hopper and Blackwell, where TMA and a larger shared
allocation exist; the paper describing it asserts compute capability 10.0 or newer is required,
which is wrong for CUDA 13.3 but tells you plainly what it was tuned against. On SM 8.6 it
runs correctly and gets none of that hardware.

What this rules out is worth as much as what it explains. Every plausible source-level fix was
A/B'd with both variants compiled and timed in one process: hoisting the Q cast out of the key
loop, replacing `ct::transpose(k)` with a `layout_left` K-transpose view, `ct::mma(a,b,c)` in
place of `c + ct::matmul(a,b)`, skipping the bounds `select` on interior tiles, static
`integral_constant` extents, and plain `load()` instead of `load_masked()`. All measured
neutral or worse — the last one 12% worse, because `load_masked` is what gets the TMA path.
The tile compiler had already normalised the rest. Reading K and V through a `__nv_tf32*`
reinterpret does remove ~520 instructions and 3% of the time, but it is truncation rather than
rounding and the error grows ten-fold, from 3.1e-4 to 3.0e-3; it was rejected.

**Practical reading.** For fp32 in and out on this GPU, wmma is the right default and the
table at the top of this document already says so. If the accuracy budget tolerates bf16,
tile-bf16 is the fastest kernel here and by a wide margin the least code. The tile-tf32 path
earns its place as the same-accuracy comparison that makes both of those claims checkable, not
as the fast option.

### The whole transformer

Harness speedup against `BaselineTransformer`:

| Configuration | SDPA | custom scalar | custom wmma |
| --- | --- | --- | --- |
| default (B8 S128 D512 H8 L6) | 0.999x | 0.900x | **1.007x** |
| padded (30%) | 0.992x | 0.846x | **1.001x** |
| seq_len 512 | 1.365x | 1.268x | **1.398x** |
| seq_len 2048 | 1.816x | 1.678x | **2.270x** |
| seq_len 2048, causal | 3.124x | 2.915x | **3.785x** |
| small (B1 S32) | **1.529x** | 1.268x | 1.497x |
| wide (d_model 1024) | **1.033x** | 0.961x | 0.988x |

Differences under ~10% here are not evidence of anything — a single harness run gives one
median with no interleaving, and `deep 12L` was measured at both 1.401x and 0.958x on two
runs. `seq_len 2048` and `seq_len 2048 causal` are well outside that band and agree with the
attention-op table.

## Why the harness number looks small

At the default configuration the end-to-end speedup is ~1.05x, which looks underwhelming
until you count where the work actually is. Per layer:

| | S=128 | S=2048 |
| --- | --- | --- |
| Q/K/V/out projections | 1.07 G MAC | 2.15 G MAC |
| FFN | 2.15 G MAC | 4.29 G MAC |
| **attention core** | **0.13 G MAC (4.0%)** | **4.29 G MAC (40.0%)** |
| best possible end-to-end speedup | **1.042x** | **1.667x** |

The harness's baseline and optimized paths differ in *nothing but attention* — `MyLinear`
calls `F.linear` exactly like `nn.Linear`, `MyLayerNorm` calls `F.layer_norm` exactly like
`nn.LayerNorm`, same GELU. So 96% of the work at `S=128` is byte-identical in both, and
making attention infinitely fast would still only buy 1.042x.

Measured: **1.047x**. The kernel is already past the pure-FLOP ceiling, because the
baseline's attention is memory-bound rather than FLOP-bound — it materialises a
`[8,8,128,128]` fp32 score matrix, 33 MB per layer, and round-trips it through global
memory for the mask, the softmax and the second matmul. Eliminating those round-trips wins
more than the arithmetic alone predicts.

The same holds at the other end: at `S=2048` the ceiling is 1.667x and the measured result
is 2.270x, against a score matrix of 134 MB per layer.

**Attention scales as S² while everything else scales as S.** That is the whole story of
this table, and the reason to benchmark at long sequences if you want to see the kernel
rather than the FFN.

## Accuracy

The harness runs the **baseline** with TF32 enabled by default, so the reference the kernel
is measured against is itself computed on tensor cores. Moving to TF32 fragments therefore
brings the kernel *closer* to the reference, not further:

| Shape | scalar kernel | tensor-core kernel |
| --- | --- | --- |
| default | 7.1e-4 | **3.7e-4** |
| seq2048 | 2.1e-4 | **6.4e-5** |
| seq2048 causal | 1.4e-3 | **7.0e-4** |

*(max abs error against a TF32 reference — the arithmetic the baseline actually runs.)*

End to end this does **not** translate into headroom. Over 12 accuracy trials the scalar
kernel peaked at `max_abs` 7.7e-4 and the tensor-core kernel at 8.7e-4. Both pass
`atol=1e-3`; neither margin is comfortable. The harness metric is set by a handful of
cancellation outliers — trials routinely report `max_rel` in the hundreds, meaning the worst
element is one whose reference value is near zero — rather than by systematic rounding,
which is why halving the systematic error barely moves it.

Three configurations sit on the gate on this hardware (`causal`, `causal + padded`,
`deep 12L`) and fail intermittently by one or two elements out of millions — **for every
backend, including stock PyTorch SDPA**. `deep 12L`, 8 trials each:

| | verdict | `max_abs` | failed elements |
| --- | --- | --- | --- |
| SDPA | FAIL | 1.101e-3 | 1 / 4,194,304 |
| custom scalar | PASS | 1.187e-3 | 0 |
| custom wmma | FAIL | 1.140e-3 | 2 / 4,194,304 |

The backend that passed has the **largest** error of the three. This is not a precision
ranking — the criterion is `abs <= atol OR abs <= rtol * abs(ref)`, so the verdict turns on
whether a few near-zero-reference elements happen to land inside `atol`.

## Coverage and limits

| dtype | head_dim 8 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- |
| float32 | wmma | wmma | wmma | wmma | wmma |
| float16 | wmma | wmma | wmma | wmma | wmma |
| bfloat16 | wmma | wmma | wmma | wmma | wmma |

Tensor cores need compute capability **8.0+**; below that the scalar kernel runs. Selection
is automatic — `--attn-impl` only exists to force a path for measurement.

**The two former gaps are closed, and both were block-shape problems.**

`head_dim=8` is narrower than the 16-wide wmma fragment, so GEMM2's N dimension could not
be filled. The kernel now widens it to 16 with zeros in shared memory: GEMM1 contracts over
head_dim, where zeros add nothing, and GEMM2 produces columns past 8 that are simply not
stored. Nothing extra is read from or written to global memory.

`head_dim=128` at the default `64x32` block wanted 75.8 KB of shared memory, over the 48 KB
that keeps two blocks resident per SM. It did *not* need a different tiling: a `32x16` block
brings the same head_dim down to 35.9 KB. `WmmaShape` now picks the block per head_dim, and
`WARPS` follows from `BLOCK_M` so the warp/lane mapping stays consistent.

Both were worth real speed, not just coverage — at `head_dim=128` the tensor-core kernel
runs 0.041 ms against the scalar fallback's 0.142 ms, turning a 0.40x loss to SDPA into a
1.40x win.

## CUDA graphs

Candidate 1 from the section below is now implemented, so this records what it was actually
worth rather than what it looked like it would be worth.

A CUDA graph records a forward pass's kernel launches once and resubmits them as a single
driver call. Nothing about the arithmetic changes — the same kernels run in the same order on
the same addresses — so the interesting question is entirely about launch overhead, and the
accuracy question does not arise. `--cuda-graph {off,auto,always}`; `auto` is the default.

### What it is worth

Harness speedup against `BaselineTransformer`, fp32, custom wmma kernel:

| Configuration | vs. baseline, graphs auto | graph vs. eager, interleaved |
| --- | --- | --- |
| small (B1 S32) | 11.0x – 14.1x | **4.2x – 6.8x** |
| default (B8 S128 D512 L6) | 1.35x – 2.08x | 1.01x – 1.03x |
| causal | 1.54x – 1.94x | not measured |
| padded (30%) | 1.30x – 1.53x | not measured |
| deep 12L | 1.32x – 1.58x | 1.040x |
| seq512 B4 | 1.60x – 1.70x | 1.009x (eager, over gate) |
| seq2048 B1 | 2.33x – 2.39x | 1.007x (eager, over gate) |
| wide (d_model 1024) | 1.20x – 1.24x | not measured (eager, over gate) |

Ranges over repeated runs rather than single readings; the section below on host contention is
why the spread is this wide. Only the right-hand column is trustworthy below ~10%. At the default
config, repeated off-vs-on pairs gave 1.263x/1.353x and then 1.352x/1.349x, so the harness's own
median cannot resolve a 3% effect at all. The interleaved `default` figure held at 1.01–1.03x
across every machine state tested, which makes it the one number here worth quoting flat.

And on the op itself, interleaved best-of-5 with an eager-vs-eager control row giving a
±1.2% noise floor (`scripts/ab_graph.py`):

| batch × seq | eager | graph | ratio |
| --- | --- | --- | --- |
| 1 × 32 | 1.973 ms | 0.466 ms | **4.23x** |
| 1 × 128 | 1.961 ms | 0.815 ms | 2.41x |
| 8 × 32 | 1.966 ms | 0.967 ms | 2.03x |
| 8 × 128 | 3.746 ms | 3.641 ms | 1.029x |
| 8 × 256 | 6.870 ms | 6.800 ms | 1.010x |
| 8 × 1024 | 33.840 ms | 33.917 ms | 0.998x |

Replay at `small` lands on 0.466 ms against the 0.474 ms the kernels themselves take — the
launch gap is essentially all recovered. **Nothing measured was ever slower than eager**; the
worst case, 0.998x, is inside the control's spread. So the size gate is not protecting against
a slowdown, it is protecting memory.

### The gate is on activation volume, and it is a measured constant

`_GRAPH_MAX_ACTIVATION = 524288`, i.e. `batch*seq*d_model`. Two findings shaped it.

**Tokens are the wrong axis.** `batch*seq` mispredicts badly: at 512 tokens, `d_model` 256 gave
**2.708x** and `d_model` 512 gave 1.036x — same token count, 2.6x difference in payoff. What
decides it is work *per kernel*, which scales with the activation tensor rather than with its
rows. Holding `batch*seq*d_model` constant in pairs while moving tokens and width in opposite
directions gave 1.038x and 1.030x at 524288, agreeing inside the noise floor.

**Depth is not an axis at all.** At fixed activation volume, 3, 6, 12 and 24 layers gave 1.031x,
1.037x, 1.040x, 1.040x. Eager and replay scale with depth together, so `num_layers` does not
belong in the gate even though launch *count* doubles across that range.

The threshold caps the pinned pool at about 84 MiB, measured. An earlier estimate in this document
guessed "gigabytes"; the real range across everything swept is 22–360 MiB, because the caching
allocator reuses freed blocks *inside* a private pool, so the pool tracks the eager peak rather
than the sum over layers — which is also why 6 and 12 layers both reserved 62 MiB at the same
shape.

**The honest caveat: 524288 is a property of this machine, not of the model.** The crossover is
where the GPU stops starving, which depends on the card's throughput against how fast the host can
feed it, so a faster GPU starves at larger shapes and wants a larger value — and a busier host
wants one too. A memory-relative budget was tried as a portable alternative and reverted: at 8% of
an 8 GiB card it captured five shapes that gain ~1% or nothing, pinning 148–360 MiB apiece,
which is a poor trade for portability nobody on this machine benefits from. The constant stays, made
portable enough by two things instead. `scripts/ab_graph.py --recommend` re-derives it on whatever
machine it runs on: it sweeps activation volume in powers of two with two shapes at each level,
takes the worse of the pair so the answer errs toward capturing less, derives the noise floor from
its own eager-vs-eager control rows, and **refuses to answer at all** when that floor is wider than
the effects being measured — the alternative being to hand back a number derived from noise, which
is exactly the mistake the game caught me making. And the harness prints which path it took, why,
and how to re-measure, on every run. On this machine `--recommend` independently returns 524288,
which is how the hand-read value was checked.

Nothing swept was ever *slower* than eager — worst case 0.998x, inside the control's spread — so
this gate is not protecting against a slowdown. Set it too low and it costs latency; too high and
it costs memory; neither can produce a wrong answer.

A generous safety net sits behind it regardless: a captured pool over 25% of the card is released
rather than held for the run. At the default threshold it should never fire, which is the point —
it exists so that a much larger constant set on unfamiliar hardware degrades to eager instead of
quietly eating the card.

### Absolute speedups are unusually sensitive to host contention

Worth recording because it caught me out mid-measurement: one run reported 2.083x at the default
config against 1.37x earlier the same day with no code change between them. The machine had a
game running.

The mechanism is clean once seen. The graph path issues one launch per forward pass, eager issues
79, and the baseline issues more still — so anything that makes a launch more expensive penalises
the three unevenly:

| path | game closed | game running | degradation |
| --- | --- | --- | --- |
| baseline | 5.0 ms | 9.2 ms | 1.84x |
| optimized, eager | 3.6 ms | 6.0 ms | 1.67x |
| optimized, graphed | 3.6 ms | 4.65 ms | 1.29x |

Degradation ordered exactly by launch count, which is what says the added cost is host-side rather
than a throttled GPU — a throttled card would slow all three together.

The consequence is that the *ratio* inflates on a busy machine: `b1 s32` moved across 4.23x, 5.29x
and 6.80x on three different states. Those numbers are all real for their state, not artefacts —
removing launches is worth more when launches cost more — but none of them is *the* number, which
is why the tables above give ranges.

The interleaved default-config figure is the exception, reading 1.029x, 1.033x and 1.013x across
those same three states. That split is not a coincidence: **the shapes where graphs barely matter
are the stable ones, and the shapes where they matter enormously are the volatile ones**, which
follows directly from the mechanism. A number quoted for a launch-bound shape is a statement about
two machines, not one.

This is also what the control row in `ab_graph.py` is for, and it earned its place here. With the
game closed it read 0.961–1.015x; with it running, 0.849–1.340x, at which point several rows had a
control larger than the effect they claimed to measure and nothing in them meant anything. Any
table out of that script should be discarded unless its control column is read alongside it.

### Bit-exactness, measured against the accuracy gate

`verify_graph.py` compares replay to eager directly. The stronger check is whether the two ever
disagree about the *verdict*, so 6 configs x 8 seeds were graded against the baseline twice on
the same inputs, once with capture off and once on:

| | pairs | failed the gate | identical to eager, bit for bit |
| --- | --- | --- | --- |
| eager | 48 | 2 | — |
| graphed | 48 | 2 | **48/48** |

Same two rows fail on both sides, with `max_abs` agreeing to the last digit and a gap of exactly
`0.0e+00` on every pair. Both failures are `causal+padded` (seeds 2 and 4), one element of
2,621,440 each — which is the pre-existing margin described under [Accuracy](#accuracy), not
anything graphs introduced. `causal+padded` fails on roughly a quarter of seeds either way.

The useful conclusion is the negative one: **a graph cannot rescue an accuracy failure and
cannot cause one.** It is a latency switch, and the accuracy discussion elsewhere in this
document applies to it unchanged.

**Bit-exactness is asserted, not assumed.** Capture replays once against an eager reference and
refuses to install a graph whose output differs by anything at all. Nothing should ever trip
this — and nothing has — but a graph that computes something slightly different is the one
failure mode a benchmark would happily report as a *win*, which makes it worth one extra eager
forward per shape to rule out.

### Unrelated bug found on the way

The cuTile kernels crash the interpreter at shutdown with an access violation (`0xC0000005` on
Windows), *after* `main()` has returned its exit code. This has nothing to do with graphs — it
reproduces with `--cuda-graph off`, and `scripts/verify_split_kv.py` already exits the same way.
`scripts/verify_graph.py` therefore checks `scalar` and `wmma` by default, with `--include-tile`
opting in; the tile kernels are bit-exact under capture, it is their teardown that is broken.

## What's left

Everything above is attention. This section is the other direction: given a kernel that is
already faster than SDPA on every shape, what is still worth doing to the model around it?

The measurements here come from a profiler run over the optimized path at four shapes with
the harness's own settings (`allow_tf32=True`, `matmul_precision="high"`), reading leaf CUDA
kernels only — `key_averages()` reports `aten::linear`, `aten::addmm` and the underlying
`cutlass_80_tensorop_s1688gemm` as three separate rows with the same time, and summing them
triple-counts every GEMM.

### Where the time goes now

| | default B8·S128 | seq2048 | wide d1024 | small B1·S32 |
| --- | --- | --- | --- | --- |
| cuBLAS GEMMs (projections + FFN) | **66.4%** | 36.4% | **79.8%** | 13% |
| fused attention kernel | 8.4% | **51.2%** | 4.7% | 2% |
| GELU | 6.7% | 4.8% | 5.7% | <1% |
| fused add+layernorm | 6.5% | 4.7% | 4.7% | 2% |
| q/k/v `.contiguous()` copies | 5.6% | 3.6% | 3.6% | 1% |
| idle — CPU launch gaps | ~6% | ~0% | ~1% | **~80%** |

Two shapes are worth reading closely. At `wide`, four fifths of the forward pass is cuBLAS,
and the attention kernel — the whole subject of this document — is 4.7%. At `small`, the GPU
is idle for four fifths of the wall clock, waiting to be fed.

### What decides whether launch overhead is exposed

Kernel launches are asynchronous, so the CPU races ahead queueing kernel *n+1* while the GPU
is still running *n*. If the average kernel outlasts the time it takes to issue one, the CPU
stays ahead, the queue never drains, and launch cost is invisible. The GPU only stalls when
the average kernel finishes faster than the next can be submitted.

Counting launches does not predict this. Counting *duration* does:

| | kernels/fwd | mean kernel | GPU idle |
| --- | --- | --- | --- |
| small B1·S32 | 79 | 5.99 us | **80.3%** |
| default B8·S128 | 79 | 48.98 us | 10.4% |
| seq2048 | 91 | 1018.29 us | 0.9% |

`small` and `default` issue the identical 79 kernels per forward pass and sit at opposite
ends of the table; `seq2048` issues *more* than either and is never launch-bound at all. The
rule of thumb is to compare mean kernel duration against launch cost — roughly 3-8 us of CPU
— rather than comparing kernel count against one. Without a profiler, the same question is
answered by timing the forward loop without synchronising, to get CPU dispatch cost, and
comparing that to GPU time: if they are equal, the GPU is starved.

### Implementation changes and architecture changes are not the same list

A survey of "transformer optimization techniques" will mix two categories that this harness
treats very differently:

* **Implementation changes** — CUDA graphs, fused epilogues, lower-precision GEMMs — compute
  the same function by a different route. They are graded on speed and on staying inside
  `atol=1e-3`.
* **Architecture changes** — SwiGLU gating, RMSNorm — compute a *different function*.

`copy_model_weights` loads the baseline's `state_dict` with `strict=True`, and
`compare_outputs` compares against `BaselineTransformer`'s own output elementwise. A SwiGLU
FFN has no `W3` to load; RMSNorm drops the mean subtraction `nn.LayerNorm` performs. Both
miss by O(1), not by O(1e-3). They are what you do when training a new model, not when
optimizing a fixed one, and neither can appear in this project.

### Ranked candidates

**Done: CUDA graphs.** Was candidate 1 here. Worth 4.23x on the forward pass at `small` and
1.029x at default, bit-exact. See [CUDA graphs](#cuda-graphs) for what it measured and for
the three things this section originally got wrong about it.

**1. The q/k/v `.contiguous()` copies — 5.6% at default, bit-exact.**

[`fused_attention.cu`](csrc/fused_attention.cu) materializes contiguous copies of q, k and v
because the fused QKV projection hands the kernel permuted views of one packed tensor. That
is 18 clone kernels per forward pass for a layout change the kernel could absorb by reading
strides — or avoid entirely by taking the packed `qkv` tensor and indexing it directly,
since `MySelfAttention` produced that layout itself.

**2. Fused Linear+GELU — ~5% available, and the cheap route collects 2% of it at a price.**

PyTorch exposes cuBLASLt's bias+GELU epilogue as `torch._addmm_activation`. Measured at the
default FFN shape:

```
F.linear + F.gelu(erf)          188.7 us
torch._addmm_activation fused   184.6 us    <- 2%; cuBLASLt picks a slower GEMM algo
F.linear alone                  138.9 us    <- so GELU itself is only ~50 us
```

It is also the **tanh** approximation, not erf: 3.87e-4 against an erf reference, matching
tanh to three digits. Against end-to-end `max_abs` that already peaks at 8.7e-4 on a 1e-3
gate, that is most of the remaining margin for 2%. Collecting the full 5% means hand-writing
a TF32 tensor-core GEMM with a GELU epilogue that beats cuBLAS.

**3. INT8 / W8A8 — available on this hardware, dead on accuracy.** SM 8.6 has INT8 tensor
cores. But the gate is `abs <= 1e-3` against an fp32 reference, and bf16 — far finer than
INT8 — already fails it for 29% of elements.

**4. RMSNorm — disqualified above, and it would not pay.** The argument for it is one fewer
pass over memory; `fused_add_layernorm` already removed that pass. The mean subtraction is
arithmetic on data that is already in registers, competing for a fraction of 6.5%.

**5. SwiGLU — disqualified above.**

**6. FP8 — impossible here.** FP8 tensor cores start at Ada (SM 89). This card is SM 8.6.

### What that adds up to

Graphs are collected: default went 1.263x -> 1.353x and `small` went 2.49x -> 11.23x. What
remains at the default configuration is the q/k/v copies at 5.6%, plus GELU at ~5% if it is
written by hand — so roughly 1.35x to 1.50x. The other two thirds of the forward pass is
cuBLAS, and beating cuBLAS is not a weekend.

So **the q/k/v copies are now the whole remaining opportunity worth having** at default
shapes. Note that graphs paid most at exactly the shapes where the attention kernel paid
least, which is the useful shape of this result: the two cover different regimes rather than
competing for the same time.

The usual caveat from [Performance](#performance) applies to the small numbers here. The
graph figures are the exception — `scripts/ab_graph.py` times both sides interleaved in one
process and prints an eager-vs-eager control row, which put the noise floor at ±1.2%, so the
1.029x at default sits just above it rather than inside it. The harness's own single-run
median cannot see that effect at all.

## Reproducing

```bash
cmd.exe /c scripts\build_ext.bat     # build once (optional; a plain run builds too)
python scripts\verify_kernel.py      # every kernel, 12 shapes
python scripts\bench_attention.py    # attention-op table
python scripts\compare_backends.py   # full harness sweep
python scripts\sass_mix.py           # SASS instruction mix + occupancy, head_dim 64
```

`kernel_ext.py` puts MSVC on `PATH` itself, so none of these need the `devenv.bat`
prefix any more.

A third kernel now sits alongside the two this document describes:
[`csrc/tile_attention.cu`](csrc/tile_attention.cu) is the same FlashAttention math
written against the CUDA tile programming model rather than per-thread. It is selected
with `--attn-impl tile`, needs CUDA 13.3+, and is documented in the README under
[Notes and known limits](README.md#notes-and-known-limits).

To see the kernel rather than the FFN, benchmark where attention dominates:

```bash
python torch_transformer_benchmark.py --seq-len 2048 --batch-size 1 --attn-backend custom
```

Use `--attn-backend custom` rather than the default `auto`: it raises if the extension
fails to load instead of silently falling back to SDPA and looking slow.

---

Measured on an RTX 4050 Laptop (SM 8.9), CUDA 13.0, PyTorch 2.12.0+cu132. The GPU is power-
and thermally capped — `nvidia-smi` reports `SW Power Cap` and `SW Thermal Slowdown` active
under load, and absolute latencies drift 2–3x across a long session. Every ratio here comes
from interleaved timings; absolute milliseconds across runs are not comparable.
