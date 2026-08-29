# Optimizing a Transformer's Attention Layer

Everything measured, and the record of what was built, what broke, and what each fix was
actually worth. [README.md](README.md) covers installing, building and how the code works;
this file covers what it does and how well.

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
- [Notes and known limits](#notes-and-known-limits)
- [CUDA graphs](#cuda-graphs)
- [What's left](#whats-left)
- [Running](#running)
- [Environment](#environment)

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

### Two diagnoses that were wrong

**The 12-layer configuration was blamed on attention.** It used to fail the accuracy check,
and the explanation given at the time was that attention's precision was at fault. Measuring
disproved it: all four implementations land within 9% of each other there, *including the two
that never lose precision at all*. The drift comes from the other two-thirds of the model, so
no amount of work on the attention kernel would have fixed it. It was the wrong target.

**And then the failure itself turned out not to be real.** The check was being run against a
tolerance of 1 in 1,000 — but the graders' actual tolerance is 1 in 500, twice as loose. The
tighter number was written in this project's own documentation and in a comment at the top of
the grading script, both left over from before the problem statement was updated, while the
code underneath had always used the looser one. Re-run against the tolerance that actually
applies, the 12-layer configuration passes on every implementation, with **no failing values
at all** rather than one in a few million — as do the two other configurations that had been
recorded as borderline.

The lesson repeats one the CUDA sections below make three more times: a number that everything
downstream cites is worth checking against the thing it describes. The measurements here were
never wrong. They were being compared against the wrong line.

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
the card — completely idle, and on most shapes it was slower than a stock fused
attention.

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

| Shape | vs. scalar kernel |
| --- | --- |
| default (B8 H8 S128 D64) | 2.50x |
| default causal | 2.60x |
| default padded | 1.70x |
| seq 512 | 1.99x |
| seq 2048 | 1.85x |
| seq 2048 causal | 2.19x |
| head_dim 32 | 2.43x |
| head_dim 16 | 1.74x |
| wide (d_model 1024, 16 heads) | 2.40x |
| **total** | **2.00x** |

Every shape improves. An earlier independent run of the same table gave 2.07x, which is the
size of the run-to-run error bar.

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

| Configuration | custom scalar | custom wmma |
| --- | --- | --- |
| default (B8 S128 D512 H8 L6) | 0.900x | **1.007x** |
| padded (30%) | 0.846x | **1.001x** |
| seq_len 512 | 1.268x | **1.398x** |
| seq_len 2048 | 1.678x | **2.270x** |
| seq_len 2048, causal | 2.915x | **3.785x** |
| small (B1 S32) | 1.268x | **1.497x** |
| wide (d_model 1024) | 0.961x | **0.988x** |

Differences under ~10% here are not evidence of anything — a single harness run gives one
median with no interleaving, and `deep 12L` was measured at both 1.401x and 0.958x on two
runs. `seq_len 2048` and `seq_len 2048 causal` are well outside that band and agree with the
attention-op table.

**Where the tensor cores show up.** At `seq_len=2048` the model goes from 1.678x with the
scalar kernel to 2.270x with the tensor-core one; with causal masking, 2.915x to 3.785x.
Those are the configurations where attention is most of the runtime, and they are the ones
the kernel was written for.

**The three borderline configurations now pass outright.** `causal`, `causal + padded` and
`deep 12L` used to fail intermittently, by one or two elements out of millions, against an
`atol` of 1e-3. The harness's actual tolerance is **2e-3 / 2%**, and re-measured against it
all three pass on every backend with **zero** failing elements. See
[Accuracy](#accuracy) for the table and for what the old readings did and did not mean.

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
kernel peaked at `max_abs` 7.7e-4 and the tensor-core kernel at 8.7e-4. The harness metric is
set by a handful of cancellation outliers — trials routinely report `max_rel` in the hundreds,
meaning the worst element is one whose reference value is near zero — rather than by
systematic rounding, which is why halving the systematic error barely moves it.

### The three borderline configurations, re-measured

`causal`, `causal + padded` and `deep 12L` were previously documented here as sitting *on* the
accuracy gate, failing intermittently by one or two elements out of millions on every
kernel. **That was measured against `atol=1e-3`, which is not the harness's
tolerance.** The argparse defaults in `torch_transformer_benchmark.py` are `atol=0.002`,
`rtol=0.02`; the 1e-3 figure came from the module docstring and from `README.md`, both of which
were stale after the problem statement's tolerance update, and both are now corrected.

Re-run against the real gate — default shape otherwise (B8 S128 D512 H8), 8 trials each,
524,288 elements per trial:

| config | backend | verdict | worst `max_abs` | failed elements |
| --- | --- | --- | --- | --- |
| causal | custom scalar | **PASS** | 1.228e-3 | 0 / 4,194,304 |
| causal | custom wmma | **PASS** | 1.339e-3 | 0 / 4,194,304 |
| causal + padded | custom scalar | **PASS** | 1.228e-3 | 0 / 4,194,304 |
| causal + padded | custom wmma | **PASS** | 1.294e-3 | 0 / 4,194,304 |
| deep 12L | custom scalar | **PASS** | 1.094e-3 | 0 / 4,194,304 |
| deep 12L | custom wmma | **PASS** | 1.104e-3 | 0 / 4,194,304 |

Six for six, no failing elements anywhere. The worst reading in the set is the wmma kernel's
1.339e-3 on `causal`, which is a **33% margin** against the 2e-3 budget.

Two things from the old write-up survive the correction, and one does not.

**Survives:** the *magnitudes*. Every kernel really does land at `max_abs` ~1.0-1.3e-3 on
these shapes — on `deep 12L` they sit within 3% of each other. The spread across kernels is
smaller than the spread across seeds.

**Survives:** the reasoning about *why* the verdict was unstable when it was unstable. The
criterion is `abs <= atol OR abs <= rtol * abs(ref)`, so a failing element is one whose
reference value happens to be near zero — `max_rel` in the hundreds on passing runs is the
signature. That is why the backend with the largest `max_abs` could pass while a more accurate
one failed. It was never a precision ranking.

**Does not survive:** the conclusion. "Three configurations sit on the gate and fail
intermittently" was an artefact of grading against a tolerance twice as tight as the one the
harness actually applies. There is no borderline config on this hardware at the real gate.

## Coverage and limits

| dtype | head_dim 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | wmma | wmma | wmma | wmma | wmma | wmma |
| float16 | wmma | wmma | wmma | wmma | wmma | wmma |
| bfloat16 | wmma | wmma | wmma | wmma | wmma | wmma |

Every column is a kernel in this repository. `auto` used to apply a *preference* on top of
coverage — it asked which path was fastest, and handed head_dim 256 to
`F.scaled_dot_product_attention` — and that is gone: this project implements attention, so
`auto` now picks the first kernel that covers the shape and raises when none does. See
[head_dim 256](#head_dim-256-covered-by-everything-fastest-at-nothing).

Tensor cores need compute capability **8.0+**; below that the scalar kernel runs. Selection
is automatic — `--attn-impl` only exists to force a path for measurement.

The scalar kernel now covers the same five head_dims, so that second path is real at every
column rather than four of five. It used to stop at 64, because a thread holds q and the
output accumulator for its row in registers and `q_reg[128] + acc[128]` is 256 registers
against a hardware ceiling of 255 — a wall, not a tuning problem. Past 64 the row is split
between two adjacent lanes instead: 64 dims of each array per thread, one `__shfl_xor_sync`
per key to make the partial dot products whole, and everything after that — running max,
rescale, accumulate, normalise — done identically by both partners on their own half. One
shuffle per key, no other communication, and the same total shared-memory traffic, since the
two threads read disjoint halves of the key row. Measured `REG:168, LOCAL:0` at head_dim 128
float32: no spill. It also doubles the block to 128 threads, which lifts occupancy from two
warps per SM to twelve.

At `wide head_dim` (B2·H4·S64·D128) it runs 0.059 ms against wmma's 0.037, and lands 8.9e-7
from an fp32 reference where wmma lands 6.2e-4 — which is the point of having it. It is a
correctness reference and pre-Ampere coverage, not a competitor.

A side effect of routing every head_dim through one `ScalarCfg`: the launcher now declines
when a (dtype, head_dim) pair wants more shared memory than a block gets, instead of
launching and failing. float64 past head_dim 16 asks for 64 KB against a 48 KB budget and
used to do exactly that.

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
runs 0.041 ms against the 0.142 ms that shape used to cost, a 3.4x improvement that turned a
loss into a win. (That 0.142 ms was ATen, not the scalar kernel, which had no head_dim 128
specialisation at the time. Older revisions of this file attribute it to the scalar kernel
for the reason worked through in [The one
regression](#the-one-regression-and-why-it-was-not-the-kernel-it-was-blamed-on).)

### head_dim 256: covered by everything, fastest at nothing

Grading shape 8 (batch 64, seq 128, `d_model` 1024, 4 heads, causal) is head_dim 256, and it
was the one shape in the set no kernel covered: every forced `--attn-impl` refused it and
`auto` served it externally. All three families cover it now — scalar and wmma at
`{8,16,32,64,128,256}`, and the tile kernels at the same set.

Each needed something different:

* **scalar.** Past head_dim 64 a query row is already split across threads so `q_reg + acc`
  clears the 255-register ceiling; 256 makes that four threads instead of two. The real
  change is the score exchange — one `__shfl_xor_sync` only completes a *pair*, so it became
  a `log2(TPR)`-step butterfly over the group. Registers per thread are now identical at 64,
  128 and 256; what grows is the block (64 → 128 → 256 threads) and what shrinks is the key
  tile (`BLOCK_N` 64 → 32 → 16, holding the staged tiles at 32 KB).
* **wmma.** Q and O are register-resident across the whole head, so at 256 *no* block shape
  fits the 48 KB a block gets for free — Q, O and both K/V tiles are all `BLOCK_* × 256`.
  head_dim 256 is the one head_dim that opts into the larger shared-memory carveout, via one
  `cudaFuncSetAttribute` per instantiation. The shape was swept: `32x16` beats `16x16` at all
  ten shapes tried by 1.15x–1.55x, and `16x32` loses everywhere — the head_dim 128 result
  again, that a wider key tile buys nothing while Q is in registers.
* **tile.** A `BlockCfg<256>` per math mode and a dispatch case; the kernel was already
  generic in `HEAD_DIM`.

**And `auto` now takes head_dim 256.** It used to decline it, on the grounds that the
kernel was about half the speed of the alternative — which was a fair reading of the
measurement and is no longer a choice this project makes, because the alternative was a
prebuilt attention.

Two things changed after that decision. The first is the rule: there is no external
attention to defer to any more. The second is the kernel, and it moved a long way. The
occupancy diagnosis below was right — a 32-row block, **one** block per SM, about 4% of what
an SM can hold — and the direct-to-global epilogue is what fixed it. Storing O from the
accumulator fragments instead of staging the whole block tile through shared memory frees
16 KB at this head_dim, which is exactly the block that was missing: head_dim 256 went from
**1 resident block per SM to 2**, and the op itself gained **1.91x**.

What that leaves, measured on the two appendix shapes that used to be routed away, causal,
at their real dimensions:

| | op, was | op, now | end to end, 4 layers |
| --- | --- | --- | --- |
| shape 8 (head_dim 256, S 128) | ~0.50 | **0.82** | **0.992x** |
| shape 9 (head_dim 128, S 128) | 0.938 | **1.009** | **1.055x** |

(the op columns are against what the external baseline used to cost, kept because they are
the same measurement the old decision was made on.)

So routing both to the kernel costs 0.8% on shape 8 and *gains* 5.5% on shape 9 — the
head_dim 128 shape was being sent away for a loss that had stopped existing once fp32
tensors started contracting in fp16 fragments.

The structural point still stands and is worth keeping: this family of kernels holds Q and
the output accumulator in registers for the whole key loop, which is what makes it fast at
head_dim 64 and what costs it occupancy at 256. Freeing the O tile bought back one of the
blocks that costs; tiling the head dimension the way a production kernel does would be a
different kernel, not a different constant.

One gap is left deliberately: the tile kernels still do not cover head_dim **128**, as they
never did. Nothing measured wanted it; it is a `BlockCfg<128>` away.

## Notes and known limits

**A CUDA graph freezes more than the kernels.** Capture bakes in the kernel chosen for each
op, the cuBLAS algorithm, `allow_tf32` / matmul precision, and the extension's own runtime
knobs — `tile_set_split_kv` among them. `ATTENTION_BACKEND` and `ATTENTION_IMPL` are part of
the graph cache key so changing those re-captures, but the rest are invisible to a key, so
changing them after the first forward pass is silently ignored by a captured model. In
particular `verify_split_kv.py`'s trick of flipping the split flag in-process would do nothing
against one.

**The size gate is a measured constant, and it was measured on one machine.**
`_GRAPH_MAX_ACTIVATION = 524288` is where replay stopped beating eager on an RTX 3070. The
crossover is really the point where the GPU stops starving, which depends on the card's
throughput relative to how fast the host can feed it — so a faster GPU starves at larger
shapes and wants a *larger* value. `scripts/ab_graph.py --recommend` re-derives it on whatever
machine it runs on; the procedure is in
[README.md, Setup step 4](README.md#4-find-the-cuda-graph-gate-value-for-your-machine).

Getting it wrong is cheap either way: too low leaves latency on the table, too high pins
memory for nothing, and neither can produce a wrong answer, because replay is bit-identical to
eager at any setting. The gate is on activation volume rather than tokens because tokens
mispredict — at 512 tokens, `d_model` 256 measured 2.708x and `d_model` 512 measured 1.036x —
and not on `num_layers` at all, since 3/6/12/24 layers measured
1.031x/1.037x/1.040x/1.040x at fixed activation volume.

**Static graph inputs have to be normal tensors, not inference tensors.** The harness calls the
model inside `torch.inference_mode()`, and an inference tensor cannot be updated in place from
outside inference mode — so a static input buffer allocated there could never be refilled. The
buffers are allocated under `torch.inference_mode(False)` for that reason, and the capture body
additionally runs under `torch.no_grad()`: parameters have `requires_grad=True` and the model is
only `.eval()`ed, so escaping inference mode without it would build an autograd graph during
capture. Capture itself works fine from inside `inference_mode`; an earlier note claiming
otherwise was wrong.

**The cuTile shutdown crash was a DLL detach order problem, and it is fixed.** An access
violation (`0xC0000005`) raised *after* `main()` had returned its exit code, so a run looked
like it passed and then reported a nonzero status. Windows Error Reporting named the faulting
module every time, at the same offset — `nvgpucomp64.dll`, fault offset `0x53ad3e` — which is
the GPU compiler that ships with the display driver, not anything in CUDA or in this repo. It
is involved at all because a tile object carries a *tile IR* fatbin section alongside the SASS
(`cuobjdump -elf build/tile_attention.cuda.o` prints `Fatbin tile ir code: arch = sm_86`) that
the driver finalises when the module loads. That is also why `CUDA_MODULE_LOADING=EAGER` with
`--attn-impl wmma` faulted: loading the tile module was enough, launching a tile kernel was
never required.

Four things it was *not*, each ruled out by measurement: not context teardown
(`cudaDeviceReset()`, `cuDevicePrimaryCtxReset()` and freeing every tensor first all still
faulted); not a premature `FreeLibrary` (pinning with `GET_MODULE_HANDLE_EX_FLAG_PIN` did not
help); not an `atexit` handler (`os._exit()` did not dodge it, though `TerminateProcess` did,
which placed the fault after every line of Python); and not cuTile itself (a standalone `.exe`
linking `tile_attention.cu` directly, launching the same kernel, exits cleanly). That leaves
DLL detach order inside `python.exe`, and loading `nvgpucomp64.dll` *first* — so that it is
detached last — is what fixes it. Measured 5/5 clean exits with the preload against 0/5
without.

`kernel_ext.preload_tile_compiler()` does the load, and it runs at `kernel_ext` import
because it only works before `torch` is imported — preloading afterwards measures as no fix
at all. Every entry point that can reach a tile kernel therefore imports `kernel_ext` above
`torch`, `torch_transformer_benchmark.py` included, since `--attn-impl tile` is reachable
from it. `kernel_ext.tile_compiler_status()` reports whether the preload landed in time.

**The accuracy budget is dominated by the baseline's own rounding, not by the kernel.** The
baseline's *own* TF32 rounding sits 9.8e-4 (non-causal) to 1.2e-3 (causal) away from an exact
fp32 result — over half the `atol=0.002` budget before the optimized path has done anything at
all. Since the harness compares against the baseline's rounded output rather than ground truth,
being *more* mathematically correct does not help. Only closeness to the baseline's specific
rounding does. This is why restructuring GEMM order (for example fusing Q/K/V into one matmul)
moves the metric while being no less correct: it was measured to push `max_abs` from 9.9e-4 to
1.12e-3. At the real 2e-3 gate that particular change no longer costs a verdict, but the
principle stands — the headroom is what is left after the reference's own noise, and that noise
is the larger term.

**fp16 / bf16 are not winnable.** The bf16 baseline sits 6.1e-2 from exact fp32, with
153,627 of 524,288 elements failing the tolerance test *for a mathematically perfect
implementation*. No restructured attention can pass at those dtypes; the limit is the
harness's tolerance versus the baseline's own noise, not the kernel.

**The tile kernel is accuracy-first, not speed-first, on Ampere.** `csrc/tile_attention.cu`
is the same FlashAttention math expressed per *block* rather than per *thread*: tiles are
fixed-size arrays the whole block owns, `ct::matmul` is a matrix multiply of two of them,
and register allocation, the load schedule, bank-conflict avoidance and intra-block
synchronisation are the compiler's job. There is no `threadIdx` in the file. Two things cap
its speed on an RTX 3070, and neither is the model's fault:

- `ct::matmul` takes no rounding-mode argument: it dispatches purely on operand element
  type, and no tensor core does a true fp32 MMA. So `--attn-impl tile` — float operands —
  runs on the fp32 CUDA cores by construction. That is why it is the *most* accurate kernel
  here (~1e-6 versus ~1e-3) and the slowest. Narrowing the operands is the only lever, and
  both narrow modes pull it: `tile-tf32` reaches the tensor cores at wmma's own precision,
  `tile-bf16` goes further to 8 mantissa bits (~4e-3, enough to fail the accuracy gate on
  all but one measured config). bfloat16 and TF32 are the two narrow types cuTile
  accumulates into `float`; `__half` accumulates into `__half`, which attention cannot use.

  Reaching TF32 needed only `#include <cuda_tf32.h>`. `crt/cuda_tile.h` has no `#include`
  lines at all — it forward-declares `__half`, `__nv_bfloat16`, `__nv_fp8_*` and `__nv_tf32`
  and leaves completing them to the caller, exactly as this kernel already did for bf16. An
  earlier version of this claimed CUDA 13.3 does not define `__nv_tf32`; it does, in
  `include/cuda_tf32.h`. Confirm which units a mode got with
  `cuobjdump -sass build/tile_attention.cuda.o | grep HMMA`: fp32 kernels contain none.
- The TMA hardware the tile model is designed around is Blackwell-only; on Ampere the
  loads fall back to software-managed async copies.

Block shape matters far more here than in the hand-written kernels, and not smoothly: the
kernel keeps Q, O, K, V and the score tile live at once, and past a footprint threshold the
compiler spills and the cost jumps an order of magnitude. At `head_dim=64`, `BLOCK_N=16`
runs at 1.5 ms where `BLOCK_N=32` runs at 10.9 ms. The per-`head_dim` shapes in `BlockCfg`
were measured on SM 8.6 and are worth re-measuring on another architecture; the tables are in
[`csrc/TUNING.md`](csrc/TUNING.md).

**Tensor cores did not cost precision the way an earlier note predicted.** The worry was
that TF32 fragments would push the kernel away from the baseline. On the attention op the
opposite holds: because the harness runs the baseline with TF32 on, the tensor-core kernel
lands about 2x *closer* to it than the scalar kernel does (6.4e-5 vs 2.1e-4 at `seq2048`,
3.7e-4 vs 7.1e-4 at the default shape). End to end that does not translate — over 12
accuracy trials the scalar kernel peaked at `max_abs` 7.7e-4 and the tensor-core kernel at
8.7e-4 — but it does not reverse either.

**The "three configurations on the accuracy gate" finding was an artefact of a stale
tolerance, and is withdrawn.** `causal`, `causal + padded` and `deep 12L` were recorded here
as failing intermittently at `max_abs` ~1.0-1.2e-3 against `atol=1e-3`. The harness's real
tolerance is `atol=2e-3, rtol=2%`; re-measured against it, all three pass on every backend
with zero failing elements, worst case 1.339e-3 — a 33% margin. The table is in
[Accuracy](#accuracy).

What remains true is the *mechanism* that made those verdicts unstable while they were
unstable, because it still sets the metric: the baseline's own TF32 rounding sits 9.8e-4 to
1.2e-3 from exact fp32, the harness grades against that rounded reference rather than ground
truth, and the criterion's `OR` means the elements at risk are the ones whose reference value
is near zero. `max_rel` in the hundreds on a comfortably passing run is that signature. So the
error floor described throughout this section is real and unchanged — it simply sits inside a
budget twice as large as this document previously assumed.

`--no-allow-tf32 --matmul-precision highest` removes the TF32 rounding from both sides, which
is still the way to see the floor itself rather than the budget.

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

| | pairs | failed the gate *(at the old 1e-3 tolerance)* | identical to eager, bit for bit |
| --- | --- | --- | --- |
| eager | 48 | 2 | — |
| graphed | 48 | 2 | **48/48** |

Same two rows fail on both sides, with `max_abs` agreeing to the last digit and a gap of exactly
`0.0e+00` on every pair. Both failures were `causal+padded` (seeds 2 and 4), one element of
2,621,440 each.

Those two failures no longer occur: this run was graded against `atol=1e-3`, and at the
harness's real 2e-3 gate `causal+padded` passes on every backend with no failing elements (see
[Accuracy](#accuracy)). **That does not weaken the result here, it strengthens it** — the
column that matters is the right-hand one, and the point being tested was never whether a
config passes but whether capture changes the answer. The two sides agreed bit for bit on all
48 pairs *including* on the two rows that were then failing, which is the harder case to
agree on.

The useful conclusion is the negative one: **a graph cannot rescue an accuracy failure and
cannot cause one.** It is a latency switch, and the accuracy discussion elsewhere in this
document applies to it unchanged.

**Bit-exactness is asserted, not assumed.** Capture replays once against an eager reference and
refuses to install a graph whose output differs by anything at all. Nothing should ever trip
this — and nothing has — but a graph that computes something slightly different is the one
failure mode a benchmark would happily report as a *win*, which makes it worth one extra eager
forward per shape to rule out.

### Unrelated bug found on the way

The cuTile kernels used to crash the interpreter at shutdown with an access violation
(`0xC0000005` on Windows), *after* `main()` had returned its exit code. This had nothing to do
with graphs — it reproduced with `--cuda-graph off`, and `scripts/verify_split_kv.py` exited
the same way. It is fixed; the diagnosis is under [Notes and known
limits](#notes-and-known-limits). `scripts/verify_graph.py` still checks `scalar` and `wmma`
by default with `--include-tile` opting in, now because those are the two impls `auto` can
select rather than because the tile exit code could not be trusted.

## Reading q/k/v in place

Candidate 1 from the section below is now implemented, so this records what it was actually
worth rather than what it looked like it would be worth.

`MySelfAttention` runs one fused QKV projection and splits the result with a view and a
permute, which leaves q, k and v as three non-contiguous views over one packed tensor. Every
kernel used to call `.contiguous()` on all three: 18 clone kernels per forward pass, six
layers times three tensors. They now read the strides instead.

### Why it is nearly free

Work out what those strides actually are, which is the whole argument:

```
qkv                       [B, S, 3D]        (S*3D,  3D,   1)
.view(B, S, 3, H, Dh)     [B,S,3,H,Dh]      (S*3D,  3D,  H*Dh, Dh, 1)
.permute(2, 0, 3, 1, 4)   [3,B,H,S,Dh]      (H*Dh, S*3D,  Dh,  3D, 1)
.unbind(0)                q,k,v [B,H,S,Dh]  (S*3D,  Dh,   3D,  1)
```

The last axis still has stride 1. head_dim was never the problem — the copy was only ever
paying for a **row pitch**, `3*d_model` instead of `head_dim`. So every coalesced load, every
`wmma::load_matrix_sync` out of shared memory, every access that walks head_dim is untouched,
and exactly two things change inside each kernel: `bh_off` becomes `b*qs0 + h*qs1`, and the
row multiplier becomes a runtime `qs2`.

Coalescing survives for the same reason. One row at head_dim 64 is 256 B, which is two fully
used 128 B sectors however far apart the rows sit. What is given up is DRAM page locality
across rows, not transaction efficiency — which is why the measured cost is nothing.

The mask had already established the pattern: it has been passed as four explicit strides
with stride-0 broadcast dimensions since the beginning. q/k/v now follow it, minus the
head_dim stride, which is not carried because it is required to be 1.

### What it is worth

| | before | after |
| --- | --- | --- |
| leaf CUDA kernels per forward, `impl=auto` | 91 | **73** |
| of which the q/k/v clones (`elementwise_kernel`) | 18 | **0** |
| forward min latency, default shape | 3.609 ms | **3.403 ms** |

The 18 is exactly six layers times three tensors, and it goes to zero rather than down. The
tile path is a useful control: it went 97 to 79, keeping 6 `elementwise_kernel` launches that
are its own `to_bshd` repacks — the tile kernels cannot write layout 1 natively — so the only
thing that moved is the thing that was meant to.

Bit-exact, and checked as such rather than assumed. `scripts/verify_kernel.py` grew a
`packed` column that reruns every case through genuinely non-contiguous views and compares
with `torch.equal`, not with a tolerance: 57 of 57 rows exact, across 12 shapes and 5 impls.
The error columns are identical to the pre-change build, and the harness's own `max_abs`
stayed at `0.000736952`.

### The tile kernels could take it too

The first plan was to exempt them. `ct::tensor_span{q + bh_off, ct::extents{S, HD}}` bakes
the row pitch into the layout *type*, so a strided pitch looked inexpressible.

It is not. `crt/cuda_tile.h` has `ct::layout_right_padded`, which decouples the row pitch
from the last extent — precisely "row-major, last axis contiguous, rows arbitrarily spaced".
Build it as `ct::layout_right_padded_mapping{ct::extents{S, HD}, pitch}` and hand the mapping
to `tensor_span`. A plain `int` pitch deduces to a dynamic stride, an `integral_constant` to
a static one, the same trade the `HD` constant beside it already documents.

The K^T trick survives intact, which was the part worth checking. `layout_left_padded` pads
the *first* extent, so `extents{HD, S}` with pitch `p` puts element (d, j) at `k[j*p + d]` —
still K^T for free, still no shuffle through shared memory.

### What still copies

A layout this ABI cannot describe: head_dim not stride-1, or q/k/v that are not three slices
of one tensor. Those fall back to `.contiguous()` rather than widening the ABI. A clone costs
microseconds; a wrong address costs a wrong answer.

The trap on the way there was `torch::empty_like(qc)`, which allocates the output. It
defaults to `MemoryFormat::Preserve`, so once `qc` became a view it would have returned a
buffer carrying *q's* strides — which every kernel would then have written as though it were
packed, scattering the result into the wrong rows. It is spelled out as `torch::empty` now.
Nothing about that failure would have looked like a layout bug.

### The one regression, and why it was not the kernel it was blamed on

`scripts/ab_layout.py` times contiguous against packed inputs alternating call by call, with
a contiguous-vs-contiguous control row for the noise floor. Every custom kernel lands inside
its control at every shape — 0.97–1.05 against controls of 0.97–1.05 — free, as predicted.

One row was not: `wide hd` / `scalar`, at head_dim 128, which read **1.41x slower on strided
input** with a control of 1.006, and was written up here as the scalar kernel running out of
memory-level parallelism — `q_reg[128]` and `acc[128]`, 256 registers, two warps per SM,
nothing queued to hide a longer latency behind.

**That explanation described a kernel that did not exist.** `dispatch_head_dim` stopped at
head_dim 64 and always had (`git log -S "launch_kernel<scalar_t, 128>"` found nothing), for
the reason the register count itself gives away: 256 registers per thread is one more than
the hardware's 255-register ceiling. What the row was actually timing is ATen. Every forced
impl except `scalar` declined loudly on a shape it did not cover; `scalar` fell through to
`attention_aten` instead, so `impl=scalar` at head_dim 128 timed ATen and printed it under
the scalar kernel's name. The code comment beside the exemption had already named this exact
failure — "it can time ATen and label it 'scalar'" — and then deferred it as "a behaviour
change, not a cleanup".

The fix is to stop exempting it: forcing any impl now means that kernel or an error. `auto`
raises too — there is no longer any fallback for it to reach, external or otherwise. The
scalar kernel has since grown a real head_dim
128 — see [Coverage and limits](#coverage-and-limits) for the lane-pair split that fits it
under the register ceiling — so that row carries a genuine measurement again, and it reads
**1.000 against a 1.000 control**. The regression that was attributed to memory-level
parallelism at head_dim 128 does not exist in the kernel it was attributed to.

The 1.41x itself is real, and it is ATen's. Measured directly, with a contiguous-vs-contiguous
control at 1.007–1.021: ATen costs 1.41x on strided input at B8·H8·S128·D64, 1.42x at
B2·H4·S64·D128 and 1.50x at head_dim 96 — the uncovered head_dim that `auto` really does route
there — falling to 1.01x by seq 2048, where the matmuls are large enough to swamp the pitch.

Handing ATen `.contiguous()` copies instead does *not* collect that back: the clone measures
1.32–1.54x on the same three shapes, within noise of just passing the strided views and worse
at head_dim 96. The copy costs what the strided reads cost, so the views are passed through
as they are — a measured tie rather than an oversight.

Restructuring the staging loop row-outer, so the 64-bit row offset is computed once per row
instead of once per element, is kept: it is an absolute win at the head_dims the scalar kernel
actually has. Its comment claimed head_dim 128 too, and has been corrected.

### The measurement that had to be thrown away

The harness's median is blind to an effect this size, and not marginally. Three consecutive
runs of the **unchanged** build reported 1.217x, 1.525x and 1.554x. Its `min` column, over
those same runs, held to within 0.5% — 3.609, 3.611, 3.625 ms — and resolved the 6% cleanly
against 3.403, 3.406, 3.407 after.

Two earlier readings of the head_dim 128 row, taken while the card was throttled, said 2.23x
and 2.28x; their control rows read 1.148 and 1.026, which should have been enough to discard
them on the spot. A later one, on a card that had been busy for an hour, read **17.1x** with
a control of 1.000 — and that one is the warning worth keeping, because the control row does
*not* catch it. A ratio taken on a downclocked card stays honest about the ratio; what moves
is which side of the comparison the clocks favour, and a control that compares two *identical*
contiguous buffers cannot see that at all. The absolute times were the tell: `scalar` at
`long seq` read 8.19 ms in that run and 2.76 ms once the card was cool, on an unchanged
binary. On a quiet card the same row reads 1.42x, three times over. The control row is not
decoration, but it is not sufficient either — see also [Absolute speedups are unusually
sensitive to host contention](#absolute-speedups-are-unusually-sensitive-to-host-contention).

## Attention in fp16

The wmma kernel contracted fp32 q/k/v in **tf32** fragments. It now contracts
them in **fp16**. The tensors stay fp32; only the shared tiles and the fragments
narrow, and the output stays fp32 because it feeds `out_proj`, which is a cuBLAS
fp32 GEMM. A `compute_t` template parameter splits "what the tensor is" from
"what the fragments contract in".

### This is not a precision-for-speed trade

tf32 is 1 sign + 8 exponent + 10 mantissa. fp16 is 1 + 5 + 10. **Same mantissa.**
Only the exponent range differs, and post-LayerNorm activations use none of it.
So the predicted accuracy cost is zero, and the measurement says zero -- not
"close enough", identical. The harness's own `max_abs`, five shapes, causal:

| | d32 | d256 | S1024 d256 | d512 | d512 h4 |
|:--|:--|:--|:--|:--|:--|
| tf32 | 0.00131567 | 0.00132135 | 0.00113010 | 0.00123608 | 0.00119376 |
| fp16 | 0.00131567 | 0.00132135 | 0.00113010 | 0.00123608 | 0.00119376 |

Every digit. Against an fp64 reference on the attention op alone, the two agree
to three significant figures at all eleven shapes swept.

That is worth stating carefully, because the same argument does **not** extend
one step further. bf16 has 8 mantissa bits, and under the *loosened* 2e-3 gate
it measures 425%-622% of budget with tens of thousands of failing elements. The
mantissa is the whole argument; drop two bits and it collapses.

### What it buys

On this card fp16 tensor cores measure **2.0x-2.25x** tf32 (39.7 against 17.7
TFLOPS at N=2048), and a 16x16x16 fp16 fragment contracts twice the K of tf32's
16x16x8, so the mma count halves too.

On the attention op, causal, graph-timed, interleaved, control ±0.4%:

| shape | tf32 | fp16 | ratio |
|:--|--:|--:|--:|
| B8 H4 S128 d8 | 11.3 | 9.3 | 1.210x |
| B16 H8 S128 d32 | 38.8 | 31.8 | 1.221x |
| B8 H8 S1024 d32 | 543.0 | 408.0 | 1.331x |
| B8 H8 S128 d64 | 37.7 | 29.4 | 1.283x |
| B1 H8 S2048 d64 | 595.8 | 426.6 | 1.397x |
| B4 H8 S512 d128 | 706.2 | 459.6 | 1.536x |

Geometric mean **1.328x** over eleven shapes.

End to end the geometric mean is **1.039x**, ranging from 1.014x to **1.123x**
at `B8 S1024 d256`. The gap between 1.33x and 1.04x is Amdahl and nothing else:
attention is 20% of the forward pass at `B16 S128 d256` and 46% at `S1024`,
which is exactly where the two ends of the table land. The harness agrees from
its own independent measurement -- `--seq-len 2048 --batch-size 1 --causal` went
**4.589x to 5.231x**.

### head_dim 128 changes verdict

The tf32 kernel *lost* to a stock fused attention at head_dim 128, which is why `auto` capped the
kernel at head_dim 64 and it never ran there. In fp16 it wins — except at
exactly S 128, which reproduced across runs:

| S | 64 | 128 | 256 | 384 | 512 | 1024 |
|:--|--:|--:|--:|--:|--:|--:|
| external / wmma | 1.552x | **0.938x** | 1.028x | 1.027x | 1.047x | 1.081x |

So the dispatch rule admits head_dim 128 from S 512 up, where the margin is
4.7%-8.1% against a ±0.4% control. S 256 and 384 win too, by 2.8%, and are
deliberately not claimed — that is close enough to the noise floor that
widening the rule for it would be claiming more than was measured. Leaving S 128
there is the important part: it is the sequence length of eleven of the
fourteen grading shapes.

### What did not work

Narrowing `compute_t` frees about a third of the block's shared memory, so block
shapes become affordable that fp32 cannot fit — at head_dim 128, 32x32 costs
41.4 KB in fp16 where only 32x16's 29.9 KB fits in fp32. Spending the slack on a
wider key tile is the obvious next move, and it loses badly: **0.537x** geomean
at head_dim 128, 0.812x for the equivalent widening at head_dim 64.

A wider `BLOCK_N` doubles the K/V staging and the score tile without adding any
parallelism — 41.4 KB drops the SM from three resident blocks to two, and each
block then does twice the work per key-tile iteration. Q is already
register-resident, so there is nothing for the extra pressure to buy. The fp16
win at head_dim 128 is entirely the fragments, not the tile. `WmmaShape` was
parameterised on the element width to run that sweep and then un-parameterised
again, on the criterion the source already stated: a different winner per dtype
is what would justify the parameter, and there is none.

### Still open, with a number on it

This narrows on the way *into* shared memory, so DRAM traffic is unchanged.
Narrowing the tensors themselves would halve it, and handing the kernel fp16
tensors directly measures **21.4 us at B8 H8 S128 d64 against 29.4** — a further
1.37x on the op. It is not free: q/k/v come out of the fused qkv GEMM, so that
GEMM needs an fp16 output epilogue, and the kernel needs an `out_t` so a
half-input call can still write the fp32 `out_proj` consumes. Without the
`out_t` the output rounds to fp16 and the op's error moves from 1.59e-3 to
1.83e-3, so the "free accuracy" argument above stops applying.

## Fused Linear + GELU, in fp16

The FFN's first layer is `GELU(x @ W1^T + b1)`. Run as two ops, the GELU pass
reads the whole `[batch*seq, ffn_dim]` GEMM result back out of global memory,
applies one cheap function, and writes it again. Folded into the GEMM's
accumulator it is free -- and at the grading appendix's shapes, where
`ffn_dim == d_model`, that pass is a third of the pair.

The kernel to do it -- `gemm_bias_gelu_kernel`, a 4x2 warp grid over a block
tile, register-prefetched staging -- had been sitting in the tree as **dead
code** on the verdict "loses to cuBLAS at every shape the model uses, 0.88x".
Two changes turned that around, and the second was worth more than the first.

### Why it was written off, and why that was wrong

**A kernel rejected on a benchmark is rejected on that benchmark's shape.** The
0.88x was measured at the harness default, where `ffn_dim = 4*d_model`. Every
appendix shape has `ffn_dim == d_model`.

**A DRAM-traffic model is void when the operands fit in L2.** The 128x128 tile
was *chosen*, not tuned: a BMxBN tile does `BM*BN*K` MACs per `(BM+BN)*K` floats
loaded, so 64x64 sustains 32 MAC/float, capping the kernel at ~7 TFLOPS against
448 GB/s. Correct arithmetic that assumes the loads reach DRAM. At `d_model` 256
the weight is 256 KB and never leaves the 3070's 4 MB L2. What binds instead is
the grid: at `M=128, N=256` a 128x128 tile is **2 blocks on a 46-SM card**, and
it measured 0.198x of cuBLAS there.

**An op benchmark at small M measures PyTorch dispatch, not the kernel.** Eager,
`F.linear` at `M=128, K=N=256` reads 32.3 us -- and at `M=1024`, eight times the
work, 30.4 us. Both are ~27 us of dispatch, which the model has already removed
by capturing a CUDA graph. Everything below is graph-timed.

Re-tiled to 64x64, tf32 went from losing everywhere to winning at 6 of 13
shapes, and needed a gate to keep it off the other 7.

### fp16: the same accuracy, twice the tensor cores

TF32 and FP16 both carry a **10-bit mantissa**, so with an fp32 accumulator they
are equally accurate. Not approximately -- measurably. Rounding operands to each
and scoring against an fp64 reference:

| | attention op | FFN GEMM | whole 6-layer model |
|:--|--:|--:|:--|
| tf32 | 1.63e-03 | 1.22e-03 | 9.82e-04 -- 49% of the 2e-3 budget |
| fp16 | 1.63e-03 | 1.22e-03 | 9.82e-04 -- 49% of the 2e-3 budget |
| bf16 | 1.18e-02 | 1.02e-02 | 9.70e-03 -- 485%, **12339 elements fail** |

What differs is throughput. On this card fp16 tensor cores measure **1.98x-2.25x**
tf32 (39.7 vs 17.7 TFLOPS at N=2048). An fp16 fragment also contracts 16 elements
of K against tf32's 8, halving the mma instruction count, and the narrowing moves
from per-fragment-load to once on the way into shared memory.

bf16 was already believed dead, but on a measurement taken against the *old*
1e-3 gate. The loosened 2e-3 / 2% tolerance was a reason to re-ask; the answer
did not change.

### The result: no gate left

Graph-timed against cuBLAS + `F.gelu`, best tile per precision, control ±1.0-2.9%:

| M | K | N | tiles64 | tf32 | fp16 | best tile |
|---:|---:|---:|---:|--:|--:|:--|
| 32 | 256 | 256 | 4 | 0.99x | **1.634x** | 64x32 |
| 128 | 64 | 64 | 2 | 1.36x | **1.665x** | 64x32 |
| 128 | 256 | 256 | 8 | 0.91x | **1.327x** | 64x32 |
| 512 | 256 | 256 | 32 | 0.71x | **1.240x** | 64x32 |
| 1024 | 32 | 32 | 16 | 2.10x | **2.402x** | 64x32 |
| 1024 | 256 | 256 | 64 | 0.86x | **1.425x** | 64x32 |
| 2048 | 256 | 256 | 128 | 1.66x | **3.188x** | 64x64 |
| 8192 | 256 | 256 | 512 | 1.34x | **2.534x** | 64x64 |
| 16384 | 256 | 256 | 1024 | 1.18x | **2.139x** | 64x64 |
| 320000 | 256 | 256 | 20000 | 1.03x | **1.919x** | 64x64 |
| 1024 | 512 | 512 | 128 | 1.19x | **2.448x** | 64x64 |
| 1024 | 1024 | 1024 | 256 | 0.86x | **1.846x** | 64x64 |
| 512 | 2048 | 2048 | 256 | 0.78x | **1.638x** | 64x64 |
| 1024 | 512 | 2048 | 512 | 0.82x | **1.707x** | 64x64 |
| 2048 | 4096 | 4096 | 2048 | 0.62x | **1.371x** | 128x128 |

Grids from 2 tiles to 20000, K and N from 32 to 4096, and **fp16 wins every
row** by 1.24x to 2.40x. So the shape gate tf32 needed is gone entirely; what
remains is the coverage check and one precondition.

`pick_gemm_tile()` keeps two thresholds, which are different constraints. Below
two full waves of blocks, 64x32 doubles the grid and wins. Past L2 -- K=N=4096,
a 64 MB weight -- the traffic argument finally does bite and 128x128 wins, one
measured point beyond any shape this model issues.

**The precondition fp16 has and tf32 does not:** operands must fit fp16's range.
Post-LayerNorm activations and `O(1/sqrt(d))` weights satisfy it by construction
here; `--linear-gelu tf32` and `off` are the escape hatches.

### End to end, with the two changes scored separately

`scripts/ab_linear_gelu.py` times all three settings in one process, interleaved,
with `off` at both ends of every round as the control. Geometric means over
eleven appendix shapes:

- fusion alone (`off` → `tf32`): **1.008x**
- fp16 alone (`tf32` → `fp16`): **1.058x**
- together (`off` → `fp16`): **1.066x**

Best shapes: 1.115x at `d_model` 32, 1.101x at S=1024, 1.091x at B16, 1.090x at
B128. Nothing regresses beyond its control.

That first line deserves reading carefully. **The fusion on its own is worth
nothing now** — because `tf32` here is ungated, so it runs at the seven shapes
where tf32 loses. The gated tf32 configuration that shipped first measured
1.032x over these same eleven shapes. fp16 roughly doubles the payoff *and*
removes the gate that tf32 needed to reach even that.

One row disagrees with the op sweep: `B64 S128 d256` reads fp16 at 0.987x of
tf32 where the op says 1.892x, against a 1.5% control. It is the shape whose
activation volume declines CUDA-graph capture, so it is timed eagerly, but that
is a hypothesis rather than an explanation. Total against `off` is still 1.064x.

### Accuracy

Every shape PASSes the harness. fp16 costs about 1e-4 of extra end-to-end error;
the tightest grading shape sits at 1.35e-3 against the 2e-3 gate, a **~33%
margin**. `verify_kernel.py`, `verify_graph.py`, default, `--causal` and
`--padding-ratio 0.3` all still PASS.

What was given up to get it: with tf32 the kernel was **bit-identical** to
`F.linear + F.gelu` wherever cuBLAS picked a tf32 kernel with the same k-order.
fp16 is not, and cannot be. `--linear-gelu tf32` buys that property back at half
the tensor-core rate.

## What's left

Everything above is attention. This section is the other direction: given a kernel that is
already ahead of a stock fused attention on every shape, what is still worth doing to the
model around it?

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
| ~~q/k/v `.contiguous()` copies~~ | ~~5.6%~~ | ~~3.6%~~ | ~~3.6%~~ | ~~1%~~ |
| idle — CPU launch gaps | ~6% | ~0% | ~1% | **~80%** |

The struck row is gone as of [Reading q/k/v in place](#reading-qkv-in-place). It is left in
because the rest of the table is the profile the ranking below was decided against, and
quietly deleting a row would misrepresent what that ranking saw.

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
  `atol=2e-3 / rtol=2%`.
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

**Done: the q/k/v `.contiguous()` copies.** Was candidate 1 here, estimated at 5.6% and
bit-exact; measured at 18 clone kernels removed per forward pass and ~6% on min latency,
bit-exact. Absorbed by reading strides, which cost nothing because head_dim was already
stride-1 — the copy had only ever been paying for a row pitch. See [Reading q/k/v in
place](#reading-qkv-in-place).

**1. Fused Linear+GELU — ~5% available. Collected, by hand-writing the GEMM.**

PyTorch exposes cuBLASLt's bias+GELU epilogue as `torch._addmm_activation`. Measured at the
default FFN shape:

```
F.linear + F.gelu(erf)          188.7 us
torch._addmm_activation fused   184.6 us    <- 2%; cuBLASLt picks a slower GEMM algo
F.linear alone                  138.9 us    <- so GELU itself is only ~50 us
```

It is also the **tanh** approximation, not erf: 3.87e-4 against an erf reference. Against an
end-to-end `max_abs` that already peaks at 8.7e-4, that is roughly a third of the remaining
margin to the 2e-3 gate — not disqualifying on its own, as an earlier version of this note
claimed under the mistaken 1e-3 budget, but a real cost for a change worth 2%. So the cheap
route was rejected on the speed, and the expensive one — a TF32 tensor-core GEMM with a
GELU epilogue that beats cuBLAS — was taken instead, and then moved from tf32 to fp16
fragments for a further 2x on the tensor cores at identical accuracy. Together they are
worth **1.066x end to end** (geometric mean, up to 1.115x) on the grading appendix's
shapes. See [Fused Linear + GELU, in fp16](#fused-linear--gelu-in-fp16).

**2. INT8 / W8A8 — available on this hardware, dead on accuracy.** SM 8.6 has INT8 tensor
cores. But bf16 — far finer than INT8 — fails the gate on its own, and that has now been
re-measured against the *current* 2e-3 / 2% tolerance rather than the old 1e-3 one: 485% of
the budget at B16 S128 d256, with 12339 of 524288 elements failing. The reduced precision
that does pay is fp16, which is not a step down from tf32 at all — same 10-bit mantissa,
twice the tensor-core rate. See [Fused Linear + GELU, in fp16](#fused-linear--gelu-in-fp16).

**3. RMSNorm — disqualified above, and it would not pay.** The argument for it is one fewer
pass over memory; `fused_add_layernorm` already removed that pass. The mean subtraction is
arithmetic on data that is already in registers, competing for a fraction of 6.5%.

**4. SwiGLU — disqualified above.**

**5. FP8 — impossible here.** FP8 tensor cores start at Ada (SM 89). This card is SM 8.6.

### What that adds up to

Both collectable items are collected. Graphs took default from 1.263x to 1.353x and `small`
from 2.49x to 11.23x; the q/k/v copies were worth a further ~6% on top of that. GELU at ~5%
was the one item left, and hand-writing the GEMM did collect it — but only where the shape
suits a 64x64 tile, which is most of the grading appendix and none of the default's 4x FFN.

So **the remaining opportunity is cuBLAS**, which is two thirds of the forward pass at
default and four fifths at `wide`. The GELU fusion is a first bite at exactly that, and the
shape of the result is instructive: this kernel beats cuBLAS by up to 1.67x on the small
GEMMs where cuBLAS is far from its own roofline, and loses by 0.78x-0.84x as soon as the
operands stop fitting in L2 and cuBLAS switches to a tile this one cannot match. Beating
cuBLAS *in general* is still not a weekend.

The three wins landed in different places, which is the useful shape of the result rather
than an accident: the attention kernel pays on long sequences, graphs pay when the GPU is
starved at small shapes, and the q/k/v copies paid a flat few percent everywhere the
attention op runs at all. They cover different regimes instead of competing for the same
time.

The usual caveat from [Performance](#performance) applies to the small numbers here. The
graph figures are the exception — `scripts/ab_graph.py` times both sides interleaved in one
process and prints an eager-vs-eager control row, which put the noise floor at ±1.2%, so the
1.029x at default sits just above it rather than inside it. The harness's own single-run
median cannot see that effect at all.

## Running

### The benchmark harness

```bash
python torch_transformer_benchmark.py
```

Runs the accuracy check first and **skips benchmarking entirely if it fails** (exit code 2),
so a passing speedup number always means a correct implementation.

Useful flags — the graders sweep these:

```bash
python torch_transformer_benchmark.py --seq-len 2048 --batch-size 1 --causal
```

`--batch-size --seq-len --d-model --heads --ffn-dim --layers --causal --dtype
--padding-ratio --input-scale --atol --rtol`

`--attn-fp16 {auto,tf32}` controls whether the wmma attention kernel contracts fp32 q/k/v in
fp16 fragments, `auto` by default. fp16 and tf32 carry the same 10-bit mantissa, so this is a
speed switch and not an accuracy one — the harness's `max_abs` comes out identical to every
digit either way. See [Attention in fp16](#attention-in-fp16).

`--linear-gelu {auto,tf32,off}` controls whether the FFN's first Linear and its GELU run as
one kernel, `auto` by default. `auto` uses fp16 fragments, which beat cuBLAS at every shape
measured; `tf32` runs the same kernel at half the tensor-core rate and is bit-identical to
`F.linear + F.gelu` where cuBLAS picks a matching tf32 kernel; `off` is cuBLAS plus a
separate GELU. The alternatives exist because this is the one optimization that can move the
accuracy number, though the margin at the tightest grading shape is ~33%. See
[Fused Linear + GELU, in fp16](#fused-linear--gelu-in-fp16).

`--cuda-graph {off,auto,always}` controls CUDA graph capture, `auto` by default. `always`
ignores the size gate and exists so `scripts/ab_graph.py` can measure the shapes `auto`
declines; it is not for benchmark runs. Replay is bit-identical to eager, so this is purely a
latency switch — see [CUDA graphs](#cuda-graphs).

To use the CUDA kernel, run through `devenv.bat` so the build can find `cl.exe`:

```bash
cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
```

`--attn-backend custom` is the only backend, and it raises if the extension fails to load
rather than quietly measuring something that is not this project.

To see the kernel rather than the FFN, benchmark where attention dominates:

```bash
python torch_transformer_benchmark.py --seq-len 2048 --batch-size 1 --attn-backend custom
```

### Choosing the attention backend

Edit `ATTENTION_BACKEND` in [`optimized/config.py`](optimized/config.py), or pass
`--attn-backend` for a single run:

| Value | Behavior |
| --- | --- |
| `custom` | Require the CUDA kernel; raise if it is unavailable, so a broken build fails loudly instead of quietly benchmarking something else and looking slow. |

**That is the only value.** There used to be two more — `auto`, which fell back to
`F.scaled_dot_product_attention` when the extension would not load, and `sdpa`, which always
used it — and both were routes to a prebuilt attention living inside the submission. A
switch that quietly stops implementing the thing the project exists to implement is worse
than no switch, so neither survives.

To A/B attention alone, vary `--attn-impl`. Same shape, same everything else:

| shape 8 | speedup | attention math |
| --- | --- | --- |
| `--attn-impl auto` | 1.308x | picks wmma, the first kernel that covers head_dim 256 |
| `--attn-impl wmma` | 1.308x | tensor-core kernel |
| `--attn-impl scalar` | 1.049x | scalar kernel |
| `--attn-impl tile-fp16` | 0.714x | cuTile kernel |

`auto` and `wmma` now agree at this shape, which they did not before: `auto` used to read
1.420x here by declining every kernel and serving the case externally.

### Choosing the kernel inside the custom backend

`ATTENTION_IMPL` (or `--attn-impl` for one run) picks which of the custom kernels handles
attention:

| Value | Behavior |
| --- | --- |
| `auto` | Tensor-core kernel where it applies, scalar kernel otherwise. |
| `scalar` | Force the scalar kernel. No tensor cores, no TF32 rounding. Six tuned instantiations at `head_dim` in {8,16,32,64,128,256}; anything else drops to the generic scalar kernel in the same file, which takes `head_dim` 1 to 2048 in every dtype. Either way it is this project's own code, so ATen still cannot be timed under its name. |
| `wmma` | Force the tensor-core kernel; raises on shapes it does not cover, so a silent substitution cannot be mistaken for a slow kernel. |
| `tile` | Force the cuTile kernel — the same math written against the CUDA tile programming model instead of per-thread. float32, `head_dim` in {8,16,32,64,128,256}. Raises rather than falling back. |
| `tile-tf32` | The same cuTile kernel with its two GEMMs narrowed to TF32, which is what puts them on the tensor cores. Same arithmetic `wmma` uses for fp32 inputs and the same ~1e-3 accuracy, so it clears the harness gate wherever `wmma` does. The tensor-core tile mode to reach for first. |
| `tile-bf16` | As above but narrowed to bfloat16 — 8 mantissa bits. Marginally faster than `tile-tf32` on some shapes and far less accurate; fails the harness gate on most configs. |
| `tile-fp16` | The same cuTile kernel narrowed to float16 — tf32's 10 mantissa bits at twice the tensor-core rate, accumulated in fp32. The fastest tile mode, and it clears the harness gate wherever `tile-tf32` does. |

The tensor-core kernel covers `head_dim` 8, 16, 32, 64, 128 and 256 in float32, float16 and
bfloat16, on compute capability 8.0 and up — every head_dim the harness can produce, since
`d_model` is divisible by `num_heads`. Nothing falls through to ATen any more. Covering a
head_dim and being the right thing to run for it are different questions; see
[head_dim 256](#head_dim-256-covered-by-everything-fastest-at-nothing) for the one place they
come apart.

A `d_model / num_heads` outside that set — 96, 100, 192, 512 — is not something the harness
produces, but it is something a caller can ask for, and it used to raise. The scalar launcher
now ends in a generic kernel (`csrc/attention_scalar.cuh`) that makes `head_dim`, the per-row
thread count and the key tile runtime values instead of template parameters, so it takes
anything from 1 to 2048 in any dtype. It sits behind the six tuned instantiations inside
`launch_scalar` rather than as a third entry in `auto`'s list, because the two are one
kernel's worth of coverage and a caller picking "the scalar kernel" is not choosing between
them. It is a coverage floor rather than a preference: it runs only for a head_dim the tuned
paths decline, and forcing it onto a head_dim they do cover returns bit-identical results
1.0x–5.3x slower. Measured end to end against the baseline at shapes that previously
raised: head_dim 100 **3.34x**, head_dim 192 **1.67x**, head_dim 96 **1.25x**, head_dim 512
**0.85x** — the last one a genuine loss, since a 512-wide head is where cuBLAS's batched GEMM
beats a CUDA-core kernel outright. `scripts/verify_generic_scalar.py` checks all of it, and
`SCALAR_FORCE_GENERIC=1` is the A/B knob.

Neither tile mode is ever chosen by `auto`: they are a separate programming model whose
performance you should opt into deliberately. They need a build that found CUDA 13.3+;
without one, `--attn-impl tile` raises instead of silently running something else. See
[Notes and known limits](#notes-and-known-limits) for why plain `tile` is the most accurate
and the slowest kernel here.

### Helper scripts

| Script | Purpose |
| --- | --- |
| `scripts/verify_kernel.py` | Every kernel against a float64 reference across 14 shapes. Fails fast and names the shape that broke. The `packed` column reruns each case through the non-contiguous views the model actually produces and demands a *bit-identical* result, not one within tolerance. It also exports `reference_attention_f64`, which the A/B scripts use as their yardstick — one expression evaluated the same way every time, rather than a library call that picks a backend per shape. |
| `scripts/verify_generic_scalar.py` | The generic scalar catch-all across 14 uncovered head_dims (1 to 2048), four dtypes and six mask/shape combinations, against the same float64 reference. Run again with `SCALAR_FORCE_GENERIC=1` and it also serves the six head_dims the specializations own, which is how the generalization is checked to be the same arithmetic and not merely a close one. |
| `scripts/verify_split_kv.py` | Checks the tile kernel's split-KV path against its own single-pass path, and asserts the split actually fired. |
| `scripts/verify_graph.py` | Checks that graph replay is *bit-identical* to eager — tolerance exactly zero — and that the graph actually fired rather than silently declining. `--test-failure` also exercises the capture-failure path; `--include-tile` adds the cuTile impls. |
| `scripts/bench_attention.py` | Times the attention op alone — scalar vs. tensor-core vs. the tile kernels — with accuracy alongside, so a speed win bought with precision is visible. |
| `scripts/compare_backends.py` | Runs the full harness once per backend and prints the comparison table above. Set `COMPARE_FULL=1` for the harness's own accuracy-trial count instead of the trimmed one. |
| `scripts/ab_split_kv.py` | A/Bs the split-KV path against single-pass, interleaved in one process with a control group. |
| `scripts/ab_layout.py` | A/Bs contiguous q/k/v against the packed views the model produces, alternating call by call, with a contiguous-vs-contiguous control row. Measures what reading strides costs, not what skipping the clones saves. |
| `scripts/ab_graph.py` | A/Bs eager against graph replay, interleaved with an eager-vs-eager control row for the noise floor. `--recommend` measures the crossover on the machine it runs on and prints the `_GRAPH_MAX_ACTIVATION` to set, refusing to answer if the control rows say the machine is too noisy to trust. |
| `scripts/tune_block_shapes.py` | Sweeps block shapes for every attention backend (wmma and the three tile math modes), per mask mode where the kernel can carry one. |
| `scripts/sass_mix.py` | SASS instruction mix and occupancy for the head_dim 64 kernels. |

`kernel_ext.py` puts MSVC on `PATH` itself, so these do not strictly need the `devenv.bat`
prefix; it remains for cases that need `cl.exe` before Python starts.

```bash
cmd.exe /c scripts\devenv.bat python scripts\compare_backends.py
```

The measurements behind the kernels' block shapes and thresholds — and the two rules every
one of them follows — are in [`csrc/TUNING.md`](csrc/TUNING.md).

---

## Environment

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB, SM 8.9 |
| Driver | 595.79 |
| CUDA Toolkit | 13.0 (V13.0.48) |
| PyTorch | 2.12.0+cu132 |
| Python | 3.11 |
| Compiler | MSVC 14.44, pinned via `-vcvars_ver` |
| OS | Windows 11 |

> **Mixed provenance, not yet reconciled.** The tables above were taken across two machines.
> The RTX 4050 Laptop in this table is the original development machine. Later work — the
> tile-kernel comparison, everything in [`csrc/TUNING.md`](csrc/TUNING.md), the split-KV
> measurements and the whole of [CUDA graphs](#cuda-graphs) — was measured on an **RTX 3070
> (8 GiB, SM 8.6, 46 SMs, driver 610.47, CUDA 13.3, PyTorch 2.12.0+cu132, Python 3.10.6)**,
> and those sections say so where it matters. Ratios within one section are internally
> consistent; ratios or milliseconds drawn across sections are not comparable. One consequence
> worth flagging: the 4050 is Ada (SM 8.9) and *does* have FP8 tensor cores, so the note in
> [What's left](#whats-left) calling FP8 unavailable applies to the 3070 only.

The GPU is a power- and thermally-capped laptop part: `nvidia-smi` reports
`SW Power Cap` and `SW Thermal Slowdown` active under load, and absolute latencies drift by
2-3x across a long session. Every ratio quoted here therefore comes from timings taken
*interleaved* — candidates measured round-robin, minimum or best-of-N — so the clock state is
shared between them. Absolute milliseconds from different runs are not comparable; ratios
within one run are.

Host load matters more than is comfortable for anything launch-bound, which the
[CUDA graphs](#cuda-graphs) section covers: a game running in the background moved the default
config's reported speedup from 1.39x to 2.08x with no code change.

`torch.compile` and Triton are unavailable in this environment (Triton has no working
Windows build here), which is why the custom-kernel path is C++/CUDA via
`torch.utils.cpp_extension` rather than Triton — and why CUDA graphs are captured by hand
rather than through `--compile-mode reduce-overhead`.
