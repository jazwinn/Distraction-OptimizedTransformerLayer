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

## Reproducing

```bash
cmd.exe /c scripts\build_ext.bat     # build once (optional; a plain run builds too)
python scripts\verify_kernel.py      # every kernel, 12 shapes
python scripts\bench_attention.py    # attention-op table
python scripts\compare_backends.py   # full harness sweep
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
