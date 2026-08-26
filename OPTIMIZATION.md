# Running Attention on Tensor Cores

How the fused attention kernel in [`csrc/fused_attention.cu`](csrc/fused_attention.cu) was
moved off scalar FMA and onto the GPU's tensor cores, what each step was worth, and where
the win does and does not show up end to end.

- [The starting point](#the-starting-point)
- [The change](#the-change)
- [What each step was worth](#what-each-step-was-worth)
- [Performance](#performance)
- [Why the harness number looks small](#why-the-harness-number-looks-small)
- [Accuracy](#accuracy)
- [Coverage and limits](#coverage-and-limits)
- [Reproducing](#reproducing)

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
| float32 | scalar | wmma | wmma | wmma | ATen |
| float16 | scalar | wmma | wmma | wmma | ATen |
| bfloat16 | scalar | wmma | wmma | wmma | ATen |

Tensor cores need compute capability **8.0+**; below that the scalar kernel runs. Selection
is automatic — `--attn-impl` only exists to force a path for measurement.

**`head_dim=128` is the notable gap.** The tensor-core kernel would need a `64x132` fp32
output tile plus two `32x132` K/V tiles in shared memory, over the 48 KB that keeps two
blocks resident per SM. Covering it means a different tiling — splitting `head_dim` across
warps instead of giving each warp the full width — not a parameter change.

## Reproducing

```bash
cmd.exe /c scripts\build_ext.bat                                  # build once
cmd.exe /c scripts\devenv.bat python scripts\verify_kernel.py     # both kernels, 12 shapes
cmd.exe /c scripts\devenv.bat python scripts\bench_attention.py   # attention-op table
cmd.exe /c scripts\devenv.bat python scripts\compare_backends.py  # full harness sweep
```

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
