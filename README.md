# Optimizing a Transformer Layer

An optimized Transformer layer for **TikTok TechJam 2026, Problem Statement 3** — accelerate a
standard Transformer layer with custom GPU kernels while staying inside a strict
per-element accuracy budget.

The work is in `UserOptimizedTransformer` and the modules it builds on, all inside
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py). Attention runs through
one of two interchangeable backends: PyTorch's `scaled_dot_product_attention`, or a
hand-written fused CUDA kernel in [`csrc/fused_attention.cu`](csrc/fused_attention.cu).

The custom backend has two kernels behind it. The default one runs both of attention's
matrix multiplies — `Q @ K^T` and `P @ V` — on the GPU's tensor cores through
`nvcuda::wmma`, with the softmax fused between them and the `[B, H, S, S]` score matrix
never leaving shared memory. A scalar fallback covers the shapes tensor cores cannot take
and stays selectable for comparison.

---

## Contents

- [Results](#results)
- [Setup](#setup)
- [Running](#running)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Notes and known limits](#notes-and-known-limits)
- [REPORT.md](REPORT.md) — what was built and what broke: a plain-language account of every problem hit and what fixed it, then the step-by-step detail of moving the kernel onto tensor cores

---

## Results

Measured on a power-capped RTX 4050 Laptop (see [Environment](#environment)).

### The attention op

This is what the kernel work moves, so it is measured directly rather than inferred from
the end-to-end number. fp32, timings interleaved and minimum-of-N so all three candidates
see the same clock state:

| Shape | wmma vs. scalar | wmma vs. SDPA |
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

Every shape improves, and the tensor-core kernel is faster than SDPA on all of them.
`scripts/bench_attention.py` reproduces this table.

### The whole transformer

Harness speedup against `BaselineTransformer`, one column per backend. Attention is only
part of a Transformer layer, so these numbers are diluted by everything else in it — at
`seq_len=128` the FFN alone does about 8x the work of the attention core, which is why the
first four rows barely move whatever the kernel does. The rows that separate the backends
are the long ones, where the score matrix grows as `S²` while the FFN grows as `S`:

| Configuration | SDPA | custom scalar | custom wmma |
| --- | --- | --- | --- |
| default (B8 S128 D512 H8 L6) | 0.999x | 0.900x | **1.007x** |
| causal | **1.055x** | 0.998x | FAIL |
| padded (30%) | 0.992x | 0.846x | **1.001x** |
| causal + padded | **1.177x** | 1.089x | FAIL |
| seq_len 512 | 1.365x | 1.268x | **1.398x** |
| seq_len 2048 | 1.816x | 1.678x | **2.270x** |
| seq_len 2048, causal | 3.124x | 2.915x | **3.785x** |
| small (B1 S32) | **1.529x** | 1.268x | 1.497x |
| wide (d_model 1024) | **1.033x** | 0.961x | 0.988x |
| deep (12 layers) | FAIL | **PASS** | FAIL |

Read this table with its error bars in mind. The GPU is thermally and power capped, and a
single harness run gives one median with no interleaving against the other backends — the
same configuration was measured at 1.401x and 0.958x on two runs of `deep 12L`. Differences
under about 10% here are not evidence of anything; `seq_len 2048` and `seq_len 2048 causal`
are outside that band and agree with the attention-op table above.

**Where the tensor cores show up.** At `seq_len=2048` the custom backend goes from 1.678x
with the scalar kernel to 2.270x with the tensor-core one, overtaking SDPA's 1.816x; with
causal masking, 2.915x to 3.785x against SDPA's 3.124x. Those are the configurations where
attention is most of the runtime, and they are the ones the kernel was written for.

**The PASS/FAIL columns are not a precision ranking.** Three configurations sit on the
accuracy gate on this machine — `causal`, `causal + padded` and `deep 12L` — and which
backend passes is close to a coin flip. On `deep 12L` over 8 trials, SDPA failed on 1
element of 4.19M, wmma on 2, and the scalar kernel passed *with the largest `max_abs` of the
three* (1.187e-3, against wmma's 1.140e-3). The gate is `abs <= atol OR rel <= rtol`, so
which elements fail depends on where the reference happens to be near zero, not on which
kernel rounds more. See [Notes and known limits](#notes-and-known-limits).

---

## Setup

### Requirements

| | |
| --- | --- |
| GPU | NVIDIA. Compute capability 8.0+ for the tensor-core kernel; anything CUDA-capable runs the scalar one |
| CUDA Toolkit | 13.0+. 13.3+ additionally enables the cuTile kernel (`--attn-impl tile`), which needs `<cuda_tile.h>`; it is picked up automatically when installed alongside an older toolkit |
| PyTorch | 2.12.0+cu132 |
| Python | 3.10+ |
| Compiler | MSVC — Visual Studio with the "Desktop development with C++" workload |
| Build tool | `ninja` |

The build targets whatever card is present: [`kernel_ext.py`](kernel_ext.py) reads the
compute capability from `torch.cuda.get_device_capability()` and passes it to `nvcc`, with
PTX for the same virtual arch alongside so the cached build still loads on a different card
of the same family. It also finds MSVC itself (via `vswhere` + `vcvarsall.bat`) and, when a
CUDA 13.3+ toolkit is installed, builds the tile kernel against it — several toolkits can
sit side by side and the tile-capable one is found by looking for the header rather than by
trusting `PATH`. Set `CUDA_TILE_HOME` to override that search.

Only the CUDA-kernel backend needs MSVC and `nvcc`. **The SDPA backend needs neither** — if
you just want to run the benchmark, skip to [Running](#running) and everything works with
PyTorch alone.

### 1. Install Python dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

```bash
pip install ninja
```

Match the wheel's CUDA build to your toolkit. This project was developed against
`torch 2.12.0+cu132` with CUDA Toolkit 13.0; a mismatch between the two is the usual cause
of extension build failures.

### 2. Build the CUDA extension (optional)

The extension is compiled by `torch.utils.cpp_extension` and cached in `build/`.
Compiling needs `cl.exe` on `PATH`, which a plain shell does not have — MSVC is not on the
global `PATH` even when Visual Studio is fully installed, only inside a developer command
prompt. So the build goes through `scripts/devenv.bat`, which sets that up:

```bash
cmd.exe /c scripts\build_ext.bat
```

Expected output:

```
[build_ext] OK -> ...\build\transformer_kernels.pyd
```

Rebuilds after the first are near-instant (`ninja: no work to do` when nothing changed),
and editing anything in `csrc/` makes the next run recompile automatically — there is no
separate build step to remember. A first build takes roughly 70 s, nearly all of it the
tensor-core kernel's template instantiations.

`scripts/devenv.bat` remains for cases that need `cl.exe` on `PATH` before Python starts.
It locates Visual Studio with `vswhere`, so there is no path to edit if yours is installed
somewhere unexpected. See [Toolset selection](#toolset-selection) if the build fails on the
compiler version.

If the tile kernel fails to build, the loader retries without it rather than losing the
other two; `kernel_ext.tile_enabled()` reports whether it made it in.


### 3. Verify

```bash
cmd.exe /c scripts\devenv.bat python scripts\verify_kernel.py
```

This checks the kernel against the baseline's exact arithmetic across 12 shapes and prints
per-shape timings. It should end with `all cases match the reference`.

### Troubleshooting

**`custom CUDA kernel unavailable, using SDPA instead`** — not an error, and not a claim
that MSVC is missing. It means `cl.exe` was not on `PATH`, so the extension could not be
built and attention fell back to SDPA; results are still correct and the benchmark still
reports a real speedup.

`cl.exe` is not on `PATH` in a normal shell even with Visual Studio fully installed — only
inside a developer command prompt. Prefix the command with `devenv.bat`:

```bash
cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
```

Every run needs the prefix, because `torch.utils.cpp_extension.load()` probes for the host
compiler before it will even check whether a rebuild is needed. To make a missing kernel a
hard error rather than a silent fallback, use `--attn-backend custom`.

To use SDPA deliberately and silence the message, set `ATTENTION_BACKEND = "sdpa"` at the
top of the file. To make a missing kernel a hard error instead of a fallback, set it to
`"custom"`.

**`'vswhere.exe' is not recognized`** — harmless, printed by `vcvarsall.bat` itself. The
build succeeds regardless.

**Extension builds but the harness is slower than expected** — confirm which backend
actually ran. With `"auto"`, a silent fallback means you may be timing SDPA rather than the
kernel; `"custom"` raises instead of falling back.

### Toolset selection

`nvcc` accepts only a window of MSVC versions, and Visual Studio defaults to the newest
toolset it has. With CUDA 13.0 and a 14.5x toolset that shows up either as
`unsupported Microsoft Visual Studio version` or — worse, because it looks like a compiler
bug rather than a configuration problem — as `cudafe++ died with status 0xC0000005`.

[`scripts/devenv.bat`](scripts/devenv.bat) therefore locates Visual Studio with `vswhere`
(rather than hard-coding a path) and pins the toolset with `-vcvars_ver`. The default is
`14.44`; override it if your CUDA version wants a different one:

```bash
set VCVARS_VER=14.43
cmd.exe /c scriptsuild_ext.bat
```

If the pinned toolset is not installed, the script lists the ones that are.

---

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

To use the CUDA kernel, run through `devenv.bat` so the build can find `cl.exe`:

```bash
cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
```

### Choosing the attention backend

Edit `ATTENTION_BACKEND` near the top of
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py):

```python
ATTENTION_BACKEND = "auto"
```

| Value | Behavior |
| --- | --- |
| `auto` | Use the CUDA kernel if it loads, otherwise fall back to SDPA with a one-time notice. |
| `sdpa` | Always use `F.scaled_dot_product_attention`. No build required. |
| `custom` | Require the CUDA kernel; raise if it is unavailable, so a broken build fails loudly instead of quietly benchmarking the fallback and looking slow. |

For a single run without editing the file, `--attn-backend` overrides it:

```bash
python torch_transformer_benchmark.py --attn-backend sdpa
```

### Choosing the kernel inside the custom backend

`ATTENTION_IMPL` (or `--attn-impl` for one run) picks which of the custom kernels handles
attention:

| Value | Behavior |
| --- | --- |
| `auto` | Tensor-core kernel where it applies, scalar kernel otherwise. |
| `scalar` | Force the scalar kernel. No tensor cores, no TF32 rounding. |
| `wmma` | Force the tensor-core kernel; raises on shapes it does not cover, so a silent fallback cannot be mistaken for a slow kernel. |
| `tile` | Force the cuTile kernel — the same math written against the CUDA tile programming model instead of per-thread. float32, `head_dim` in {8,16,32,64}. Raises rather than falling back. |
| `tile-tf32` | The same cuTile kernel with its two GEMMs narrowed to TF32, which is what puts them on the tensor cores. Same arithmetic `wmma` uses for fp32 inputs and the same ~1e-3 accuracy, so it clears the harness gate wherever `wmma` does. The tensor-core tile mode to reach for first. |
| `tile-bf16` | As above but narrowed to bfloat16 — 8 mantissa bits. Marginally faster than `tile-tf32` on some shapes and ~4 orders of magnitude less accurate; fails the harness gate on most configs. |

The tensor-core kernel covers `head_dim` 8, 16, 32, 64 and 128 in float32, float16 and
bfloat16, on compute capability 8.0 and up — every head_dim the harness can produce, since
`d_model` is divisible by `num_heads`. Nothing falls through to ATen any more.

Neither tile mode is ever chosen by `auto`: they are a separate programming model whose
performance you should opt into deliberately. It needs a build that found CUDA 13.3+ (see
[Building the CUDA extension](#2-build-the-cuda-extension-optional)); without one,
`--attn-impl tile` raises instead of silently running something else. On an RTX 3070 plain
`tile` is the most accurate kernel here (fp32 throughout, ~1e-6 against an exact reference)
and the slowest; `tile-tf32` trades that for the tensor cores at wmma's ~1e-3 — see
[Notes and known limits](#notes-and-known-limits).

### Helper scripts

| Script | Purpose |
| --- | --- |
| `scripts/verify_kernel.py` | Kernel vs. reference vs. SDPA across 12 shapes. Fails fast and names the shape that broke. |
| `scripts/bench_attention.py` | Times the attention op alone — scalar vs. tensor-core vs. SDPA — with accuracy alongside, so a speed win bought with precision is visible. |
| `scripts/compare_backends.py` | Runs the full harness once per backend and prints the comparison table above. Set `COMPARE_FULL=1` for the harness's own accuracy-trial count instead of the trimmed one. |

All three want `devenv.bat` in front of them:

```bash
cmd.exe /c scripts\devenv.bat python scripts\compare_backends.py
```

---

## How it works

### What must be preserved

The harness compares against `BaselineTransformer` element by element:

```
abs(user - ref) <= atol   OR   abs(user - ref) <= rtol * abs(ref)
```

with `atol=0.001`, `rtol=0.01` by default. Every element must satisfy one of the two. That
makes several baseline details load-bearing rather than incidental:

- **Pre-norm residuals** — `x = x + Attn(norm1(x))`, then `x = x + FFN(norm2(x))`. The
  residual adds the *un-normalized* `x`.
- **Exact GELU** (`approximate="none"`). The tanh approximation alone drifts far enough to
  fail `atol`.
- **Softmax accumulated in fp32**, then cast back.
- **Bias on all four attention projections and both FFN layers.**
- **Padding masks invalid *key* positions**, and the final output is zero-filled at padded
  rows after `final_norm`.

### Optimizations applied

**Fused attention.** Both backends collapse `matmul → mask → softmax → matmul` into one
operation and never write the `[B, H, S, S]` score matrix to global memory. The custom
kernel does this FlashAttention-style: K/V are streamed through shared memory a tile at a
time, with a running max and running sum so the softmax is computed incrementally.

**Both matmuls on tensor cores.** The default custom kernel runs `Q @ K^T` and `P @ V`
through `nvcuda::wmma` fragments rather than scalar FMA. A block owns 64 query rows split
across 4 warps, one 16-row stripe each — 16 being the `M` of a wmma tile — and walks the
keys in tiles of 32. Per tile a warp computes its `16×32` score block, softmaxes it, and
multiplies straight into its `16×head_dim` output block. fp32 inputs use TF32 fragments with
fp32 accumulate, which is the same arithmetic cuBLAS gives the baseline when
`torch.backends.cuda.matmul.allow_tf32` is on (the harness default); half and bfloat16 use
native `16×16×16` fragments.

Three details are what make it a win rather than a wash — each was worth measuring on its
own, and the first two were regressions until fixed:

- **Padded shared-memory leading dimensions.** A fragment load walks a *column* of a tile,
  so a row stride of 16, 32 or 64 floats puts every row of the fragment in one shared-memory
  bank and serialises the load. Padding each tile's leading dimension by the smallest amount
  wmma permits (4 floats, or 8 halves) rotates successive rows off each other.

- **Q and O held in registers.** Q is read into fragments once per block and never re-read.
  O accumulates in accumulator fragments for the whole key loop instead of being written
  back to shared memory each tile. That second one needs the per-row softmax rescale to be
  applied to fragment *elements*, and the element-to-row mapping is architecture-defined —
  CUDA deliberately does not document it. Rather than hard-code a layout, the kernel probes
  it: it stores one fragment whose elements are tagged with `(lane, slot)`, reads back where
  each tag landed, and inverts the mapping. One 16×16 tile per warp, once per block, exact
  by construction on any device the kernel compiles for.

- **One softmax lane per query row, not per key column.** The obvious mapping — lane ==
  key column — needs a 5-step butterfly per row, 16 rows deep, and that cost does not
  shrink with `head_dim`, so at `head_dim=16` it swamped both GEMMs. Giving each lane a
  whole row segment leaves one shuffle: between the two lanes that share a row.

**Causal tile skipping.** Under causal masking, no query in a block looks past that block's
last query row, so whole key tiles beyond it are never loaded or computed — not computed
and discarded.

**A scalar kernel as the fallback.** One thread per query row, plain fp32 FMA, no tensor
cores and so no TF32 rounding at all. It covers what wmma cannot (`head_dim` 8, pre-Ampere
cards) and stays selectable through `--attn-impl scalar` so the tensor-core win can be
measured rather than assumed.

**Cached causal mask.** The mask depends only on `seq_len`, which is fixed for a model
instance, so it is built once and reused instead of being rebuilt on all 6 layers of every
forward pass.

**Hoisted mask-triviality check.** The default `--padding-ratio 0` still passes an all-ones
mask. Detecting that once per forward pass (rather than once per layer) lets attention take
the faster no-mask path, and removes 5 redundant GPU→CPU syncs from the hot path.

**Custom modules that keep baseline parameter names.** `MyLinear`, `MyLayerNorm`,
`MySelfAttention`, and `MyTransformerBlock` reuse the baseline's attribute names and
parameter shapes, so `state_dict` keys line up and strict weight loading works untouched —
full freedom over `forward`, no custom weight-mapping code.

### Why the score matrix is the bottleneck

Attention computes one score per (query, key) pair — `S × S` numbers, not `S`. Double the
sequence length and that table quadruples. The baseline writes it to memory and reads it
back for each of masking, softmax, and the `×V` matmul. Those round trips, not the
arithmetic, are what cost the time: at `B=8, H=8, S=2048` in fp32 the score tensor alone is
about 1 GB *per layer*.

---

## Repository layout

```
torch_transformer_benchmark.py   harness + baseline + the optimized implementation
csrc/fused_attention.cu          custom fused attention CUDA kernels (tensor-core + scalar)
csrc/tile_attention.cu           the same attention on the CUDA tile programming model
csrc/tile_attention.h            plain-pointer boundary between the two translation units
kernel_ext.py                    JIT loader; returns None instead of raising if unavailable
scripts/devenv.bat               runs a command with MSVC on PATH
scripts/build_ext.bat            one-shot build + load check
scripts/verify_kernel.py         correctness harness for the kernel
scripts/bench_attention.py       attention-op-only benchmark
scripts/compare_backends.py      full-harness comparison across backends
```

`build/` is generated and git-ignored.

---

## Notes and known limits

**The accuracy budget is tighter than it looks.** The baseline's *own* TF32 rounding sits
9.8e-4 (non-causal) to 1.2e-3 (causal) away from an exact fp32 result — at or above
`atol=0.001`. Since the harness compares against the baseline's rounded output rather than
ground truth, being *more* mathematically correct does not help. Only closeness to the
baseline's specific rounding does. This is why restructuring GEMM order (for example fusing
Q/K/V into one matmul) can fail the gate while being no less correct: it was measured to
push `max_abs` from 9.9e-4 to 1.12e-3.

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
  earlier version of this file claimed CUDA 13.3 does not define `__nv_tf32`; it does, in
  `include/cuda_tf32.h`. Confirm which units a mode got with
  `cuobjdump -sass build/tile_attention.cuda.o | grep HMMA`: fp32 kernels contain none.
- The TMA hardware the tile model is designed around is Blackwell-only; on Ampere the
  loads fall back to software-managed async copies.

Block shape matters far more here than in the hand-written kernels, and not smoothly: the
kernel keeps Q, O, K, V and the score tile live at once, and past a footprint threshold the
compiler spills and the cost jumps an order of magnitude. At `head_dim=64`, `BLOCK_N=16`
runs at 1.5 ms where `BLOCK_N=32` runs at 10.9 ms. The per-`head_dim` shapes in `BlockCfg`
were measured on SM 8.6 and are worth re-measuring on another architecture.

**The tensor-core kernel now covers every head_dim, and both former gaps were block-shape
problems rather than design ones.** `head_dim=8` is narrower than the 16-wide wmma
fragment, so GEMM2's N dimension could not be filled; the kernel widens it to 16 with zeros
in shared memory, which costs no extra global traffic because GEMM1 contracts over head_dim
(zeros add nothing) and the padded output columns are simply not stored. `head_dim=128` at
the default `64×32` block wanted 75.8 KB of shared memory, over the 48 KB that keeps two
blocks resident per SM — but a `32×16` block brings it to 35.9 KB, so `WmmaShape` picks the
block per head_dim instead. Both changes bought speed, not just coverage: at `head_dim=128`
wmma runs 0.041 ms where the old scalar fallback took 0.142 ms.

The scalar kernel still exists for pre-Ampere cards and for A/B measurement; ATen is no
longer reached for any head_dim the harness produces.

**Tensor cores did not cost precision the way the old note here predicted.** The worry was
that TF32 fragments would push the kernel away from the baseline. On the attention op the
opposite holds: because the harness runs the baseline with TF32 on, the tensor-core kernel
lands about 2x *closer* to it than the scalar kernel does (6.4e-5 vs 2.1e-4 at `seq2048`,
3.7e-4 vs 7.1e-4 at the default shape). End to end that does not translate — over 12
accuracy trials the scalar kernel peaked at `max_abs` 7.7e-4 and the tensor-core kernel at
8.7e-4 — but it does not reverse either.

**Three configurations sit on the accuracy gate, and which backend passes is close to
chance.** `causal`, `causal + padded` and `deep 12L` all fail intermittently, by one or two
elements out of hundreds of thousands to millions, at `max_abs` ≈ 1.0–1.2e-3 against
`atol=1e-3`. This is not specific to the custom kernels: plain `--attn-backend sdpa` fails
`causal` and `deep 12L` too.

The clearest evidence that it is chance rather than a precision ranking comes from
`deep 12L` over 8 trials per backend:

| | verdict | `max_abs` | failed elements |
| --- | --- | --- | --- |
| SDPA | FAIL | 1.101e-3 | 1 / 4,194,304 |
| custom scalar | PASS | 1.187e-3 | 0 |
| custom wmma | FAIL | 1.140e-3 | 2 / 4,194,304 |

The backend that passed has the *largest* `max_abs` of the three. The criterion is
`abs <= atol OR abs <= rtol * abs(ref)`, so a large absolute error is forgiven wherever the
reference is large, and what decides the verdict is whether the few elements with a
near-zero reference happen to land inside `atol`. Trials routinely report `max_rel` in the
hundreds, which is the signature of exactly that. Halving the systematic error, as the
tensor-core kernel does, barely moves it.

The baseline's own causal attention is that far from any reordered implementation of the
same math, so the outcome depends on the GPU and on cuBLAS/SDPA kernel selection — the
Results table was measured on the RTX 4050 in [Environment](#environment), and the same
configurations were reported as passing on an RTX 3070. Running with
`--no-allow-tf32 --matmul-precision highest` removes the TF32 rounding from both sides and
the margin returns.

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

The GPU is a power- and thermally-capped laptop part: `nvidia-smi` reports
`SW Power Cap` and `SW Thermal Slowdown` active under load, and absolute latencies drift by
2–3x across a long session. Every ratio quoted here therefore comes from timings taken
*interleaved* — candidates measured round-robin, minimum of N — so the clock state is
shared between them. Absolute milliseconds from different runs are not comparable; ratios
within one run are.

`torch.compile` and Triton are unavailable in this environment (Triton has no working
Windows build here), which is why the custom-kernel path is C++/CUDA via
`torch.utils.cpp_extension` rather than Triton.
