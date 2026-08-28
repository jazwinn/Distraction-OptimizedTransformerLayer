# Optimizing a Transformer Layer

An optimized Transformer layer for **TikTok TechJam 2026, Problem Statement 3** — accelerate a
standard Transformer layer with custom GPU kernels while staying inside a strict
per-element accuracy budget.

The work lives in the [`optimized/`](optimized/) package, which
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) mixes into
`UserOptimizedTransformer` — the harness file itself stays close to the one that was issued,
so the diff against it is readable. `optimized/model.py` holds the forward pass,
`optimized/layers.py` its submodules, `optimized/kernels.py` the dispatch into CUDA,
`optimized/graphs.py` the CUDA-graph machinery, and `optimized/config.py` every runtime knob.

Attention runs through one of two interchangeable backends: PyTorch's
`scaled_dot_product_attention`, or a hand-written fused CUDA kernel in
[`csrc/fused_attention.cu`](csrc/fused_attention.cu).

The custom backend has two kernels behind it. The default one runs both of attention's
matrix multiplies — `Q @ K^T` and `P @ V` — on the GPU's tensor cores through
`nvcuda::wmma`, with the softmax fused between them and the `[B, H, S, S]` score matrix
never leaving shared memory. A scalar fallback covers the shapes tensor cores cannot take
and stays selectable for comparison. A third kernel in
[`csrc/tile_attention.cu`](csrc/tile_attention.cu) expresses the same maths against the CUDA
tile programming model, as a portability-versus-speed comparison rather than as the default.

Around the kernels, the forward pass fuses its elementwise work into the surrounding
operations and — on shapes where launch overhead is what dominates — is captured into a CUDA
graph and replayed, which is bit-identical to running it eagerly.

---

## Contents

- [Setup](#setup) — install, build, verify, and tune the graph gate for your machine
- [How it works](#how-it-works) — what the accuracy gate forces, and every optimization applied
- [Repository layout](#repository-layout)
- **[REPORT.md](REPORT.md)** — results, how to run the harness and the helper scripts, accuracy
  analysis, known limits, and the engineering record of what broke and what fixed it

Numbers do not live in this file. Every measurement is in [REPORT.md](REPORT.md), with the
machine it was taken on and the methodology beside it.

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
you just want to run the benchmark, `python torch_transformer_benchmark.py` works with
PyTorch alone.

### 1. Install Python dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

```bash
pip install ninja
```

Match the wheel's CUDA build to your toolkit. This project was developed against
`torch 2.12.0+cu132`; a mismatch between the two is the usual cause of extension build
failures.

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

This checks every kernel against the baseline's exact arithmetic across 12 shapes and prints
per-shape timings. It should end with `every kernel matches the reference on every case`.

Two more correctness harnesses are worth running after a build, both described in
[REPORT.md](REPORT.md):

```bash
cmd.exe /c scripts\devenv.bat python scripts\verify_split_kv.py
cmd.exe /c scripts\devenv.bat python scripts\verify_graph.py
```

### 4. Find the CUDA-graph gate value for your machine

CUDA graph capture is **on by default**, and how widely it applies is decided by one
constant in [`optimized/config.py`](optimized/config.py):

```python
_GRAPH_MAX_ACTIVATION = 1 << 19    # 524288
```

`auto` captures a shape when `batch * seq_len * d_model` is at or under that value. The
number is where graph replay stopped beating eager execution **on the machine this was
developed on**, and it does not transfer: the crossover is really the point where the GPU
stops starving between kernel launches, which depends on the card's throughput relative to
how fast the host can issue work. A faster GPU starves at larger shapes and wants a *larger*
value here.

To measure it on your own machine:

```bash
cmd.exe /c scripts\devenv.bat python scripts\ab_graph.py --recommend
```

The script sweeps activation volume in powers of two, times eager against replay
interleaved in one process, and prints the value to set:

```
activation              shapes      worst       best    control
----------------------------------------------------------------
     16384        b1s32d512 +1     4.279x     5.520x     1.001x
     65536       b1s128d512 +1     2.484x     3.260x     1.010x
    262144       b4s128d512 +1     1.044x     1.356x     1.009x
    524288       b8s128d512 +1     1.033x     1.035x     1.009x
   1048576       b4s512d512 +1     1.009x     1.018x     1.008x
   2097152       b8s512d512 +1     1.001x     1.004x     1.001x

noise floor from the control rows: +/-1.5%
a gain counts as real above 1.015x

  set _GRAPH_MAX_ACTIVATION = 524288    # 1 << 19
```

Three things make that output trustworthy rather than just a number:

- The **`control` column is eager timed against eager** — identical code on both sides, so its
  true value is exactly `1.000x` and whatever it actually reads is your machine's noise. A
  gain only counts if it clears that.
- It **refuses to answer** when the control rows are wider than the effects being measured,
  and tells you to close whatever is loading the machine, instead of returning a threshold
  derived from noise. A game running in the background was enough to trigger this during
  development.
- It samples **two shapes at each activation volume and keeps the worse one**, so it errs
  toward capturing less.

**This step is optional.** Getting the constant wrong is cheap in both directions — too low
leaves some latency unclaimed, too high pins some GPU memory for no gain — and neither can
produce a wrong answer, because replay executes the identical kernels in the identical order.
If you skip it, the harness still prints which path each run took and why:

```
[info] CUDA graph captured: shape=(8, 128, 512) float32 mask=off ... replay matches eager exactly
[info] CUDA graph declined for shape (4, 512, 512): batch*seq*d_model is 1048576, over the
       524288 above which replay measured no gain on this hardware.
```

`--cuda-graph {off,auto,always}` overrides the whole mechanism for one run.

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

To use SDPA deliberately and silence the message, set `ATTENTION_BACKEND = "sdpa"` in
[`optimized/config.py`](optimized/config.py).

**`'vswhere.exe' is not recognized`** — harmless, printed by `vcvarsall.bat` itself. The
build succeeds regardless.

**Extension builds but the harness is slower than expected** — confirm which backend
actually ran. With `"auto"`, a silent fallback means you may be timing SDPA rather than the
kernel; `"custom"` raises instead of falling back.

**A run prints a passing verdict and then exits non-zero** — this was a cuTile teardown
fault (`0xC0000005` *after* `main()` returned) and it is fixed: importing `kernel_ext`
before `torch` preloads the display driver's GPU compiler, which is what the fault was in.
If you see it again, check that whatever you ran imports `kernel_ext` above `import torch`
— `kernel_ext.tile_compiler_status()` says whether the preload landed in time. The
diagnosis is in `preload_tile_compiler()`'s docstring and in [REPORT.md](REPORT.md).

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
cmd.exe /c scripts\build_ext.bat
```

If the pinned toolset is not installed, the script lists the ones that are.

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

How tight that budget actually is, and why being *more* accurate than the baseline does not
help, is in [REPORT.md](REPORT.md).

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
cores and so no TF32 rounding at all. It covers what wmma cannot (pre-Ampere cards) and
stays selectable through `--attn-impl scalar` so the tensor-core win can be measured rather
than assumed.

Past head_dim 64 it is *half* a row per thread. A thread keeps q and the output accumulator
for its row in registers — that is what makes the key loop a register FMA rather than a
reload — and at head_dim 128 those two arrays want 256 registers against a hardware ceiling
of 255. So two adjacent lanes share the row: 64 dims of each array apiece, one
`__shfl_xor_sync` per key to add their partial dot products together, and the softmax
bookkeeping repeated identically on both halves rather than communicated. Nothing else
crosses between them, the two threads read disjoint halves of the key row so shared-memory
traffic is unchanged, and the doubled block lifts occupancy from two warps per SM to twelve.
Both partners share a row index, so they take every branch together — the `i < S` guard, the
causal break, the mask skip — which is what makes a two-lane shuffle mask safe.

**Fused residual add + LayerNorm.** Every LayerNorm in the model consumes the output of a
residual add, so the two are one kernel: the sum is held on chip rather than written to
global memory and read straight back. The kernel returns both results, because the caller
needs the un-normalised sum for its own skip connection. The only norm that cannot fuse is
the very first, which has no add before it.

**One GEMM for Q, K and V.** Three separate `[B*S, d] × [d, d]` projections leave the GPU
with too few output tiles to fill it, so cuBLAS splits the contraction and launches a second
kernel to add the partial sums back together. One `[d, 3d]` GEMM fills the card in a single
pass instead. The weights are concatenated lazily and cached, so the fused copy is built
once rather than per forward pass.

**Cached causal mask.** The mask depends only on `seq_len`, which is fixed for a model
instance, so it is built once and reused instead of being rebuilt on every layer of every
forward pass.

**Hoisted mask-triviality check.** The default `--padding-ratio 0` still passes an all-ones
mask. Detecting that once per forward pass (rather than once per layer) lets attention take
the faster no-mask path, and removes the redundant GPU→CPU syncs from the hot path. The
answer is memoized on a weak reference to the mask tensor, so the steady state has no sync
at all — which is also what makes graph capture possible, since a device-to-host read inside
a capture region is illegal.

**CUDA graph capture and replay.** Kernel launches are asynchronous, so the CPU runs ahead
queueing the next kernel while the GPU works on the current one. When the average kernel
outlasts the time it takes to issue one, launch cost is invisible; when it does not, the GPU
starves between launches. On the launch-bound shapes it does: the forward pass issues ~79
kernels, and at small batch and sequence length most of the wall clock is the GPU waiting to
be fed.

Capture records that launch sequence once and replays it as a single submission. The
arithmetic is untouched — the same kernels, in the same order, on the same addresses — so
replay is **bit-identical** to eager execution, which the capture routine verifies before
installing a graph rather than assuming. `_forward_eager` is the captured region, and
`forward` keeps the one device-to-host sync on the outside of it. Static input buffers are
allocated as normal (non-inference) tensors so they can be refilled from any mode, real
inputs are copied in per call, and a graph is cached per
`(shape, dtype, device, mask mode, backend, impl)`.

**Custom modules that keep baseline parameter names.** `MyLinear`, `MyLayerNorm`,
`MySelfAttention` and `MyTransformerBlock` in [`optimized/layers.py`](optimized/layers.py)
reuse the baseline's attribute names and parameter shapes, so `state_dict` keys line up and
strict weight loading works untouched — full freedom over `forward`, no custom
weight-mapping code. `UserOptimizedTransformer` inherits from both `OptimizedTransformer` and
`BaselineTransformer`, which is what keeps the two `isinstance`-compatible for the harness.

### Why the score matrix is the bottleneck

Attention computes one score per (query, key) pair — `S × S` numbers, not `S`. Double the
sequence length and that table quadruples. The baseline writes it to memory and reads it
back for each of masking, softmax, and the `×V` matmul. Those round trips, not the
arithmetic, are what cost the time: at `B=8, H=8, S=2048` in fp32 the score tensor alone is
about 1 GB *per layer*.

That is also why the end-to-end speedup is much smaller than the attention-op speedup at
short sequences, and much larger at long ones. [REPORT.md](REPORT.md) works through where
the time actually goes.

---

## Repository layout

```
torch_transformer_benchmark.py            the harness as issued, plus three hooks into optimized/
kernel_ext.py                             JIT loader; returns None instead of raising if unavailable
requirements.txt                          pinned Python dependencies

optimized/__init__.py                     package exports
optimized/config.py                       the runtime knobs: backend, kernel, CUDA graphs
optimized/cli.py                          --attn-backend / --attn-impl / --cuda-graph
optimized/model.py                        OptimizedTransformer: the whole forward pass
optimized/layers.py                       its submodules, named to match the baseline's
optimized/kernels.py                      dispatch into csrc/, with an SDPA fallback
optimized/graphs.py                       CUDA graph capture, replay and teardown
optimized/util.py                         small shared helpers

csrc/fused_attention.cu                   fused attention CUDA kernels (tensor-core + scalar),
                                          fused add+LayerNorm, and the extension bindings
csrc/tile_attention.cu                    the same attention on the CUDA tile programming model
csrc/tile_attention.h                     plain-pointer boundary between the two translation units
csrc/TUNING.md                            the measurements behind every block shape and threshold

scripts/devenv.bat                        runs a command with MSVC on PATH
scripts/build_ext.bat                     one-shot build + load check

scripts/verify_kernel.py                  correctness: every kernel, 12 shapes
scripts/verify_split_kv.py                correctness: the tile kernel's split-KV path
scripts/verify_graph.py                   correctness: graph replay is bit-identical to eager

scripts/bench_attention.py                timings: the attention op alone
scripts/compare_backends.py               timings: the full harness, once per backend
scripts/ab_split_kv.py                    timings: split-KV against single-pass, interleaved
scripts/ab_layout.py                      timings: strided q/k/v against contiguous, interleaved
scripts/ab_graph.py                       timings: replay against eager, interleaved;
                                          --recommend tunes _GRAPH_MAX_ACTIVATION
scripts/tune_block_shapes.py             block-shape sweep for every attention backend
scripts/sass_mix.py                       SASS instruction mix and occupancy, head_dim 64

REPORT.md                                 results, running instructions, and the engineering record
```

`build/` is generated and git-ignored.
