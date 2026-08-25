# Optimizing a Transformer Layer

An optimized Transformer layer for **TikTok TechJam 2026, Problem Statement 3** — accelerate a
standard Transformer layer with custom GPU kernels while staying inside a strict
per-element accuracy budget.

The work is in `UserOptimizedTransformer` and the modules it builds on, all inside
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py). Attention runs through
one of two interchangeable backends: PyTorch's `scaled_dot_product_attention`, or a
hand-written fused CUDA kernel in [`csrc/fused_attention.cu`](csrc/fused_attention.cu).

---

## Contents

- [Results](#results)
- [Setup](#setup)
- [Running](#running)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Notes and known limits](#notes-and-known-limits)

---

## Results

Measured on an RTX 3070 (see [Environment](#environment)). Speedup is the harness's own
median-latency ratio against the baseline; every configuration passes the accuracy gate
unless marked otherwise.

| Configuration | SDPA backend | Custom CUDA kernel |
| --- | --- | --- |
| default (B8 S128 D512 H8 L6) | **1.110x** | 1.029x |
| causal | **1.262x** | 1.183x |
| padded (30%) | **1.090x** | 1.034x |
| causal + padded | **1.178x** | 1.126x |
| seq_len 512 | **1.379x** | 1.258x |
| seq_len 2048 | **1.720x** | 1.667x |
| seq_len 2048, causal | **2.889x** | 2.472x |
| small (B1 S32) | **1.417x** | 1.277x |
| wide (d_model 1024) | **1.095x** | 1.027x |
| deep (12 layers) | FAIL | **1.028x** |

Two things worth reading off this table:

**Speedup scales with sequence length.** The baseline materializes a `[B, H, S, S]` score
matrix and round-trips it through global memory several times. That cost grows as `S²`,
so eliminating it matters most exactly where it hurts most — 2.89x at `seq_len=2048` with
causal masking, versus 1.11x at the default `seq_len=128`.

**The custom kernel is the only backend that passes `deep 12L`.** It accumulates in exact
fp32, while SDPA rounds through TF32 tensor cores. Over 12 layers that rounding drifts far
enough to fail a single element out of 2.6M. The same choice that buys the accuracy costs
5–15% of the speed everywhere else — the kernel gives up tensor cores to get it.

---

## Setup

### Requirements

| | |
| --- | --- |
| GPU | NVIDIA, compute capability 8.6 (see note below for other cards) |
| CUDA Toolkit | 13.0 (`nvcc` on `PATH`) |
| PyTorch | 2.12.0+cu132 |
| Python | 3.10 |
| Compiler | MSVC — Visual Studio 2022 with the C++ desktop workload |
| Build tool | `ninja` |

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

The extension is JIT-compiled by `torch.utils.cpp_extension` on first use and cached in
`build/`. It needs `cl.exe` on `PATH`, which a plain shell does not have — so every build
goes through `scripts/devenv.bat`, which calls `vcvarsall.bat` first.

```bash
cmd.exe /c scripts\build_ext.bat
```

Expected output:

```
[build_ext] OK -> ...\build\transformer_kernels.pyd
```

Rebuilds after the first are near-instant (`ninja: no work to do` when nothing changed).

If `vcvarsall.bat` is somewhere else on your machine, edit the path at the top of
[`scripts/devenv.bat`](scripts/devenv.bat).

### 3. Verify

```bash
cmd.exe /c scripts\devenv.bat python scripts\verify_kernel.py
```

This checks the kernel against the baseline's exact arithmetic across 12 shapes and prints
per-shape timings. It should end with `all cases match the reference`.

### Targeting a different GPU

The build hard-codes SM 8.6. For another card, change the `-gencode` flag in
[`kernel_ext.py`](kernel_ext.py):

```python
"-gencode=arch=compute_86,code=sm_86",   # RTX 3070
```

The kernel itself is architecture-agnostic — it uses no SM-specific intrinsics.

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

Set `TTB_ATTN_BACKEND`:

| Value | Behavior |
| --- | --- |
| `auto` *(default)* | Use the CUDA kernel if it loads, otherwise fall back to SDPA with a one-time warning. |
| `sdpa` | Always use `F.scaled_dot_product_attention`. No build required. |
| `custom` | Require the CUDA kernel; raise if it is unavailable, so a broken build fails loudly instead of quietly benchmarking the fallback and looking slow. |

```bash
set TTB_ATTN_BACKEND=sdpa
```

### Helper scripts

| Script | Purpose |
| --- | --- |
| `scripts/verify_kernel.py` | Kernel vs. reference vs. SDPA across 12 shapes. Fails fast and names the shape that broke. |
| `scripts/bench_attention.py` | Times the attention op alone, with accuracy alongside, so a speed win bought with precision is visible. |
| `scripts/compare_backends.py` | Runs the full harness once per backend and prints the comparison table above. |

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
kernel does this FlashAttention-style: one thread per query row, K/V streamed through
shared memory a tile at a time, with a running max and running sum so the softmax is
computed incrementally.

**Causal tile skipping.** Under causal masking, no thread in a block looks past that
block's last query row, so whole key tiles beyond it are never loaded or computed — not
computed and discarded. This is where the 2.89x on `seq2048 causal` comes from.

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
csrc/fused_attention.cu          custom fused attention CUDA kernel
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

**`head_dim=128` falls back to ATen.** Keeping 128 accumulators plus 128 query values in
registers exceeds what is addressable without spilling, so those shapes take the reference
path inside the extension. This is why `wide d1024` is the weakest custom result.

**SDPA is still faster overall.** The custom kernel loses 5–15% on most shapes because
SDPA's backends use tensor cores for both matmuls while this kernel uses plain FMA. It wins
2–3x on tiny and odd shapes, where launch overhead dominates and simplicity pays. Closing
the rest of the gap means WMMA/tensor-core tiles — at the cost of the fp32 precision that
makes `deep 12L` pass.

---

## Environment

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070, 8 GB, SM 8.6 |
| Driver | 610.47 |
| CUDA Toolkit | 13.0 (V13.0.48) |
| PyTorch | 2.12.0+cu132 |
| Python | 3.10 |
| Compiler | MSVC 14.44 (Visual Studio 2022) |
| OS | Windows 11 |

`torch.compile` and Triton are unavailable in this environment (Triton has no working
Windows build here), which is why the custom-kernel path is C++/CUDA via
`torch.utils.cpp_extension` rather than Triton.
