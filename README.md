# Distraction - *A faster attention layer for Transformers*



A submission to **TikTok TechJam 2026, Problem Statement 3: "Implement a GPU Kernel for a
Transformer Layer."** It runs the same Transformer layer as the reference implementation, produces
the same answers, and runs faster.

- [Project overview](#project-overview)
- [Setup and installation](#setup-and-installation)
- [Reproducing the results](#reproducing-the-results)
- [Limitations, and what I would improve](#limitations-and-what-i-would-improve)
- [Team](#team)
- [Further reading](#further-reading)

---

## Project overview

### The task

A **Transformer** is the design behind most modern AI systems — chatbots, translation, speech,
recommendations. A **kernel** is a small program that runs on the graphics card (GPU) rather than on
the main processor. The task was to write faster kernels for one Transformer layer.

The catch is accuracy. The fast version's output is compared against the reference version's
output, number by number, and every single number must be either **within 0.002** of the reference
or **within 2%** of it. The benchmark runs that check before it times anything, and skips the
timing entirely if it fails — so a speedup number always belongs to a correct result.

> Make it faster without meaningfully changing the answers.

That constraint shapes almost every decision here, because the obvious ways to gain speed — using
less precise numbers, taking mathematical shortcuts — are exactly the ways to fail the check.

### Why the original is slow

The heart of a Transformer is **self-attention**. To work out what "it" refers to in a sentence,
every word compares itself against every other word and scores how relevant each one is. Those
scores form a grid: one number for every pair of words.

That grid is the problem. Double the length of the text and the grid *quadruples*, because it grows
in both directions at once. At 2048 words it is 128 MiB for a single layer — and the reference
implementation sends it out to the card's main memory and reads it back five separate times, once
each to create it, rescale it, mask it, convert the scores to weights, and finally use it.

Measuring the reference showed that real arithmetic falls from 70% of the time at short inputs to
**under a third** at long ones. Everything else is data being moved around.

At the other extreme, very small inputs are slow for the opposite reason: the card finishes each
small piece of work before the next one arrives and spends most of the pass waiting to be given
something to do.

### What was built

Attention was rewritten so that the grid of scores is built, masked, scored and used **in small
pieces that never leave the chip's fast on-chip memory**. Three complete versions of it exist:

| Version | What it is |
| --- | --- |
| **Tensor-core kernel** | The default. Runs both of attention's multiplications on the card's dedicated matrix-multiply hardware. |
| **Simple kernel** | One thread per word, on the general-purpose units. Covers older cards and unusual sizes, and acts as the reference the faster ones are checked against. |
| **Tile kernel** | The same maths written in a newer, higher-level style NVIDIA offers. Kept as a comparison between ease of writing and speed. |

**There is no prebuilt attention anywhere in this project.** Most libraries would hand an awkward
input size to an existing implementation; this one does not, because implementing attention *is* the
task. A size nothing covers is reported as an error rather than quietly served by somebody else's
code.

Around that sit nineteen further changes — combining neighbouring steps into single kernels, doing
three small multiplications as one large one, reading data where it already sits instead of copying
it, skipping regions of the grid that would only be thrown away, and recording the whole sequence of
GPU commands once so it can be replayed as a single instruction. Each one is listed with its
measured effect in [TechnicalReport.md](TechnicalReport.md), section 5.

### The machine everything was measured on

Every number in this project comes from one machine:

| Part | Detail |
| --- | --- |
| **Graphics card** | **NVIDIA GeForce RTX 3070, 8 GB** — Ampere generation, driver 610.47 |
| Processor | AMD Ryzen 7 5800X |
| Memory | 32 GB DDR4 |
| Operating system | Windows 11 |
| GPU toolkit | NVIDIA CUDA 13.3 |
| C++ compiler | Microsoft Visual C++ 14.44 |
| Framework | PyTorch 2.12, Python 3.10 |

The graphics card is the part that matters. It sets which kernels can run at all, and it sets the
speedups: the largest gains here come from keeping the card busy, so a faster card would starve at
larger input sizes and a slower one at smaller ones. Every tuned constant in the project was chosen
by measurement on this card, and would need re-measuring on a different one.

### Results

### What is in the repository

| Location | What is in it |
| --- | --- |
| `csrc/` | The GPU code — every kernel, about 5,700 lines |
| `optimized/` | The Python side: the rewritten layer, and which kernel runs when |
| `scripts/` | Correctness checks and measurement scripts |
| `dashboard/` | A local web page for running benchmarks and reading the results |
| `torch_transformer_benchmark.py` | The organizers' benchmark, with this project wired into it |

---

## Setup and installation

### What you need

| Requirement | Detail |
| --- | --- |
| CUDA Toolkit | 13.3 or newer |
| PyTorch | 2.12 with CUDA support |
| Python | 3.10 or newer |
| Compiler | MSVC — Visual Studio with the "Desktop development with C++" workload |
| Build tool | `ninja` |

The GPU code is compiled on your own machine on first use, so a C++ compiler is genuinely required.
There is no PyTorch-only shortcut: the kernels *are* the project, and a build that fails stops the
benchmark rather than quietly measuring something else.

### 1. Install the Python packages

```bash
pip install -r requirements.txt
```

That installs PyTorch and `ninja`, and nothing else — everything else this project uses comes with
Python itself.

The version of PyTorch has to match your CUDA Toolkit, so `requirements.txt` points pip at NVIDIA's
build of it rather than the default one. If your toolkit is not 13.x, change the `cu132` in that
file to match. A mismatch between the two is the usual cause of build failures, and plain
`pip install torch` on Windows gives a version with no GPU support at all.

### 2. Build the GPU code

```bash
cmd.exe /c scripts\build_ext.bat
```

It should finish with:

```
[build_ext] OK -> ...\build\transformer_kernels.pyd
```

The first build takes about 70 seconds. After that it is near-instant, and editing any GPU source
file triggers a rebuild automatically — there is no separate build step to remember.

Visual Studio's compiler is not available in an ordinary terminal, only in its own developer
prompt. `scripts\build_ext.bat` finds it and sets that up for you, which is why the command goes
through the script rather than calling Python directly.

### 3. Check it works

```bash
cmd.exe /c scripts\devenv.bat python scripts\verify_kernel.py
```

This runs every kernel against a reference on fifteen different input sizes and compares the results
number by number. It should end with:

```
every kernel matches the reference on every case
```

Two more checks are worth running after a build:

```bash
cmd.exe /c scripts\devenv.bat python scripts\verify_attn_axes.py   # every kernel and format combination
cmd.exe /c scripts\devenv.bat python scripts\verify_graph.py       # the recorded version matches the normal one exactly
```

### If something goes wrong

**"the CUDA extension failed to load, and there is no fallback."** The compiler was not found. Run
the command through `scripts\devenv.bat`, which puts it on the path first:

```bash
cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
```

**"unsupported Microsoft Visual Studio version", or a crash inside `cudafe++`.** NVIDIA's compiler
only accepts a range of Visual Studio versions, and Visual Studio defaults to the newest one it has.
The build script pins version 14.44; change it if your CUDA version wants a different one:

```bash
set VCVARS_VER=14.43
cmd.exe /c scripts\build_ext.bat
```

If that version is not installed, the script lists the ones that are.

**"'vswhere.exe' is not recognized."** Harmless. The build succeeds anyway.

**It runs, but is slower than expected.** Check which kernel actually ran by passing `--attn-impl`.

---

## Reproducing the results

### The easiest way

```bash
python -m dashboard
```

This opens a local web page. To run the full set of official test cases:

1. Go to the **Run** tab.
2. In the **Shape** card, switch the toggle from *One shape* to **Many shapes**.
3. Leave every preset selected — those are the fourteen official test cases, ticked by default.
   Anything the current settings cannot run is greyed out and skipped.
4. In the **Optimizations** card, leave everything on **auto**. That is the configuration the
   results were measured with, and the one the benchmark uses by default.
5. Press **Run benchmark**.

The cases run one at a time and fill in a row each: whether it passed the accuracy check, its worst
error, how many numbers failed, the two timings, and the speedup. Above the table, the summary
gives the geometric mean across them.

The page does not change how anything is measured. Every number in it is printed by the benchmark
itself, and the exact command is shown before it runs, so any row can be traced back to something
you could type into a terminal yourself. The server never touches the graphics card; each run is a
separate process that exits and gives its memory back when it finishes.

The **Compare** tab does the same for two configurations side by side, and **Profile** shows where
the time inside a single pass actually goes.

### From the command line

One configuration at a time:

```bash
cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py --batch-size 64 --seq-len 128 --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --causal
```

That is the first of the official test cases. All fourteen are listed in
`dashboard/presets.json` with a note on what each one varies. The benchmark checks accuracy first
and skips the timing entirely if it fails, so a speedup number always belongs to a correct result.

The largest case — 32 sequences of 100,000 words — needs no special flags. The benchmark works out
that it will not fit in memory, splits the batch into pieces, and skips the reference implementation
because that cannot run at all.

To measure one change on its own, four switches turn the main optimizations on and off:

| Switch | What it changes |
| --- | --- |
| `--attn-impl` | Which attention kernel runs: `auto`, `scalar`, `wmma` (tensor cores) or `tile` |
| `--attn-precision` | How precisely it calculates: `auto`, `fp32` (full precision), or `tf32`, `fp16` and `bf16` (progressively less, and progressively faster) |
| `--linear-gelu` | Whether two steps of the feed-forward part are fused into one kernel |
| `--cuda-graph` | Whether the GPU command sequence is recorded and replayed |

Forcing a kernel onto a size it does not cover reports an error rather than silently using a
different one, so a measurement can never be attributed to the wrong kernel.

### Rules that keep the numbers honest

Three rules sit behind every figure quoted here, and the dashboard enforces all three:

1. **Never compare timings from separate runs.** Graphics cards slow themselves down as they heat
   up, so two versions have to be timed alternately within one session to face the same conditions.
   One block size appeared to win by 1.58× measured separately, and placed fourth of five when timed
   properly.
2. **Always run a control.** Run the same version against itself. The true answer is exactly 1.00×,
   so whatever it actually reports is how much the machine is varying at that moment — and any real
   difference has to beat that to mean anything.
3. **Timing and profiling are different jobs.** Profiling tools inflate the numbers they collect, so
   the Profile tab reports proportions and never a speedup.

---

## Limitations, and what I would improve

**The matrix-multiply hardware is barely used.** The kernels do reach the card's dedicated
matrix-multiply units, but the counters show they spend only 13% of their time actually running
them, and occupy 22% of what the card could be doing at once. Every kernel in the model turns out to
be limited by memory speed rather than by arithmetic. Two things follow: there is real headroom
left, and it will not come from doing less arithmetic — the kernels are waiting on data. Further
gains have to come from moving less of it, or from keeping more work resident so the waiting
overlaps with something useful.

**The accuracy budget is mostly spent before this project does anything.** The allowance is a
difference of 0.002. The reference implementation's own rounding already sits between 0.00098 and
0.0012 away from a mathematically exact answer — over half the budget.

That margin cannot be traded for speed. The obvious next step would be an even narrower number
format, and the one below the current choice is unusable: measured against the same budget it lands
at 425% to 622%, with tens of thousands of failing values, and that is for a mathematically perfect
implementation. It also means the margin is not entirely under this project's control — a different
driver moves a number that is already more than half spent.

**The design grew rather than being planned.** I had a solid foundation in the basics of GPU
programming, but the specific ground this project stands on was learned over a few days while
building against it. Several of the rules that decide which kernel runs exist because of a
measurement rather than because the structure needed them. The results are measured and they hold,
but a second attempt would be simpler and — given that the kernels are nowhere near the card's
limits — faster.

**Given more time, that is where it would go**, before it went on new features.

---

## Team

**Solo project.** Everything here — the GPU code, the Python layer, the measurement scripts and the
dashboard — was designed, written, measured and debugged by **Ng Jaz Winn**, the sole entrant.
`torch_transformer_benchmark.py` is the organizers' file, changed only by the few lines that connect
this project to it.



---

## Further reading

| Document | What it covers |
| --- | -
| **[TechnicalReport.md](TechnicalReport.md)** | The full write-up: the problem, the tools used, why the original is slow, all twenty optimizations and what each was worth, results, limitations, and the AI-use disclosure. **Start here.** |
| [attention_dispatch_graph.md](attention_dispatch_graph.md) | Which kernel runs for which input, as a diagram |
| [dashboard/README.md](dashboard/README.md) | What each tab of the dashboard does |
| [Record.md](Record.md) | The engineering record: every measurement in the order it was taken, including the ones that came back negative |
| [csrc/TUNING.md](csrc/TUNING.md) | The measurements behind every tuned constant in the kernels |
