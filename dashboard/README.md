# Benchmark dashboard

A local web front end for `torch_transformer_benchmark.py` and the scripts in
`scripts/`. Pick a shape, tick the optimizations, run, read a table.

```
python -m dashboard
```

Opens `http://127.0.0.1:8000`. No new dependencies — everything here is Python
standard library plus one hand-written HTML page.

```
python -m dashboard --port 8123    # if 8000 is taken (it also auto-advances)
python -m dashboard --no-browser   # do not open a tab
```

## What it does not do

It **does not change how anything is measured.** Every number in the table is
printed by the harness itself; this reads that output and lays it out. Nothing
in `optimized/`, `csrc/`, `kernel_ext.py` or `scripts/` was modified to make this
work.

Two edits exist outside this folder. `dashboard/runs/` is in `.gitignore`, and
`torch_transformer_benchmark.py` gained an NVTX wrapper for the Profile tab —
about twenty lines in one place, inert unless `BENCH_NVTX=1` is set, which only a
profile run does. A normal run constructs nothing and behaves exactly as before.
Without it a profiler cannot tell the baseline's kernels from the optimized
model's, because they run in one process and share their ATen and cuBLAS kernels
by name.

The command it is about to run is always shown before it runs, so any row in the
table can be traced back to something you could paste into a terminal yourself.

## GPU cost

The server never imports torch and never opens a CUDA context. Confirm it:

```
python -m dashboard &
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The dashboard's PID is not in that list. Each benchmark runs in a child process
that exits when it finishes, giving its memory back; between runs the GPU is as
idle as if the dashboard were closed.

The queue runs **one child at a time**. That is a measurement rule, not a
simplification: two benchmarks sharing a card contend for SMs and bandwidth and
both report numbers that mean nothing.

## The tabs

**Run** — one configuration. Shape, dtype, tolerances, iteration counts, and an
Optimizations panel covering `--attn-backend`, `--attn-impl`, `--attn-fp16`,
`--linear-gelu` and `--cuda-graph`.

**Compare** — two configurations against the same shape or shapes, run one after
the other. The toggle in the Shape card picks between *One shape*, which is the
typed shape fields, and *Many shapes*, which is the same tickable preset list
Sweep uses. Results pivot: a row is a shape, and each config gets its own column
group — median, speedup and accuracy for A beside the same three for B, then the
A-vs-B ratio and the verdict.

Leave *Run a control* ticked: it runs config A a second time, and the A/A ratio
it reports is the noise floor for your machine at that moment. Its true value is
1.000x, so whatever it comes back as is the bar an A-vs-B difference has to clear
before it means anything. This is the same control column every A/B script in
`scripts/` prints, and `csrc/TUNING.md` puts the floor near 4.3%.

Across many shapes the control runs **once per shape**, because the floor
belongs to the shape rather than the machine — a 32-token sequence is far noisier
than a long one, and a floor borrowed from a large shape would make the small
rows look conclusive when they are not. That makes the arithmetic three runs per
shape rather than two; the count beside the Run button says what is about to be
queued, and 96 runs is the cap.

Runs are ordered A, B, control for one shape before moving to the next, so the
pair being compared is timed within seconds of itself. If a shape is impossible
for either side — an `--attn-impl` that does not cover its head_dim, say — the
whole shape is skipped with the reason shown, and the rest of the selection still
runs. Half a pair is not a comparison.

**Sweep** — one configuration across many shapes, one child process per shape,
rows filling in as each finishes.

**Profile** — one traced run under Nsight Systems, and where its GPU time
actually goes. Three steps: *prepare* builds the extension, *capture* runs the
harness under `nsys`, *analyse* turns `nsys stats` into tables.

The headline is the share of a forward the GPU spends running a kernel. Above
about 80% the time is in the kernels themselves and the table below says which;
under about 50% the shape is launch-bound, and no amount of kernel optimization
will move it much. Everything is the **median forward**, never the total: the
first forward pays for cuBLAS and module loading, and one measured run had it at
24 s against 4.7 ms for every other forward.

Four things about it are worth knowing:

* **Traced timings are not benchmark timings.** Tracing inflates them, so the
  capture step's output is not parsed and no speedup is shown. Use Run for
  numbers and Profile for proportions.

* **The iteration counts are deliberately small** (`--accuracy-trials 1
  --warmup 1 --repeats 5 --benchmark-rounds 1`). Tracing records every launch and
  the analysis pays for each one.

* **Attribution comes from NVTX.** Both models run in one process and share ATen
  and cuBLAS kernels by name, so the only thing separating them is the range the
  harness emits when `BENCH_NVTX=1`, which the dashboard sets. A run without it
  emits nothing and the view says so.

* **CUDA graphs are traced at node granularity.** `nsys` records a graph launch
  as one opaque entry by default, which made a graphed forward read as "2 kernels"
  and hid everything inside it. `--cuda-graph-trace=node` is not optional here.

The traced harness runs through `_profile_shim.py`, and it has to. Torch
resolves the MSVC linker by shelling out to `where cl`; inside a profiled
process that returns empty output, and torch's guard (`len(cl_paths) >= 1`, true
even for `[""]`) turns that into `command = "/link.exe"`. The link fails, the
extension does not load. That used to fall back to a prebuilt attention
**without reporting an error**, so the profile measured ATen and looked
entirely plausible; it now raises instead, which turns this into a loud failure
rather than a misleading profile. The
shim answers that one call from `shutil.which` and touches nothing else.

If the summary ever says **no custom kernels ran**, that is what has happened
again. The *launch through `scripts/devenv.bat`* checkbox is a second thing to
try, though it did not help this particular failure. Which kernels count as
yours is read from `csrc/` by scanning for `__global__` definitions, so renaming
or adding one needs no change here.

Reports accumulate in `runs/` as `<job-id>.nsys-rep` plus a `.sqlite` export,
about a megabyte per run. The Report card shows the size, opens the trace in the
Nsight Systems GUI, and deletes both files.

### Kernel counters

*Kernel counters* runs the same shape under **Nsight Compute** and answers what
the timeline cannot: whether a kernel is limited by arithmetic, by bandwidth, or
by neither. It reports compute and memory as percentages of this card's peak,
achieved against theoretical occupancy, registers per thread, and a verdict —
plus Nsight Compute's own guidance, one line each with the reasoning on hover.

ncu replays every kernel it profiles, once per section, so two limits keep it
finite: it profiles **only kernels defined in `csrc/`** (the names are read from
there) and only the first N launches, 12 by default. Even so it is far slower
than the timeline run; that is the tool, not the wrapper.

Read it as: above ~80% of either pipe the kernel is near the roofline and only
less work will help; both under ~40% means it is waiting rather than working,
and occupancy is the first thing to look at. Those thresholds are Nsight
Compute's own.

**Counter collection needs permission.** It returns `ERR_NVGPUCTRPERM` unless
the process is elevated, or `RmProfilingAdminOnly` is 0 under
`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` **and the
driver has reloaded** — setting the value alone changes nothing until a reboot.

Neither of those can be read off and trusted, so the Profilers card does not
try: *test counters* collects one for real and reports what happened, and until
you press it the card says *not tested*. A run that is refused shows the refusal
rather than an empty table.

**Scripts** — every `scripts/*.py`, with a form built from its own argparse
where it has one and a free-text box where it does not. Output is streamed raw;
each script prints its own table and this does not try to reinterpret them.

**History** — finished jobs from `runs/history.jsonl`, newest first. Full logs
sit beside it as `runs/<job-id>.log`.

## Kernel switches with no command-line flag

Several optimizations have no argparse flag at all. They are environment
variables the extension reads at startup:

| variable | read in |
| --- | --- |
| `TILE_SPLIT_KV` | `csrc/tile_attention.cu` |
| `WMMA_CAUSAL_REVERSE`, `WMMA_FP16` | `csrc/attention_wmma.cuh` |
| `LAYERNORM_FUSED_REDUCE`, `LAYERNORM_WARP_WIDTH`, `LAYERNORM_WARP_ROWS`, `LAYERNORM_BLOCK_THREADS` | `csrc/add_layernorm.cuh` |

Because each run is a fresh child process, the dashboard can set these in that
child's environment — so they are checkboxes here, under *Kernel switches* in
each Optimizations panel, with no change to `csrc/`. A switch you have not
touched is not put in the environment at all, so an untouched run's command is
byte-for-byte the one the harness would get by hand.

## Stopping things

**Stop button** — kills the child's whole process tree (`taskkill /F /T` on
Windows, because a harness run that rebuilds the extension has ninja and cl.exe
underneath it). It reports what happened: if the job had already finished, or
the request failed, the button says so rather than sitting there looking like it
worked. Stopping a queued job that has not started yet also works.

**Ctrl-C on the server** — kills whatever benchmark is running on the way out.
Without that, quitting the dashboard would leave the child holding the GPU with
no UI left to stop it from.

**A server that was force-killed or crashed** cannot run its own cleanup, so the
next startup reaps the orphan: the running child's pid and argv are recorded in
`runs/running.json`, and startup kills that process if it is still alive *and*
its command line still matches. Both conditions, because Windows recycles pids
and a pid match alone could take out something innocent. It prints what it
killed.

**What the Stop button cannot reach:** a benchmark you started yourself in a
terminal. The dashboard only knows about children it spawned. Those are
identifiable — it always launches with an absolute interpreter path and `-u`:

```
'C:\...\python.exe' -u 'C:\...\torch_transformer_benchmark.py' --batch-size 8 ...
```

A hand-typed `python torch_transformer_benchmark.py ...` is not one of them.
If Ctrl-C will not stop such a run, it is usually blocked in a CUDA call that
does not return — Python cannot service the interrupt until it does. Kill it by
pid:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*torch_transformer_benchmark*' } |
  Select-Object ProcessId, CommandLine

taskkill /F /T /PID <pid>
```

## Preflight

Before anything is spawned, the server checks the combination and blocks the
ones that cannot produce a number. Each of these otherwise costs 5–15 s of torch
startup to discover:

- a forced `--attn-impl` outside its head-dim coverage (wmma/scalar cover
  `{8,16,32,64,128}`, the tile kernels `{8,16,32,64}`) — the kernel raises
  rather than falling back, by design
- `d_model` not divisible by `heads`
- a tile impl under `--dtype float16`/`bfloat16`, where the launcher raises
- `--compile-user` together with an explicit `--cuda-graph`
- a shape whose estimated footprint exceeds the card

The memory rule is not the dashboard's own. `auto_stream_rows()` and
`baseline_can_run()` in the harness decide how to slice a batch and whether the
baseline can run, from `_PEAK_ACTIVATION_FACTOR` and `_MEMORY_BUDGET_FRACTION`;
the dashboard reads those two constants out of the source and applies the same
arithmetic, so retuning either there moves the prediction here. It predicts
6.1 GiB for grading shape 6 — the same figure the harness's own comment records.

That leaves exactly one memory error: **a single row not fitting**. The harness
streams a batch that does not fit and skips a baseline that cannot run, so
neither is fatal any more; it cannot stream below one row, and that is the wall.

Warnings do not block. "This will run but will not measure what you think" —
a tile mode under fp16 that silently falls back, say — is a judgement left to
whoever is at the keyboard.

## Presets: the 14 grading shapes

`presets.json` holds the official Appendix test shapes, transcribed from the
grading table. The appendix column *QKV Dim* is `d_model` here, and `ffn_dim`
equals it in every shape — so there is no 4x FFN expansion anywhere in the set,
and attention plus per-layer overhead dominates rather than the FFN. All 14 are
causal; `layers` is 4 except shape 14, which is 2.

Shape 1 is the base and the rest vary one axis from it: 2–6 batch, 7–8 d_model,
9–11 heads, 12–13 seq_len, and 14 is its own thing.

`head_dim = d_model / heads` decides which attention kernel can run each shape —
wmma and scalar cover `{8,16,32,64,128}`, the tile kernels `{8,16,32,64}` — so
the Sweep list shows it per row:

```
shape            head_dim   note
1-6, 12, 13            32
7, 11                   8
10, 14                 64
9                     128   tile kernels cannot take it
8                     256   wmma only; tile kernels cannot take it
```

**Shape 14 runs**, and the list says how. Its full `[32,100000,1024]` input is
12.2 GiB, so the harness streams it as 32 slices of one row; its baseline's
scores are 596 GiB *even one row at a time*, so the baseline is skipped. The
shape list marks both with a badge, because each changes what the row's numbers
mean:

- **`32 slices`** — the reported median is per slice. The whole-batch figure is
  shown beside it.
- **`no baseline`** — real latencies, but no speedup and no accuracy verdict,
  because there is nothing to compare against.

Measured through the dashboard: 32 slices of 2.93 s, 93.7 s for the full batch,
34,158 token/s, peak 3.8 GiB of an 8 GiB card.

The file is re-read on every request, so edits appear on the next page load with
no restart. Keys: `name`, `batch_size`, `seq_len`, `d_model`, `heads`, `ffn_dim`,
`layers`, `causal`, and an optional `note`. Anything else is ignored.

### Editing them

The **Presets** tab is a table over the same file: edit a cell, *add shape* to
append one (seeded from the last row, since shapes in this set differ from each
other by one axis), `×` to remove one, *Save* to write. *Revert* reloads the file
and throws away unsaved edits. `head_dim` is computed per row as you type, so a
`d_model` that is not divisible by `heads` shows up immediately rather than at
save time.

Writing to a file from a web page is worth being careful about, so:

- **Nothing is written until every row is valid.** A rejected save changes the
  file not at all, and each problem is reported against the field it came from —
  `row 2 [heads] d_model 512 is not divisible by heads 7` — and marks that input
  red. Partial saves are not offered: silently dropping the rows that failed is
  worse than saving nothing.
- **The write is atomic.** A temporary file in the same directory, then
  `os.replace`, so a reader sees either the old file or the new one and never a
  half-written one. The previous contents are kept as `presets.json.bak`.
- **Validation lives on the server** (`presets.validate`), not in the browser, so
  the rule that decides what lands in the file is the same one that reports
  problems. It checks names are present and unique, that every dimension is a
  whole number of at least 1 and under a sanity cap, and that `d_model % heads
  == 0` — the harness's own rule, so a preset cannot be saved that
  `TransformerConfig.validate()` would reject the moment it ran.
- **The `_comment` block is preserved** across saves. It documents the format for
  anyone who opens the file in an editor, and losing it to a UI save would be a
  slow-acting papercut.

After a successful save the shape list and both preset dropdowns rebuild from the
response, so nothing keeps offering a shape that no longer exists. Switching to
the tab reloads from disk unless you have unsaved edits, which picks up a change
made in a text editor while the page was open.

## Running many shapes at once

The **Sweep** tab takes one configuration and runs it across every shape you
tick. Everything runnable starts ticked, so pressing Run sweeps the whole
grading set without a prior edit.

They run **one at a time**, not concurrently, and that is deliberate: two
benchmarks sharing the card contend for SMs and bandwidth, and both then report
numbers that mean nothing. Rows fill in as each finishes, so a long sweep stays
readable while it runs, and Stop ends it between shapes.

A shape preflight would refuse is greyed out with its reason rather than
silently skipped, and if you queue one anyway the sweep drops that row and
carries on with the rest — one uncovered `head_dim` should not cost you the
other twelve.

The whole 13-shape runnable set takes a couple of minutes at *quick · 5/20/1*
and correspondingly longer at *full · 20/100/3*. Use quick to find which shapes
are interesting, then re-run those few at full settings — quick is too noisy to
separate two close configurations.

## Reading the controls

Every control starts on the value that is **actually in force** — no row saying
"default", and no row explaining what leaving it alone would do. The dropdown
simply opens on the mode the run will use.

For the five optimization flags that value is not argparse's. Argparse reports
`default=None`, because the real setting lives in `optimized/config.py` and
`cli.py` only overrides it when the flag is passed. The dashboard reads the
constant out of `config.py` and preselects it.

Picking the preselected value back does not lengthen the command: `build_argv`
drops any value equal to the effective one, so an untouched form still runs a
bare `python -u torch_transformer_benchmark.py`. A field only highlights as
changed when it will genuinely alter the run.

It also reads the **per-mode explanations `config.py` already carries** in the
comment beside each constant, and shows them: as the tooltip on each dropdown
option, and in full underneath the field for whichever mode is selected —
including `auto`, whose name says nothing about what it does. Selecting
`--attn-impl auto` displays:

> auto: the first kernel that covers the shape: the tensor-core kernel wherever
> it applies, the scalar kernel where that is all there is, and an error where
> neither does. It used to prefer SDPA from head_dim 128 up, which is gone.

That text is not written here. It is the comment above `ATTENTION_IMPL` in
`optimized/config.py`, parsed at page load, so there is no second description to
drift out of date. Rewording that comment rewords the tooltip.

Number fields work the same way: the greyed placeholder is the value the run
will use if you leave the box empty, and hovering says where it comes from.

## How the form stays in sync

Nothing here hard-codes the harness's flags. `argspec.py` parses
`torch_transformer_benchmark.py` and `optimized/cli.py` with `ast` and reads
their `add_argument` calls — source is parsed, never executed, because importing
`optimized.cli` would pull in torch and defeat the whole point.

So **adding a flag to `optimized/cli.py` makes it appear in the dashboard on the
next page load**, with its `help=` text as the tooltip. There is no second copy
of the flag list to forget to update. If the extraction ever comes back
implausibly short — a renamed function, a moved file — a built-in fallback list
is used and the page says so.

To add a new *environment-variable* knob, add one entry to `ENV_KNOBS` in
`knobs.py`; those have no argparse to read.

## Layout

```
__main__.py   startup; binds 127.0.0.1 and nothing else
server.py     the HTTP routes
jobs.py       the serial queue and the child processes it spawns
runspec.py    a form submission -> an argv list and an environment
argspec.py    the harness's argparse calls, read out of the source
knobs.py      the env-var knobs, and the preflight rules
parse.py      harness stdout -> a row of numbers
presets.py    saved shapes
static/       one HTML page, one stylesheet, one script
runs/         per-job logs and history.jsonl (gitignored)
```

`argspec`, `knobs`, `parse` and `presets` import nothing from the rest of the
package and can be read on their own.

## Parsing

`parse.py` reads the harness's printed output. That format is already
load-bearing elsewhere — `scripts/compare_backends.py` scrapes the same
`summary: (PASS|FAIL)` and `speedup\s+: ([\d.]+)x` lines — so this is the
existing contract read more completely, not a new one. A change to the harness's
output breaks both together, and the mismatch is visible because the raw log sits
next to the parsed table in the UI.

## Security

The server exists to run programs on request, so it binds loopback only and
there is no flag to change that. Subprocesses are launched from argument lists,
never through a shell; script names are matched against a real listing of
`scripts/`; every form value is checked against the flag spec it claims to fill
and dropped if it does not fit.
