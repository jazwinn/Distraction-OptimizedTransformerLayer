"""SASS instruction-mix comparison of the head_dim 64 attention kernels.

cuobjdump on the *built object*, not -cubin: tile kernels are lowered after
-cubin would emit, so -cubin gives no SASS for them at all.

Counts are static and per *thread*. Every kernel here runs 128 threads per block
(tile: EIATTR_REQNTID = 0x80; wmma: BLOCK_M/16 warps = 4 at head_dim 64), so the
per-thread numbers are directly comparable -- but only after accounting for the
fact that a bigger block shape does more work per pass. The last table does that,
and the "MACs/tile" column is *verified* rather than assumed: HMMA count x MACs
per HMMA shape x 4 warps reproduces BLOCK_M*BLOCK_N*HEAD_DIM*2 exactly for all
three tensor-core kernels, which pins the emitted body at exactly one key tile.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _cuda_tool(name: str) -> str:
    """Absolute path to a CUDA binary utility, or the bare name as a fallback.

    Prefers the toolkit kernel_ext builds with, because the objects read here
    were produced by it and an older cuobjdump does not necessarily understand
    a newer object's sections. CUDA_PATH / CUDA_HOME, then PATH, stand in when
    that lookup finds nothing.
    """
    exe = name + (".exe" if sys.platform == "win32" else "")
    homes = []
    try:
        import kernel_ext
        homes.append(kernel_ext._find_tile_cuda_home())
    except Exception:                                   # noqa: BLE001
        pass
    homes += [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    for home in homes:
        if home:
            candidate = os.path.join(home, "bin", exe)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which(exe) or exe


CU_FILT = _cuda_tool("cu++filt")
CUOBJDUMP = _cuda_tool("cuobjdump")
OBJS = ("build/tile_attention.cuda.o", "build/fused_attention.cuda.o")
WARPS = 4          # 128 threads / 32, for every kernel in the table

# Kernels whose shared memory is dynamic, so cuobjdump -res-usage
# reports 0. Values derived from WmmaCfg in attention_wmma.cuh.
DYNAMIC_SHARED = {"wmma tf32    64x32": 44800}
THREADS = 128

FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]\s+)?([A-Z][A-Z0-9_.]*)")
RES = re.compile(r"^\s*Function (\S+):\s*$")
RESVAL = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+)")

BUCKETS = [
    ("HMMA",     lambda op: op.startswith("HMMA")),
    ("shared",   lambda op: op.split(".")[0] in
                 {"LDS", "STS", "LDSM", "STSM", "ATOMS"}),
    ("global",   lambda op: op.split(".")[0] in
                 {"LDG", "STG", "LD", "ST", "LDGSTS", "RED", "ATOM", "ATOMG"}),
    ("MUFU",     lambda op: op.startswith("MUFU")),
    ("SHFL",     lambda op: op.startswith("SHFL")),
    ("BAR",      lambda op: op.split(".")[0] in {"BAR", "BARRIER"}),
    # 32-bit float ALU on the CUDA cores -- arithmetic that is NOT on the
    # tensor cores. Conversions (F2F/I2F/F2I) are excluded on purpose: under
    # tf32 those are the narrowing cast, which is a separate story.
    ("scalarFP", lambda op: op.split(".")[0] in
                 {"FFMA", "FADD", "FMUL", "FSET", "FSETP", "FMNMX", "FSEL",
                  "FRND", "FCHK"}),
]

# MACs one warp-level instruction of each tensor-core shape performs, sm_86.
HMMA_MACS = {"1684": 16 * 8 * 4, "1688": 16 * 8 * 8, "16816": 16 * 8 * 16}


def tile(bm, bn, mask, math, split=0):
    want = (f"tile_attention_kernel<(int){bm}, (int){bn}, (int)64, "
            f"(<unnamed>::MaskMode){mask}, (tile_attn::MathMode){math}, "
            f"(bool){split}>")
    return lambda dm: want in dm


# head_dim 64, no mask, single-pass unless noted. Each tile math mode appears at
# *its own* tuned block shape, because that is what actually launches -- fp32
# spills above 32x16 at this head_dim, so forcing one shape on all of them would
# be measuring a configuration nobody would run.
SELECT = [
    ("tile fp32    32x16", 32, 16, tile(32, 16, 0, 0)),
    ("tile tf32   128x32", 128, 32, tile(128, 32, 0, 2)),
    ("tile bf16    64x64", 64, 64, tile(64, 64, 0, 1)),
    ("wmma tf32    64x32", 64, 32,
     lambda dm: "fused_attention_wmma_kernel<float, (int)64>" in dm),
    ("scalar fp32       ", 0, 0,
     lambda dm: "fused_attention_kernel<float, (int)64>" in dm),
    ("tile tf32   128x32 SPLIT", 128, 32, tile(128, 32, 0, 2, 1)),
]


def demangle(names):
    p = subprocess.run([CU_FILT], input="\n".join(names),
                       capture_output=True, text=True)
    return dict(zip(names, p.stdout.strip().splitlines()))


def sass():
    out: dict[str, Counter] = {}
    for obj in OBJS:
        p = subprocess.run([CUOBJDUMP, "-sass", obj],
                           capture_output=True, text=True, cwd=ROOT)
        cur = None
        for line in p.stdout.splitlines():
            m = FUNC.match(line)
            if m:
                cur = m.group(1)
                out.setdefault(cur, Counter())
                continue
            if cur is None:
                continue
            m = INSN.search(line)
            if m:
                out[cur][m.group(1)] += 1
    return out


def resources():
    out = {}
    for obj in OBJS:
        p = subprocess.run([CUOBJDUMP, "-res-usage", obj],
                           capture_output=True, text=True, cwd=ROOT)
        cur = None
        for line in p.stdout.splitlines():
            m = RES.match(line)
            if m:
                cur = m.group(1)
                continue
            m = RESVAL.search(line)
            if m and cur:
                out[cur] = dict(reg=int(m.group(1)), stack=int(m.group(2)),
                                shared=int(m.group(3)))
                cur = None
    return out


def classify(counter):
    row = {c: 0 for c, _ in BUCKETS}
    row["total"] = sum(counter.values())
    row["hmma_macs"] = 0
    row["hmma_shape"] = "-"
    for op, n in counter.items():
        for name, pred in BUCKETS:
            if pred(op):
                row[name] += n
                break
        if op.startswith("HMMA"):
            for shape, macs in HMMA_MACS.items():
                if f".{shape}." in op:
                    row["hmma_macs"] += n * macs * WARPS
                    row["hmma_shape"] = shape
                    break
    return row


def main():
    counts = sass()
    res = resources()
    names = demangle(list(counts))
    rows = [(names.get(m, m), m, classify(c)) for m, c in counts.items()]

    hdr = ["total", "HMMA", "HMMA%", "shared", "global", "scalarFP",
           "MUFU", "BAR", "SHFL"]
    print("static SASS per thread -- head_dim 64, MaskMode::None, sm_86, "
          "128 threads/block")
    print()
    print(f"{'kernel':<26}" + "".join(f"{c:>9}" for c in hdr))
    print("-" * 107)
    found = []
    for label, bm, bn, pat in SELECT:
        hit = [(mg, r) for dm, mg, r in rows if pat(dm)]
        if not hit:
            print(f"{label:<26}{'NOT FOUND':>9}")
            continue
        mg, r = hit[0]
        found.append((label, bm, bn, mg, r))
        pct = 100.0 * r["HMMA"] / r["total"] if r["total"] else 0.0
        print(f"{label:<26}{r['total']:>9}{r['HMMA']:>9}{pct:>8.1f}%"
              f"{r['shared']:>9}{r['global']:>9}{r['scalarFP']:>9}"
              f"{r['MUFU']:>9}{r['BAR']:>9}{r['SHFL']:>9}")

    print()
    print("resources, and the work one emitted loop body covers")
    print()
    print(f"{'kernel':<26}{'MMA shape':>11}{'regs':>6}{'shared KB':>11}"
          f"{'stack':>7}{'blk/SM':>8}{'MACs/tile':>11}{'HMMA covers':>13}"
          f"{'insn/MAC':>10}")
    print("-" * 107)
    for label, bm, bn, mg, r in found:
        u = res.get(mg, {})
        # wmma takes its shared memory *dynamically* (extern __shared__ sized at
        # launch from WmmaCfg::SMEM), so -res-usage reports 0 for it. For
        # scalar_t=float, HEAD_DIM=64: QO 17408 + 2*KV 8704 + S 9216 + 3 row
        # vectors of 256 = 44800 B. The tile kernels' shared is static, so
        # cuobjdump does see theirs.
        shared_kb = (DYNAMIC_SHARED.get(label.strip(), u.get("shared", 0))
                     / 1024.0)
        # sm_86 gives a block at most 99 KB of the 100 KB opt-in shared pool.
        per_sm = int(99.0 // shared_kb) if shared_kb else 0
        macs = bm * bn * 64 * 2
        covers = (f"{r['hmma_macs']}" +
                  ("  ok" if macs and r["hmma_macs"] == macs else ""))
        ipm = f"{r['total'] * THREADS / macs:.2f}" if macs else "-"
        print(f"{label:<26}{r['hmma_shape']:>11}{u.get('reg', 0):>6}"
              f"{shared_kb:>10.1f}K{u.get('stack', 0):>7}{per_sm:>8}"
              f"{macs if macs else '-':>11}{covers:>13}{ipm:>10}")
    print()
    print("MACs/tile = BLOCK_M*BLOCK_N*64*2 (both GEMMs). 'HMMA covers' is "
          "HMMA x MACs-per-shape x 4 warps;")
    print("matching MACs/tile confirms the emitted body is exactly one key "
          "tile, so insn/MAC is exact")
    print("for the three tensor kernels. The fp32 rows have no HMMA to pin "
          "them, so their insn/MAC")
    print("assumes the same one-tile body and is an upper bound.")


if __name__ == "__main__":
    main()
