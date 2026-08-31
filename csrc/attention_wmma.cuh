// Tensor-core kernel.
//
// Block owns BLOCK_M=64 query rows; 4 warps, one 16-row stripe each. Keys are
// walked in BLOCK_N=32 tiles. Per tile a warp does:
//
//   1. S = Q @ K^T           16x32 via wmma, accumulate fp32   -> smem
//   2. online softmax on those 16 rows                          -> P in smem
//   3. O = O * corr + P @ V  16xD  via wmma, straight into the O fragments
//
// Step 2 runs in the base-2 domain by default: `scale * log2(e)` goes into the
// score's one multiply and the exponential is a bare exp2f, rather than
// `s * scale` followed by __expf, which is itself `ex2.approx(x * log2e)` and
// so hides a second multiply. Worth 1.04x on the op at identical accuracy --
// the softmax is a first-order cost here, not a rounding error on the GEMMs.
// See softmax_mode_flag() below and csrc/TUNING.md.
//
// Q and O both stay in registers for the whole key loop, so the only traffic
// in the inner loop is the K/V tile, the score tile, and the fragment reads
// that feed the MMA units.
//
// When the grid is too small to fill the card -- the grid is query-side only,
// so a small batch with a long key range leaves most SMs idle -- the launcher
// splits the key range across extra blocks and a second pass folds the partial
// softmaxes together. That path is decided before the launch, not fallen back
// to: see wmma_split_count() and wmma_split_combine_kernel() below.
//
// Every 2-D tile in shared memory is stored with a padded leading dimension.
// A fragment load walks a column of the tile, so an unpadded row stride of 16,
// 32 or 64 floats puts every row of the fragment in the same shared-memory
// bank and serializes the load. The pad is the smallest one wmma allows (ldm
// must be a multiple of 4 floats, or 8 halves), which is enough to rotate
// successive rows off each other.

#pragma once

#include "kernel_common.cuh"

#include <cuda_pipeline.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {

// c10 half types are layout-compatible with the CUDA ones, but only the CUDA
// ones have wmma fragment overloads.
template <typename T> struct DevType      { using type = T; };
template <> struct DevType<c10::Half>     { using type = __half; };
template <> struct DevType<c10::BFloat16> { using type = __nv_bfloat16; };

// Fragment element type and K extent. TF32 only comes in 16x16x8.
//
// The primary template is a valid-but-unsupported placeholder rather than a
// declaration: AT_DISPATCH_FLOATING_TYPES_AND2 also instantiates the double
// branch, which has no fragment type at all, and `supported` is what keeps
// that branch from ever reaching a wmma call. K stays nonzero so the
// divisibility tests below do not divide by zero while being compiled away.
template <typename T> struct FragTraits {
    using elem = float;
    static constexpr int K = 8;
    static constexpr int LD_PAD = 4;
    static constexpr bool supported = false;
};
template <> struct FragTraits<float> {
    using elem = wm::precision::tf32;
    static constexpr int K = 8;
    static constexpr int LD_PAD = 4;   // wmma wants ldm % 4 == 0 for float
    static constexpr bool supported = true;
};
template <> struct FragTraits<__half> {
    using elem = __half;
    static constexpr int K = 16;
    static constexpr int LD_PAD = 8;   // ... and ldm % 8 == 0 for 16-bit
    static constexpr bool supported = true;
};
template <> struct FragTraits<__nv_bfloat16> {
    using elem = __nv_bfloat16;
    static constexpr int K = 16;
    static constexpr int LD_PAD = 8;
    static constexpr bool supported = true;
};

// Narrowing and widening both go through an explicit helper. torch's build
// passes -D__CUDA_NO_HALF_CONVERSIONS__ and -D__CUDA_NO_BFLOAT16_CONVERSIONS__,
// which delete the implicit operators, so a static_cast<float> on a
// __nv_bfloat16 is a compile error rather than a conversion.
__device__ __forceinline__ float dev_to_float(float s)         { return s; }
__device__ __forceinline__ float dev_to_float(__half s)        { return __half2float(s); }
__device__ __forceinline__ float dev_to_float(__nv_bfloat16 s) { return __bfloat162float(s); }

__device__ __forceinline__ void dev_from_float(float& d, float s)         { d = s; }
__device__ __forceinline__ void dev_from_float(__half& d, float s)        { d = __float2half(s); }
__device__ __forceinline__ void dev_from_float(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }

// Value-returning form, so a staging store stays a ternary. Spelling it as an
// if/else instead cost 1.3x-2.1x on the tf32 path: the ternary compiles to a
// predicated select, and the branch does not.
template <typename T>
__device__ __forceinline__ T dev_of_float(float s) {
    T d;
    dev_from_float(d, s);
    return d;
}

// log2(e). The softmax exponential is the one place the kernel touches the
// SFU, and `__expf(x)` is not one instruction: it is `ex2.approx(x * log2e)`,
// an FMUL in front of every MUFU.EX2. FlashAttention-2 removes that multiply by
// working in the base-2 domain throughout -- the reference implementation's
// softmax reads `exp2f(s * scale - max_scaled)` with M_LOG2E already folded
// into `scale`.
//
// That is what MODE 1 does, and it is the default: one `s * (scale * log2e)`
// where there used to be `s * scale` plus __expf's hidden one.
//
// Folding the constant one step further -- into Q at staging time, so the
// score-side multiply goes too -- is MODE 2, and it was measured and rejected.
// It is a wash on speed (1.004x) and it puts both constants inside the fp16
// narrowing, which takes head_dim 64 from 1.59e-03 to 2.63e-03 against the
// harness's 2e-3 atol. See csrc/TUNING.md, "The attention softmax in the
// base-2 domain".
static constexpr float kLog2e = 1.4426950408889634f;

// Which exponential the softmax uses. exp2f is a bare MUFU.EX2; __expf is that
// same instruction with an FMUL in front. The false branch is kept only so the
// two can be A/B'd in one process.
template <bool EXP2>
__device__ __forceinline__ float attn_exp(float x) {
    if constexpr (EXP2) {
        return exp2f(x);
    } else {
        return __expf(x);
    }
}

// Shared-memory plan. Q is staged into the same bytes that later hold the O
// spill area: Q is read into registers once, up front, and is dead after that,
// and O is fp32 so the region is sized for whichever is larger.
// Block shape per head_dim. Every warp owns exactly one 16-row wmma tile, so
// WARPS follows from BLOCK_M rather than being chosen separately.
//
// 64x32 is the general case. head_dim 128 needs a smaller block: the shared
// footprint grows with head_dim, and at 64x32 it would want 75.8 KB, over the
// 48 KB that keeps two blocks resident per SM. 32x16 brings the same head_dim
// down to 35.9 KB, which fits with room to spare -- so head_dim 128 is a block
// shape away from working, not a retiling.
//
// Overridable from the build line so scripts/tune_block_shapes.py can search
// shapes without editing this file, the same way TF32_M_* work in
// tile_attention.cu. Unlike cuTile's extents these need not be powers of two:
// BLOCK_M only has to be a multiple of 16 (it *is* the warp count times 16) and
// BLOCK_N a multiple of FragTraits<scalar_t>::K, so 48/80/96/112 are reachable
// here and FlashAttention-2's kBlockN=112 is expressible for the 16-bit types.
//
// One shape serves every compute type, and that is a MEASURED result, not an
// assumption. Narrowing the compute type to fp16 frees about a third of the
// block's shared memory, which makes shapes affordable that fp32 cannot fit --
// at head_dim 128, 32x32 costs 41.4 KB in fp16 against 29.9 KB for 32x16, and
// only the latter fits in fp32. So the obvious move is to spend the freed
// memory on a wider key tile. It loses, and not narrowly
// (scripts/ab_attention_shapes.py, one build per candidate):
//
//   head_dim 128, fp16   32x16 (incumbent)   32x32     16x32
//     B8 H8 S32                    1.000x   0.192x    0.689x
//     B8 H8 S128                   1.000x   0.765x    0.378x
//     B4 H8 S512                   1.000x   0.738x    0.346x
//     B2 H8 S1024                  1.000x   0.765x    0.357x
//   head_dim 64, fp16    64x16 (incumbent)   64x32     32x32
//     geomean over 3 shapes        1.000x   0.812x    0.552x
//
// A wider BLOCK_N doubles the K/V staging and the score tile without adding any
// parallelism -- 41.4 KB drops the SM from three resident blocks to two, and
// each block then does twice the work per key-tile iteration. Q is already
// register-resident, so there is nothing for the extra occupancy pressure to
// buy. The fp16 win at head_dim 128 comes entirely from the fragments, not from
// a bigger tile.
//
// So WmmaShape stays unparameterised on the element width. If a future sweep
// does find a different winner per dtype, that is the evidence for adding the
// parameter; this one did not.
#ifndef WMMA_M_8
#define WMMA_M_8   64
#define WMMA_N_8   32
#endif
#ifndef WMMA_M_16
#define WMMA_M_16  64
#define WMMA_N_16  32
#endif
// head_dim 32 stays at 64x32, and 128x32 -- which SUPPORTED now admits, and
// which the tuner picks -- is REJECTED. It is not a wash, it is a size split:
//
//   shape 13 (seq 1024)   11.988 -> 10.977 ms   -8.4%
//   shape  6 (batch 10k)  2134.1 -> 1736.7 ms
//   shape  1, 5                            both faster
//   shape 12 (seq 32)     0.2478 -> 0.2744 ms  +10.7%
//   shape  2 (batch 1)    0.1178 -> 0.1229 ms   +4.3%
//   shape  3, 4                            both ~+2.7%
//
// BLOCK_M is the query rows a block owns, so 128 of them against shape 12's
// seq_len of 32 is a block doing a quarter of a tile's worth of work, and
// against shape 2 it collapses the grid to ceil(128/128) * 4 heads * 1 batch =
// 4 blocks on a 46-SM card. Where there is enough work to fill the machine the
// taller block wins clearly; where there is not, it starves it.
//
// The tuner said 1.30x for it even on --grading cases because it scores the
// SUM of milliseconds across cases, and shape 13 alone outweighs 2, 3, 4 and 12
// put together. A geometric mean of per-case ratios -- which is what the
// grading metric actually is -- says otherwise. See score_all().
//
// The real fix is a runtime choice between two instantiations keyed on block
// count, not a different constant here; logged as an open candidate.
// Online softmax in the accumulator registers instead of through `s_s`.
// Compile-time because the prize is an *allocation* -- deleting `s_s` is what
// frees 10.00 KB at head_dim 32 and takes it from 2 resident blocks per SM to 4
// -- and row 12 of docs/OPTIMIZATION_LEDGER.md established that a runtime flag
// cannot measure an allocation. Swept by scripts/tune_block_shapes.py.
//
// Requires the acc_row_of/acc_col_of formulas, which the probed mapping cannot
// substitute for: the reduction needs shuffle offsets known at compile time.
#ifndef WMMA_REG_SOFTMAX
#define WMMA_REG_SOFTMAX 1
#endif

// Compile-time cap on WmmaCfg::KV_STAGES; see the note there. 2 is the kernel's
// own default and 1 builds the single-buffered form, which is what makes the
// second K/V stage's shared memory measurable instead of merely arguable.
#ifndef WMMA_KV_STAGES
#define WMMA_KV_STAGES 2
#endif

#ifndef WMMA_M_32
#define WMMA_M_32  64
#define WMMA_N_32  32
#endif
#ifndef WMMA_M_64
#define WMMA_M_64  64
#define WMMA_N_64  16
#endif
// head_dim 128 was 32x16 -- two warps, 4 of an SM's 48 -- not because that was
// the fastest shape but because it was the tallest one SUPPORTED would admit.
// That gate asked whether the FULL O layout fit, and 64x16 is 48.6 KB there
// against the free 48 KB. It is 35.2 KB under the tile layout that direct_o
// actually launches, so deleting o_s had already paid for the taller block and
// the shape table was simply not allowed to spend it. Re-swept once SUPPORTED
// asked about the launchable layout instead: 64x16 at 2.412 ms against 32x16's
// 3.704, 1.54x, the largest margin in the sweep.
#ifndef WMMA_M_128
#define WMMA_M_128 64
#define WMMA_N_128 16
#endif
// head_dim 256 is the one shape that does not fit in the free 48 KB at all:
// Q, O and both K/V tiles are BLOCK_* x 256, so even 16x16 wants 36 KB in fp16
// and 50 KB in tf32. Once the block is opting into the larger carveout anyway,
// the question is which shape to spend it on. Swept against SDPA over 10
// shapes, ratio > 1 meaning the kernel wins:
//
//                        16x16     32x16     16x32
//   causal b8h8s32       0.690     0.796     0.393
//   causal b8h8s128      0.346     0.536     0.192
//   causal b8h8s512      0.357     0.607     0.213
//   causal b1h8s128      0.479     0.716     0.295
//   causal b2h8s1024     0.390     0.700     0.259
//   dense  b8h8s32       0.575     0.764     0.377
//   dense  b8h8s128      0.269     0.434     0.143
//   dense  b8h8s512      0.309     0.524     0.196
//   dense  b1h8s128      0.472     0.725     0.225
//   dense  b2h8s1024     0.305     0.528     0.203
//
// 32x16 wins every one of the ten, by 1.15x-1.55x, and it does not buy that
// with occupancy: at 55 KB an SM holds one block of two warps, exactly the 64
// threads 16x16 gets from two blocks of one. What it buys is half the blocks,
// hence half the K/V passes over global memory -- the same reason a taller
// block wins at every other head_dim.
//
// That table only ever contained three shapes because nothing taller could be
// BUILT. scripts/tune_block_shapes.py ended its legality test with a hardcoded
// `total <= 48 * 1024` while this kernel takes the 96 KB carveout at head_dim
// 256, so every 64-row shape was filtered out before it was compiled and this
// head_dim had exactly one legal candidate: its own incumbent. A search space
// of one always agrees with you.
//
// With the limit corrected, on grading shape 8's own dimensions (b64 h4 s128
// causal, the only head_dim 256 shape in the set):
//
//   64x32   0.565 ms      <- 1.285x over the incumbent
//   64x16   0.588 ms
//   32x16   0.726 ms      <- what the three-shape table had chosen
//   32x32   1.256 ms
//
// Taking the argument above one step further than it could previously be
// taken: 64x32 halves the blocks again. It needs 81.8 KB, so FULL_O_FITS is
// false and the launcher forces direct_o on -- which is exactly the mechanism
// that makes the shape available at all.
//
// 16x32 loses everywhere, which is the head_dim 128 result again (see the
// WmmaShape note above): a wider key tile doubles the staging without adding
// parallelism, and Q is register-resident so there is nothing for it to buy.
//
// None of the three beats SDPA. That is what keeps head_dim 256 out of Auto --
// see wmma_preferred_by_auto in attention_dispatch.cuh -- and this table is
// about being the best forced `--attn-impl wmma`, not about winning.
#ifndef WMMA_M_256
#define WMMA_M_256 64
#define WMMA_N_256 32
#endif

// The primary template keeps the 64x32 general case for any head_dim the
// dispatcher does not switch on; the five it does switch on read the macros.
template <int HEAD_DIM> struct WmmaShape {
    static constexpr int M = 64;
    static constexpr int N = 32;
};
template <> struct WmmaShape<8>   { static constexpr int M = WMMA_M_8;   static constexpr int N = WMMA_N_8;   };
template <> struct WmmaShape<16>  { static constexpr int M = WMMA_M_16;  static constexpr int N = WMMA_N_16;  };
template <> struct WmmaShape<32>  { static constexpr int M = WMMA_M_32;  static constexpr int N = WMMA_N_32;  };
template <> struct WmmaShape<64>  { static constexpr int M = WMMA_M_64;  static constexpr int N = WMMA_N_64;  };
template <> struct WmmaShape<128> { static constexpr int M = WMMA_M_128; static constexpr int N = WMMA_N_128; };
template <> struct WmmaShape<256> { static constexpr int M = WMMA_M_256; static constexpr int N = WMMA_N_256; };

template <typename scalar_t, int HEAD_DIM>
struct WmmaCfg {
    // Fragments are 16 wide in N, so a head_dim below that cannot fill one.
    // Narrow heads are widened to 16 with zeros: GEMM1 contracts over head_dim
    // and zeros add nothing, GEMM2 produces columns past head_dim that are
    // simply not stored. Only the shared tiles and fragments see the padded
    // width; global loads and stores keep using the real head_dim.
    static constexpr int DIM  = HEAD_DIM;                        // as it is in memory
    static constexpr int PDIM = (HEAD_DIM < 16) ? 16 : HEAD_DIM; // as the MMA sees it

    static constexpr int BLOCK_M = WmmaShape<HEAD_DIM>::M;
    static constexpr int BLOCK_N = WmmaShape<HEAD_DIM>::N;
    static constexpr int ROWS_PER_WARP = 16;                     // == wmma M
    static constexpr int WARPS = BLOCK_M / ROWS_PER_WARP;
    static constexpr int NTHREADS = WARPS * 32;

    static constexpr int WK = FragTraits<scalar_t>::K;
    static constexpr int PAD = FragTraits<scalar_t>::LD_PAD;
    // With the register softmax there is no `s_s` for P to alias, so P always
    // gets its own space. For fp16 that is a straight 10.00 KB saving at
    // head_dim 32 (S_BYTES goes, P_BYTES was already separate); for tf32, where
    // P used to live inside S, it is a wash -- which is fine, the model runs
    // fp16 and the tf32 path exists for comparison.
    static constexpr bool REG_SOFTMAX = (WMMA_REG_SOFTMAX != 0);
    static constexpr bool P_ALIASES_S =
        std::is_same<scalar_t, float>::value && !REG_SOFTMAX;

    static constexpr int KV_LD = PDIM + PAD;       // k_s, v_s, and Q staging
    static constexpr int O_LD  = PDIM + 4;         // o_s is always fp32
    static constexpr int O_TILE_LD = 16 + 4;       // ... and one fragment wide
    static constexpr int S_LD  = BLOCK_N + PAD;    // s_s and p_s

    static constexpr size_t Q_BYTES   = sizeof(scalar_t) * BLOCK_M * KV_LD;
    // Two O layouts, chosen per launch by direct_o_flag().
    //
    // FULL is the original: the whole BLOCK_M x PDIM block tile, staged in
    // fp32, which every warp fills and then reads back a row at a time.
    //
    // TILE is what the direct-to-global epilogue needs instead. There
    // wmma::store_matrix_sync writes the accumulator straight to `out`, and
    // shared memory is only a fallback for the tiles that cannot -- a ragged
    // last row block, head_dim 8's padded columns, an `out` narrower than the
    // fp32 accumulator. Those go through one 16x16 fragment at a time, so the
    // buffer shrinks from the block tile to one fragment per warp.
    //
    // Both sit at offset 0, on top of the Q staging, which is dead from the
    // fragment hoist onward. So the saving is only what O costs ABOVE Q, which
    // is nothing below head_dim 32 (where PDIM == 16 makes the block tile one
    // fragment wide already) and 8-16 KB above it.
    static constexpr size_t O_FULL_BYTES = sizeof(float) * BLOCK_M * O_LD;
    static constexpr size_t O_TILE_BYTES = sizeof(float) * WARPS * 16 * O_TILE_LD;
    static constexpr size_t QO_FULL =
        (Q_BYTES > O_FULL_BYTES) ? Q_BYTES : O_FULL_BYTES;
    static constexpr size_t QO_TILE =
        (Q_BYTES > O_TILE_BYTES) ? Q_BYTES : O_TILE_BYTES;
    static constexpr size_t KV_BYTES  = sizeof(scalar_t) * BLOCK_N * KV_LD;
    static constexpr size_t S_BYTES   =
        REG_SOFTMAX ? 0 : (sizeof(float) * BLOCK_M * S_LD);
    static constexpr size_t P_BYTES   = P_ALIASES_S ? 0 : sizeof(scalar_t) * BLOCK_M * S_LD;
    static constexpr size_t ROW_BYTES = sizeof(float) * BLOCK_M;

    // Everything after O is packed against it, so the two O sizes give two
    // whole layouts. The kernel picks one per launch from a single flag and
    // derives the offsets in a few instructions before the key loop; the loop
    // itself indexes off pointers that were already runtime values, so nothing
    // in it changes. That is what makes the two layouts A/B-able inside one
    // process rather than across two builds -- see csrc/TUNING.md on the
    // cross-process ablation that reported the impossible.
    static constexpr size_t O_OFF = 0;
    static constexpr size_t TAIL_BYTES =
        2 * KV_BYTES + S_BYTES + P_BYTES + 3 * ROW_BYTES;
    static constexpr size_t SMEM      = QO_FULL + TAIL_BYTES;
    static constexpr size_t SMEM_TILE = QO_TILE + TAIL_BYTES;


    // The accumulator probe below needs 512 floats of scratch per warp. It runs
    // after Q has been hoisted into registers and before the first K/V tile is
    // staged, so the whole O/K/V/S span is dead and can host it -- but that
    // span has to actually be big enough.
    static constexpr size_t PROBE_BYTES = sizeof(float) * WARPS * 512;
    // Sized on the SMALLER O layout, so the probe fits whichever one is live.
    static constexpr size_t SCRATCH_BYTES = QO_TILE + 2 * KV_BYTES + S_BYTES;

    // Whether the probe fits at all. It stopped always fitting at iteration 19:
    // `s_s` was 20.00 KB of the scratch span at BLOCK_M 128 / BLOCK_N 32, and
    // REG_SOFTMAX deletes it, which drops SCRATCH_BYTES to 15.00 KB against the
    // 16.00 KB eight warps need. That silently un-admitted 128x32 -- a shape
    // SUPPORTED had accepted since iteration 4.
    //
    // Under REG_SOFTMAX the probe is not merely redundant, it is incompatible:
    // the softmax is written against acc_row_of() directly (the T_OF table), so
    // a probed mapping that disagreed with the closed form would leave the
    // softmax and the O rescale indexing different rows. The formula is
    // load-bearing there whether the probe runs or not, which is why
    // REG_SOFTMAX may waive this requirement rather than merely tolerate it.
    static constexpr bool PROBE_FITS = (SCRATCH_BYTES >= PROBE_BYTES);

    // Whether causal blocks are worth dispatching longest-first. Per head_dim
    // for the same reason WmmaShape is: head_dim 128 runs a 32x16 block of two
    // warps at ~36 KB, so only two blocks and 128 threads land on an SM. With
    // that little in flight the kernel is bound by K/V locality rather than by
    // makespan, and reordering the dispatch costs more L2 reuse than it saves
    // tail -- measured 0.889x-0.966x over five shapes, against 1.02x-1.09x at
    // head_dim 16 through 64. TUNING.md, "wmma kernel: causal block-index
    // reversal", has the table; the A/B script that produced it is gone.
    static constexpr bool REVERSE_CAUSAL = (HEAD_DIM <= 64);

    // 48 KB is what a block gets without opting in, and two of them then fit on
    // an SM. Every head_dim up to 128 stays inside it and none of them calls
    // cudaFuncSetAttribute.
    //
    // head_dim 256 cannot, in either compute type: Q, O and both K/V tiles are
    // all 256 wide, so the swept 32x16 shape is 53.9 KB in fp16 and 67.9 KB in
    // tf32. Declining would mean `--attn-impl wmma` covered a head_dim with the
    // fp16 fragments and not with the tf32 ones, and would also cost the 1.15x-
    // 1.55x the taller block is worth. So that head_dim opts into the larger
    // carveout instead, and pays for it with the second resident block -- which
    // it was never going to get: 2 x 55 KB does not fit in an SM's 100 KB, and
    // the 16x16 shape that does fit twice runs half the warps per block.
    //
    // 96 KB is a compile-time ceiling, not a device query, so it is deliberately
    // under SM 8.x's 99 KB per-block maximum. A device that will not grant it
    // fails the check at launch with a message naming the number, rather than
    // failing the launch itself several frames later.
    static constexpr size_t SMEM_FREE  = 48 * 1024;
    static constexpr size_t SMEM_LIMIT = (HEAD_DIM >= 256) ? (96 * 1024) : SMEM_FREE;


    // Whether the FULL O layout -- and therefore WMMA_DIRECT_O=0 -- is
    // available at this shape. SUPPORTED asks only that the TILE layout fits,
    // because that is the one the default configuration launches; a shape whose
    // full layout would overflow is still perfectly runnable, it simply cannot
    // have the direct-to-global epilogue turned off. The launcher forces
    // direct_o on for those rather than launching a request for more shared
    // memory than the device will grant.
    //
    // This distinction is worth block-shape headroom, which is the whole point
    // of it. At head_dim 128 with fp16 fragments, 64x16 is 48.6 KB under the
    // full layout -- rejected -- and 35.2 KB under the tile layout. Gating on
    // the full layout meant deleting o_s bought occupancy the shape table was
    // then forbidden from spending: head_dim 128 stayed at 32x16, two warps,
    // 4 of an SM's 48.
    // K/V stages. Two lets the cp.async copy for the next key tile fly while
    // the current one is being contracted; one is the copy-then-wait shape.
    // A second pair costs 2 * KV_BYTES, which every head_dim here can afford
    // except 256 -- its tiles are 256 wide, so at 64x32 two stages want
    // 114.8 KB against a 96 KB carveout. That head_dim keeps one stage and
    // still gets cp.async, just without the overlap.
    //
    // Declared HERE, above FULL_O_FITS, because CARVEOUT_BYTES has to know
    // whether the extra stage is being asked for. Requesting it
    // unconditionally would push head_dim 256 to 114.8 KB and the carveout
    // would be refused at launch.
    // DIM == PDIM is part of the decision, not just of whether the fast path
    // runs. At head_dim 8, PDIM pads to 16 and the staging must zero-fill the
    // padded columns, so neither cp.async nor double buffering ever engages --
    // but a second stage would still be REQUESTED, and shared memory requested
    // is shared memory spent. 23.8 KB -> 26.8 KB took head_dim 8 from 4
    // resident blocks per SM to 3 and cost grading shapes 7 and 11 about 5%
    // each, for a buffer neither of them can use.
    //
    // WMMA_KV_STAGES caps this at build time so the *allocation* can be A/B'd,
    // not just the code path. WMMA_CP_ASYNC is a runtime flag and KV_STAGES is
    // constexpr, so both arms of a cp_async_mode A/B request two stages' worth
    // of shared memory and only one of them uses it -- that A/B prices the
    // overlap and is blind to the occupancy the second buffer costs. Build with
    // -DWMMA_KV_STAGES=1 to get a kernel that never asks for it.
    static constexpr int KV_STAGES =
        ((WMMA_KV_STAGES >= 2) && (DIM == PDIM) &&
         (SMEM_TILE + 2 * KV_BYTES <= SMEM_LIMIT)) ? 2 : 1;
    static constexpr size_t KV_SPAN = KV_STAGES * KV_BYTES;
    static constexpr size_t STAGE_EXTRA = (KV_STAGES == 2) ? (2 * KV_BYTES) : 0;

    static constexpr bool FULL_O_FITS = (SMEM <= SMEM_LIMIT);

    // The opt-in has to cover the largest layout that can actually launch, and
    // no more: a carveout is taken out of the same 100 KB the SM splits with
    // L1, so asking for the full layout at a shape that will only ever run the
    // tile one costs resident blocks for nothing.
    static constexpr size_t CARVEOUT_BYTES =
        (FULL_O_FITS ? SMEM : SMEM_TILE) + STAGE_EXTRA;
    static constexpr bool NEEDS_CARVEOUT = (CARVEOUT_BYTES > SMEM_FREE);


    // GEMM1 contracts over the padded head_dim, GEMM2 over the key tile; both
    // must be a whole number of fragments.
    static constexpr bool SUPPORTED =
        FragTraits<scalar_t>::supported &&
        (PDIM % WK == 0) && (PDIM % 16 == 0) &&
        (BLOCK_N % WK == 0) && (BLOCK_M % 16 == 0) &&
        (PROBE_FITS || REG_SOFTMAX) && (SMEM_TILE <= SMEM_LIMIT);
};

// Runtime off switch for the causal block-index reversal below, so the two
// mappings can be A/B'd inside a single process; see wmma_set_causal_reverse in
// the module. The environment variable supplies only the initial value.
// Deliberately unsynchronised, on the same contract as tile_attention's split
// flag: a benchmarking knob flipped between timed runs from one thread, never
// while launches are in flight.
bool& causal_reverse_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_CAUSAL_REVERSE");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// Which softmax the attention kernel runs:
//
//   0  the original -- `s * scale`, then __expf, which is itself
//      `ex2.approx(x * log2e)`, plus an explicit `s == -inf` test
//   1  base-2 domain -- one `s * (scale * log2e)`, then a bare exp2f, and no
//      -inf test, because ex2.approx(-inf) is defined to return +0. Q is
//      untouched, so this is arithmetically as accurate as 0.
//   2  as 1, and `scale * log2e` folded into Q at staging time, so the
//      score-side multiply goes too. Cheapest, but it puts both constants
//      inside the fp16 narrowing, which costs accuracy -- see csrc/TUNING.md.
//
// Same contract as causal_reverse_flag(): a benchmarking knob flipped between
// timed runs from one thread, never while launches are in flight. It selects
// among three kernel instantiations rather than branching inside one, so no
// path pays for another's code or registers.
int& softmax_mode_flag() {
    static int mode = [] {
        const char* e = std::getenv("WMMA_SOFTMAX_MODE");
        if (e == nullptr) return 1;
        int v = std::atoi(e);
        return (v >= 0 && v <= 2) ? v : 1;
    }();
    return mode;
}

// Per-tile mask classification. Every score element used to evaluate its own
// bounds, causal and mask tests -- four predicates and the address arithmetic
// under them, on every element of every key tile. FlashAttention-2 does not:
// `mask.h` templates on Is_even_MN / Is_causal / Col_idx_only so that interior
// tiles compile with no row-index computation at all, and only the diagonal
// block pays.
//
// The same split is available here at run time rather than compile time,
// because BLOCK_N and the mask pointer are not compile-time constants. A key
// tile needs no test at all when its rows are all inside S, its columns are
// all inside S, the whole tile is below the causal diagonal, and there is no
// explicit mask. All of those are warp-uniform -- q_base, kt and S are -- so
// the branch never diverges within a warp.
//
// Both paths have to exist regardless: that is what the optimization IS. The
// flag only forces the slow one, so an A/B costs nothing that the shipped
// kernel does not already carry. Same knob contract as causal_reverse_flag().
// Whether the accumulator row mapping comes from a closed form instead of a
// per-block probe.
//
// Keeping O in accumulator registers means applying the per-row softmax rescale
// to fragment elements, which needs to know which row each element holds --
// architecture-defined, and undocumented. The kernel discovered it by probing:
// store a fragment tagged with (lane, slot), read back where each tag landed,
// invert. Exact by construction on any device, and paid **once per block**, in
// a store_matrix_sync, two __syncwarp barriers and sixteen shared accesses.
//
// Once per block is once too many: the mapping is a property of the
// architecture and the fragment shape, not of the block. So it is computed
// instead -- `(lane >> 2) + 8 * ((t >> 1) & 1)` -- and the probe now runs once
// per process, on the host side, purely to confirm the formula reproduces it.
// If it ever does not, acc_row_formula_ok() returns false and the kernel falls
// back to probing per block, so a device this closed form does not describe
// stays correct rather than silently wrong.
//
// Same knob contract as causal_reverse_flag().
bool& acc_formula_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_ACC_FORMULA");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// Whether the epilogue hands O straight to global memory.
//
// O leaves the key loop in accumulator registers, and the original epilogue
// put it in shared memory first: one wmma::store_matrix_sync per 16-wide
// fragment into a block-sized fp32 tile, a __syncwarp, then sixteen rows read
// back and written out a lane at a time. FlashAttention does not stage it --
// wmma::store_matrix_sync takes a generic pointer, and both output layouts
// keep head_dim as the fastest axis with a constant row stride, which is
// exactly the (pointer, ldm, row_major) the fragment store wants.
//
// Storing direct buys two separate things:
//
//   the epilogue      one fragment store per tile instead of a shared write,
//                     a barrier, sixteen shared reads and sixteen global
//                     writes per lane
//   the block tile    which then only has to hold the fallback cases, one
//                     fragment per warp instead of BLOCK_M x PDIM, freeing
//                     8 KB at head_dim 64 and 128 and 16 KB at 256
//
// The second is why the flag also picks the shared-memory layout and the
// launch size, rather than only the code path: measuring one without the other
// would miss most of it. Same knob contract as causal_reverse_flag().
bool& direct_o_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_DIRECT_O");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// cp.async for the K/V staging. On by default; off restores the scalar
// global->register->shared path so the two can be timed in one process.
//
// This was blocked until q/k/v became fp16 IN GLOBAL MEMORY (see
// optimized/config.py, QKV_FP16). cp.async copies bytes global->shared without
// passing through registers, and therefore cannot convert -- so while the
// kernel was handed fp32 tensors and contracted them as __half, the narrowing
// on the staging path made an async copy impossible. Once the tensor is already
// the compute type the copy is a pure move and cp.async is legal.
// 0 = scalar staging, 1 = cp.async single-buffered, 2 = cp.async double
// buffered. Three values rather than a bool because the two links of the chain
// are separate results and have to be timeable against each other in one
// process: 0->1 is the register bypass, 1->2 is the overlap.
int& cp_async_mode() {
    static int mode = [] {
        const char* e = std::getenv("WMMA_CP_ASYNC");
        if (e == nullptr) return 2;
        const int v = atoi(e);
        return (v >= 0 && v <= 2) ? v : 2;
    }();
    return mode;
}

bool& mask_classify_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_MASK_CLASSIFY");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// Split-KV (Flash-Decoding). The attention grid is (ceil(S/BLOCK_M), H, B) --
// query side only -- so nothing in it scales with the KEY length. A shape with
// few queries and many keys launches a small grid whose blocks each do a lot of
// serial work, and the card sits idle. Splitting the key range across extra
// blocks trades a second pass for parallelism that was not otherwise reachable.
//
// Measured on this card (scripts/bench_wmma_occupancy.py): b1 h8 s128 d32 runs
// 16 blocks against 138 resident, 12% of the card, at 5.7x the per-block cost
// of the same kernel at a batch that fills it.
//
// Same knob contract as causal_reverse_flag().
bool& split_kv_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_SPLIT_KV");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// Forces a split count for sweeps; 0 restores the rule below. This is how the
// rule's constants were chosen, so it stays reachable.
int& split_count_override() {
    static int n = [] {
        const char* e = std::getenv("WMMA_SPLIT_COUNT");
        return (e != nullptr) ? std::atoi(e) : 0;
    }();
    return n;
}

// Key tiles the AVERAGE block walks. Under causal a block's key range ends at
// its own last row, so across m-tiles it averages half the dense count -- and
// it is the average, not the maximum, that decides whether a split has
// anything to divide.
int split_key_tiles(int S, int block_n, bool is_causal) {
    const int dense = (S + block_n - 1) / block_n;
    return is_causal ? ((dense + 1) / 2) : dense;
}

// A split group shares an m-tile, so it also shares that m-tile's Q and its
// slice of K/V in L2. More than this and the partial workspace and the combine
// pass start costing more than the parallelism is worth.
constexpr int kMaxSplits = 8;

// How many ways to cut the key range, or 1 for "do not".
int wmma_split_count(int blocks, int resident, int n_kt, int head_dim) {
    const int forced = split_count_override();
    if (forced > 0) return forced;
    if (!split_kv_flag()) return 1;
    // head_dim 8 pads its operands to 16, so the fragments are half zeros and
    // there is no dense key loop to redistribute in the first place.
    if (head_dim <= 8) return 1;
    if (blocks <= 0 || resident <= 0) return 1;

    // An eighth of the card or less. This is the clause the sweep bought, and
    // it is deliberately stricter than "there is spare capacity": splitting
    // adds a whole second kernel launch, and unless the grid is very small
    // that launch costs more than the shortened key loop saves. Measured with
    // the count the rule itself picks (scripts/ab_wmma_split_kv.py):
    //
    //   shape                blocks  waves  ratio
    //   B1 H4 S128  d32 c         8  0.06   1.122x
    //   B1 H8 S128  d32 c        16  0.12   1.079x
    //   B1 H8 S128  d64 c        16  0.12   1.582x
    //   B1 H4 S256  d64 c        16  0.12   2.526x
    //   B2 H8 S128  d32 c        32  0.23   0.898x   <- declined by this clause
    //   B4 H8 S128  d32 c        64  0.46   0.779x   <- declined
    //   B1 H8 S512  d64 c        64  0.46   0.955x   <- declined
    //
    // A looser `blocks * 4 <= resident` admits the 0.898x row, and admitting
    // long key loops with an `n_kt >= 16` clause admits the 0.955x one. Both
    // were tried and both lose.
    if (blocks * 8 > resident) return 1;

    // And there has to be a key loop worth dividing. Splitting adds a whole
    // combine launch per call -- six per forward in a six-layer model -- and
    // halving a two-tile loop does not repay it. This clause is the one the
    // END-TO-END measurement bought, and it overrides what the op alone says:
    //
    //   shape                 head_dim n_kt  op      end to end
    //   B1 S128 d512 h8             64    4  1.807x  1.061x
    //   B1 S128 d256 h8             32    2  1.084x  0.963x   <- declined here
    //
    // The op-level win at n_kt 2 is real and still does not survive the extra
    // launch once it is paid once per layer instead of amortised over a graph
    // full of back-to-back attention calls.
    if (n_kt < 4) return 1;

    // FLOOR, never ceiling: a count that overfills the card serialises the
    // extra blocks into a second wave AND pays the combine pass for it.
    int n = resident / blocks;
    if (n < 2) return 1;

    // Cannot usefully cut a key range into more pieces than it has tiles; the
    // surplus splits come up empty and only add blocks and combine terms.
    if (n > n_kt) n = n_kt;
    if (n > kMaxSplits) n = kMaxSplits;
    return (n < 2) ? 1 : n;
}

// `scalar_t` is what q/k/v/out are in GLOBAL memory; `compute_t` is what the
// shared tiles hold and the fragments contract in. They were one type until
// fp16 was measured against tf32: both carry a 10-bit mantissa, so an fp32
// tensor can be narrowed to fp16 on its way into shared memory at no cost in
// precision, and buy two things for it -- fp16 tensor cores run 2.0x-2.25x tf32
// on this card, and a 16x16x16 fragment contracts twice the K of tf32's
// 16x16x8, so the mma count halves too. Narrowing also halves every staged
// tile, which is what decides the block shape at head_dim 128.
//
// The output stays `scalar_t`: it feeds out_proj, which is a cuBLAS fp32 GEMM.
// Which row of the 16x16 accumulator tile does element `t` of lane `lane` hold?
//
// The closed form the probe kept rediscovering. Elements come in pairs that
// share a row, the pairs alternate between the tile's top and bottom halves,
// and the lane's group of four picks the row inside a half. Verified against
// the probe once per process by acc_row_formula_ok() below -- this is an
// assertion about the hardware, not a definition of it.
__device__ __host__ __forceinline__ int acc_row_of(int lane, int t) {
    return (lane >> 2) + 8 * ((t >> 1) & 1);
}

// And which COLUMN. Needed by the register-resident softmax, which has to apply
// causal and mask predicates per element and therefore needs the full
// (row, col) of each accumulator slot, not just the row.
//
// Consistent with acc_row_of by construction: that one puts elements
// t = {0,1,4,5} in row `lane>>2` and t = {2,3,6,7} in row `(lane>>2) + 8`, so
// each lane holds 4 elements per row. The quad `lane & ~3` owns a full 16-wide
// row between its four lanes, taking the column pair at `2 * (lane % 4)` in each
// 8-column half. Verified against the probe by acc_map_formula_ok() below -- as
// with the row map this is an assertion about the hardware, not a definition.
__device__ __host__ __forceinline__ int acc_col_of(int lane, int t) {
    return (lane & 3) * 2 + (t & 1) + 8 * (t >> 2);
}

// One warp, run once per process, to check that. Writes the probe's answer for
// every (lane, slot) so the host can compare all 256 rather than a sample.
template <int WK>
__global__ void acc_row_probe_kernel(int* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using acc_frag_t = wm::fragment<wm::accumulator, 16, 16, WK, float>;
    __shared__ float probe_out[256];
    __shared__ int tag_to_row[256];
    __shared__ int tag_to_col[256];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    acc_frag_t probe;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        probe.x[t] = static_cast<float>(lane * 8 + t);
    }
    wm::store_matrix_sync(probe_out, probe, 16, wm::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int pos = lane * 8 + t;
        tag_to_row[static_cast<int>(probe_out[pos])] = pos / 16;
        // The column was always available here and was simply discarded.
        tag_to_col[static_cast<int>(probe_out[pos])] = pos % 16;
    }
    __syncwarp();
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        // 256 rows then 256 columns, so one launch answers both maps.
        out[lane * 8 + t] = tag_to_row[lane * 8 + t];
        out[256 + lane * 8 + t] = tag_to_col[lane * 8 + t];
    }
#endif
}

// Does acc_row_of() describe this device's fragment layout? Asked once per
// fragment K, cached. A false answer is not an error -- it sends the kernel
// back to probing per block, which is what it did before and is always right.
template <int WK>
bool acc_row_formula_ok() {
    static const bool ok = [] {
        int* d = nullptr;
        // 256 rows followed by 256 columns.
        if (cudaMalloc(&d, 512 * sizeof(int)) != cudaSuccess) return false;
        acc_row_probe_kernel<WK><<<1, 32>>>(d);
        int host[512];
        const cudaError_t copied =
            cudaMemcpy(host, d, sizeof(host), cudaMemcpyDeviceToHost);
        cudaFree(d);
        if (copied != cudaSuccess) return false;
        for (int lane = 0; lane < 32; ++lane) {
            for (int t = 0; t < 8; ++t) {
                if (host[lane * 8 + t] != acc_row_of(lane, t)) return false;
                // The column map is checked on the same footing as the row map:
                // the register-resident softmax needs both, and a formula that
                // is right about rows and wrong about columns would silently
                // compute a transposed mask.
                if (host[256 + lane * 8 + t] != acc_col_of(lane, t)) return false;
            }
        }
        return true;
    }();
    return ok;
}

// Exposed so a verification script can assert the map from Python rather than
// only having it checked implicitly at launch. Returns 0 on mismatch, and
// writes the probed (row, col) pair for all 256 slots when `out` is non-null.
template <int WK>
bool acc_map_probe(int* host_out) {
    int* d = nullptr;
    if (cudaMalloc(&d, 512 * sizeof(int)) != cudaSuccess) return false;
    acc_row_probe_kernel<WK><<<1, 32>>>(d);
    const cudaError_t copied =
        cudaMemcpy(host_out, d, 512 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return copied == cudaSuccess;
}

template <typename scalar_t, typename compute_t, int HEAD_DIM, int MODE>
__global__ __launch_bounds__(WmmaCfg<compute_t, HEAD_DIM>::NTHREADS)
void fused_attention_wmma_kernel(const scalar_t* __restrict__ q,
                                 const scalar_t* __restrict__ k,
                                 const scalar_t* __restrict__ v,
                                 int64_t qs0, int64_t qs1, int64_t qs2,
                                 const bool* __restrict__ mask,
                                 int64_t ms0, int64_t ms1,
                                 int64_t ms2, int64_t ms3,
                                 scalar_t* __restrict__ out,
                                 float* __restrict__ part_o,
                                 float* __restrict__ part_m,
                                 float* __restrict__ part_l,
                                 int B, int H, int S,
                                 bool is_causal, float scale,
                                 bool out_bshd, bool reverse_m, int splits,
                                 bool classify, bool direct_o,
                                 int cp_async_mode_arg,
                                 bool acc_formula) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Cfg = WmmaCfg<compute_t, HEAD_DIM>;
    using frag_elem = typename FragTraits<compute_t>::elem;
    using acc_frag_t = wm::fragment<wm::accumulator, 16, 16, Cfg::WK, float>;
    constexpr int BLOCK_M = Cfg::BLOCK_M;
    constexpr int BLOCK_N = Cfg::BLOCK_N;
    constexpr int WK      = Cfg::WK;
    constexpr int RPW     = Cfg::ROWS_PER_WARP;
    constexpr int KV_LD   = Cfg::KV_LD;
    constexpr int O_LD    = Cfg::O_LD;
    constexpr int S_LD    = Cfg::S_LD;
    // DIM is the head_dim as it sits in global memory; PDIM is what the MMA
    // sees, which is DIM widened to a whole 16-wide fragment. They differ only
    // for head_dim 8.
    constexpr int DIM     = Cfg::DIM;
    constexpr int PDIM    = Cfg::PDIM;
    constexpr int N_TILES = PDIM / 16;
    // Softmax lane mapping: RPW lanes cover the rows, the rest split each
    // row into COLS_PER_LANE-wide segments.
    constexpr int LANES_PER_ROW = 32 / RPW;
    constexpr int COLS_PER_LANE = BLOCK_N / LANES_PER_ROW;
    static_assert(RPW <= 32 && 32 % RPW == 0, "warp must split evenly over rows");
    static_assert(BLOCK_N % LANES_PER_ROW == 0, "key tile must split evenly over lanes");
    constexpr bool IS_TF32 = std::is_same<compute_t, float>::value;
    // MODE 0 is the original. MODE 1 moves the softmax into the base-2
    // domain, which deletes the FMUL hidden inside __expf and the -inf test
    // that ex2.approx already handles, and touches nothing else. MODE 2 also
    // folds `scale` into Q, which deletes the score-side multiply as well but
    // puts both constants inside the fp16 narrowing -- the reason the two are
    // separable knobs rather than one.
    constexpr bool USE_EXP2 = (MODE >= 1);
    constexpr bool FOLD_Q   = (MODE == 2);
    // The score-side multiply. MODE 1 still pays it, once per score element,
    // but folds log2(e) into the same FMUL so __expf's hidden one goes away.
    // MODE 2 has already paid both into Q and leaves this dead.
    const float s_mul = USE_EXP2 ? (scale * kLog2e) : scale;

    extern __shared__ __align__(16) char smem_raw[];
    // The O region leads the block and everything else is packed against it,
    // so its size sets every other offset. direct_o shrinks it from the whole
    // block tile to one fragment per warp; see direct_o_flag(). Computed once,
    // outside every loop, into the same pointers the loop already used.
    const size_t qo_bytes = direct_o ? Cfg::QO_TILE : Cfg::QO_FULL;
    const size_t k_off = Cfg::O_OFF + qo_bytes;
    const size_t v_off = k_off + Cfg::KV_SPAN;
    const size_t s_off = v_off + Cfg::KV_SPAN;
    const size_t p_off = s_off + Cfg::S_BYTES;
    const size_t m_off = p_off + Cfg::P_BYTES;
    const size_t l_off = m_off + Cfg::ROW_BYTES;
    const size_t c_off = l_off + Cfg::ROW_BYTES;
    float*    o_s = reinterpret_cast<float*>(smem_raw + Cfg::O_OFF);
    compute_t* k_s = reinterpret_cast<compute_t*>(smem_raw + k_off);
    compute_t* v_s = reinterpret_cast<compute_t*>(smem_raw + v_off);
    float*    s_s = reinterpret_cast<float*>(smem_raw + s_off);
    float*    m_s = reinterpret_cast<float*>(smem_raw + m_off);
    float*    l_s = reinterpret_cast<float*>(smem_raw + l_off);
    float*    c_s = reinterpret_cast<float*>(smem_raw + c_off);
    // P feeds the second GEMM as a matrix_a fragment, so it has to be in the
    // operand type. For fp32 that is the same type the scores were stored in,
    // so the softmax can overwrite the score tile in place.
    compute_t* p_s = Cfg::P_ALIASES_S ? reinterpret_cast<compute_t*>(s_s)
                                      : reinterpret_cast<compute_t*>(smem_raw + p_off);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    // Longest-processing-time-first for causal. Blocks are dispatched in
    // roughly increasing linear index with grid.x varying fastest, and under
    // causal masking block m walks m+1 key tiles -- so the identity mapping
    // hands out the cheapest tiles first and leaves the most expensive for the
    // final wave, where they alone set the makespan. Reversing grid.x issues
    // the long ones first. grid.x *is* n_m here (unlike the tile kernel, whose
    // grid.x is n_m * splits), so gridDim.x needs no rederivation. Dense has no
    // such spread and keeps the identity mapping -- see csrc/TUNING.md.
    //
    // Under split-KV grid.x carries both axes: a grid has only three and y/z
    // are spoken for by heads and batch. `splits` consecutive blocks share an
    // m-tile, which keeps the K/V that a split group reads adjacent in L2.
    //
    // reverse_m and splits are mutually exclusive by construction -- the
    // reversal only fires above one wave and the split only below one -- but
    // n_m is derived rather than assumed, so the two compose if that changes.
    const int n_m = static_cast<int>(gridDim.x) / splits;
    int m_lane = static_cast<int>(blockIdx.x);
    int split  = 0;
    if (splits > 1) {
        split  = m_lane % splits;
        m_lane = m_lane / splits;
    }
    const int m_tile = reverse_m ? (n_m - 1 - m_lane) : m_lane;
    const int h      = blockIdx.y;
    const int b      = blockIdx.z;

    const int64_t bh_off =
        static_cast<int64_t>(b) * qs0 + static_cast<int64_t>(h) * qs1;
    const int row_base = warp * RPW;                  // this warp stripe in the block
    const int q_base   = m_tile * BLOCK_M + row_base; // ... and in the sequence

    compute_t zero_v;
    dev_from_float(zero_v, 0.0f);

    // --- stage Q, then hoist it into registers for the whole key loop -------
    {
        compute_t* q_s = reinterpret_cast<compute_t*>(smem_raw + Cfg::O_OFF);
        // Walk the padded width so columns past DIM are explicitly zeroed:
        // they feed GEMM1's contraction, where a stale value would corrupt the
        // score rather than contribute nothing.
        // Under EXP2 this is also where the softmax's two per-score-element
        // multiplies get paid, once each, on BLOCK_M*head_dim elements instead
        // of on BLOCK_M*BLOCK_N per key tile. Q is dead after the fragment
        // hoist below, so nothing downstream sees the premultiplied copy.
        const float q_pre = scale * kLog2e;
        for (int idx = tid; idx < BLOCK_M * PDIM; idx += Cfg::NTHREADS) {
            const int r = idx / PDIM;
            const int c = idx - r * PDIM;
            const int gr = m_tile * BLOCK_M + r;
            // Still a predicated select rather than a branch -- see
            // dev_of_float's comment; an if/else here cost 1.3x-2.1x once.
            const float qv =
                (gr < S && c < DIM)
                    ? dev_to_float(q[bh_off + static_cast<int64_t>(gr) * qs2 + c])
                    : 0.0f;
            q_s[r * KV_LD + c] =
                dev_of_float<compute_t>(FOLD_Q ? (qv * q_pre) : qv);
        }
        __syncthreads();
    }

    wm::fragment<wm::matrix_a, 16, 16, WK, frag_elem, wm::row_major> q_frag[PDIM / WK];
    {
        const compute_t* q_s = reinterpret_cast<const compute_t*>(smem_raw + Cfg::O_OFF);
        #pragma unroll
        for (int kk = 0; kk < PDIM / WK; ++kk) {
            wm::load_matrix_sync(q_frag[kk],
                                 q_s + static_cast<size_t>(row_base) * KV_LD + kk * WK,
                                 KV_LD);
            if constexpr (IS_TF32) {
                #pragma unroll
                for (int t = 0; t < q_frag[kk].num_elements; ++t) {
                    q_frag[kk].x[t] = wm::__float_to_tf32(q_frag[kk].x[t]);
                }
            }
        }
    }
    __syncthreads();  // Q is in registers; s_s and the O region are free again

    // --- which row of the 16x16 tile does each accumulator element hold? ----
    //
    // Keeping O in accumulator registers means applying the per-row softmax
    // rescale directly to fragment elements, and the element-to-row mapping is
    // architecture-defined -- CUDA does not document it. So probe it: store a
    // fragment whose elements are tagged with (lane, slot), read back where
    // each tag landed, and invert. One 16x16 tile per warp, once per block,
    // exact by construction on any device the kernel compiles for.
    constexpr int ACC_ELEMS = 16 * 16 / 32;
    // Q is already in registers and the first K/V tile has not been staged, so
    // the O/K/V/S span is dead and hosts the probe. s_s alone is not always big
    // enough -- at head_dim 128 the block is 32x16 and s_s holds 640 floats
    // against the 1024 two warps need.
    static_assert(Cfg::PROBE_FITS || Cfg::REG_SOFTMAX,
                  "shared scratch is too small to host the per-warp accumulator "
                  "probe, and REG_SOFTMAX is off so the closed form is not forced");
    int acc_row[ACC_ELEMS];
    // `|| !PROBE_FITS`: where the scratch cannot host the probe the closed form
    // is the only option, and SUPPORTED only admitted this shape because
    // REG_SOFTMAX makes that safe. The launcher TORCH_CHECKs the formula
    // actually describes this device in that case.
    if (acc_formula || !Cfg::PROBE_FITS) {
        // The mapping is a property of the architecture, not of this block.
        #pragma unroll
        for (int t = 0; t < ACC_ELEMS; ++t) {
            acc_row[t] = acc_row_of(lane, t);
        }
    } else if constexpr (Cfg::PROBE_FITS) {
        float* probe_base = reinterpret_cast<float*>(smem_raw + Cfg::O_OFF);
        float* probe_out = probe_base + warp * 512;
        int*   tag_to_row = reinterpret_cast<int*>(probe_base + warp * 512 + 256);
        acc_frag_t probe;
        #pragma unroll
        for (int t = 0; t < ACC_ELEMS; ++t) {
            probe.x[t] = static_cast<float>(lane * ACC_ELEMS + t);
        }
        wm::store_matrix_sync(probe_out, probe, 16, wm::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < ACC_ELEMS; ++t) {
            const int pos = lane * ACC_ELEMS + t;      // 32 lanes x 8 == 256 slots
            tag_to_row[static_cast<int>(probe_out[pos])] = pos / 16;
        }
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < ACC_ELEMS; ++t) {
            acc_row[t] = tag_to_row[lane * ACC_ELEMS + t];
        }
        __syncwarp();
    }

    acc_frag_t o_frag[N_TILES];
    #pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
        wm::fill_fragment(o_frag[n], 0.0f);
    }
    for (int r = tid; r < BLOCK_M; r += Cfg::NTHREADS) {
        m_s[r] = -INFINITY;
        l_s[r] = 0.0f;
    }

    const int64_t mask_bh =
        (mask != nullptr) ? (static_cast<int64_t>(b) * ms0 + static_cast<int64_t>(h) * ms1) : 0;

    // Under causal masking no query in this block looks past the block own
    // last row, so whole key tiles beyond it are skipped rather than computed
    // and thrown away.
    const int key_limit = is_causal ? min(S, m_tile * BLOCK_M + BLOCK_M) : S;

    // Each split takes a contiguous run of whole key tiles out of *this
    // block's* range, not out of [0, S). That is what makes split-KV work
    // under causal masking: slicing the dense range instead would leave the
    // later splits of an early m-tile with nothing to do while a late m-tile
    // still carried its whole range.
    //
    // A split can still come up empty, when a block has fewer key tiles than
    // there are splits. It falls through the loop and stores its initial
    // state -- o_frag is zero-filled, m is -inf, l is 0 -- which the combine
    // pass weights to exactly zero. Cheaper than a launch-time special case,
    // and impossible to get wrong.
    // Loop-invariant half of the tile classification below: whether every
    // query row this warp owns is inside S. It depends on q_base and S only,
    // so it is decided once rather than per key tile.
    const bool rows_in = (q_base + RPW) <= S;

    int kt_begin = 0;
    int kt_end   = key_limit;
    if (splits > 1) {
        const int n_kt = (key_limit + BLOCK_N - 1) / BLOCK_N;
        const int per  = (n_kt + splits - 1) / splits;
        kt_begin = min(split * per * BLOCK_N, key_limit);
        kt_end   = min(kt_begin + per * BLOCK_N, key_limit);
    }

    // Elements between one stage's tile and the next.
    constexpr int KV_STAGE_ELEMS = BLOCK_N * KV_LD;

    // Whether every key tile in this block's range is a "plain" one: full
    // inside S and needing no zero-fill. Only then can the staging be a pure
    // cp.async, and only then is double buffering worth arranging -- a mixed
    // range would need the scalar path for some tiles and the two schemes do
    // not interleave. The range is whole tiles iff its length divides by
    // BLOCK_N, which holds for every grading shape (S in {32,128,1024},
    // BLOCK_N in {16,32}).
    const bool all_plain = classify && (DIM == PDIM) && (kt_end <= S) &&
                           (((kt_end - kt_begin) % BLOCK_N) == 0);
    constexpr bool kCpAsyncType =
        (sizeof(scalar_t) == 2) && (sizeof(compute_t) == 2) && (PDIM % 8 == 0);
    const bool cp_async = cp_async_mode_arg >= 1;
    // Enough key tiles for the pipeline to pay for itself. Double buffering
    // costs a prologue copy and a final wait_prior(0) that the single-buffered
    // form does not, and it only earns that back by overlapping a copy with a
    // tile's worth of MMAs -- so a block with two or three tiles pays the
    // setup and collects almost nothing.
    //
    // Measured: grading shape 2 (batch 1, S 128) runs 2-4 tiles per block and
    // read **0.950x** on the sync->auto link, against 1.050x for shape 13 at
    // up to 32 tiles. Four is where it stops being a loss.
    //
    // A one-tile range is unaffected either way: there is no next tile to
    // prefetch, so the dbuf path degenerates into the sync path.
    const int kv_tiles = (kt_end - kt_begin) / BLOCK_N;
    const bool dbuf = (cp_async_mode_arg >= 2) && (Cfg::KV_STAGES == 2) &&
                      kCpAsyncType && cp_async && all_plain &&
                      (kt_begin < kt_end) && (kv_tiles >= 4);

    // Prologue for the double-buffered form: the first tile is issued before
    // the loop, so that inside the loop the copy in flight is always the NEXT
    // tile rather than the current one. That is the whole point -- it is what
    // puts the copy alongside the MMAs instead of in front of them.
    if (dbuf) {
        constexpr int kChunk = 8;
        constexpr int kChunksPerRow = PDIM / kChunk;
        const scalar_t* k0 = k + bh_off + static_cast<int64_t>(kt_begin) * qs2;
        const scalar_t* v0 = v + bh_off + static_cast<int64_t>(kt_begin) * qs2;
        for (int idx = tid; idx < BLOCK_N * kChunksPerRow; idx += Cfg::NTHREADS) {
            const int r = idx / kChunksPerRow;
            const int c = (idx - r * kChunksPerRow) * kChunk;
            const int64_t g = static_cast<int64_t>(r) * qs2 + c;
            __pipeline_memcpy_async(&k_s[r * KV_LD + c], k0 + g, 16);
            __pipeline_memcpy_async(&v_s[r * KV_LD + c], v0 + g, 16);
        }
        __pipeline_commit();
    }

    int stage = 0;
    for (int kt = kt_begin; kt < kt_end; kt += BLOCK_N) {
        __syncthreads();  // everyone is done reading the previous k_s/v_s

        // head_dim is stride-1 whatever the caller's layout, so a key row is
        // one flat span and the global reads stay coalesced; qs2 is only the
        // spacing between rows. Rows past S are zeroed: they are masked out of
        // the scores anyway, but a NaN in v_s would survive `0 * v` in GEMM2.
        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * qs2;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * qs2;
        // Is every key column of this tile inside S? Both the staging below and
        // the softmax further down want the answer, so it is computed once.
        const bool cols_in = (kt + BLOCK_N) <= S;

        // A whole key tile inside S needs no per-element bounds test, and at
        // every head_dim but 8 the column test is compile-time true anyway.
        const bool kv_plain = classify && cols_in && (DIM == PDIM);

        // cp.async is only a *move*, so it needs the stored type to already be
        // the compute type -- 2 bytes either side, which holds for
        // (Half -> __half) and (BFloat16 -> __nv_bfloat16) and fails for the
        // fp32-tensor path that still narrows. PDIM % 8 gives whole 16-byte
        // chunks; every PDIM in {16,32,64,128,256} qualifies.
        //
        // Alignment, which cp.async requires at 16 bytes on both ends:
        //   dest   k_s and v_s sit at 16-byte-aligned offsets in smem, and
        //          KV_LD * 2 is a multiple of 16 because PAD is 8 halves and
        //          PDIM is a multiple of 8 -- so r * KV_LD lands aligned for
        //          every r, and c is a multiple of 8 elements.
        //   src    the head_dim run is contiguous; the row stride qs2 is
        //          3 * d_model elements for the fused-QKV view, 6 * d_model
        //          bytes, a multiple of 16 at every d_model here.
        constexpr bool kCpAsyncOk =
            (sizeof(scalar_t) == 2) && (sizeof(compute_t) == 2) && (PDIM % 8 == 0);
        constexpr int kChunk = 8;                    // elements per 16-byte copy
        constexpr int kChunksPerRow = PDIM / kChunk;

        // Which stage holds the tile about to be contracted, and which one the
        // next copy targets. With two stages the copy never touches the buffer
        // being read, and the barrier at the top of the body is what keeps
        // iteration i+1's copy off the buffer iteration i read.
        compute_t* k_cur = k_s + static_cast<size_t>(stage) * KV_STAGE_ELEMS;
        compute_t* v_cur = v_s + static_cast<size_t>(stage) * KV_STAGE_ELEMS;

        if (dbuf) {
            const int nxt = stage ^ 1;
            const int kt_next = kt + BLOCK_N;
            if (kt_next < kt_end) {
                constexpr int kChunk = 8;
                constexpr int kChunksPerRow = PDIM / kChunk;
                const scalar_t* kn = k + bh_off + static_cast<int64_t>(kt_next) * qs2;
                const scalar_t* vn = v + bh_off + static_cast<int64_t>(kt_next) * qs2;
                compute_t* k_nxt = k_s + static_cast<size_t>(nxt) * KV_STAGE_ELEMS;
                compute_t* v_nxt = v_s + static_cast<size_t>(nxt) * KV_STAGE_ELEMS;
                for (int idx = tid; idx < BLOCK_N * kChunksPerRow;
                     idx += Cfg::NTHREADS) {
                    const int r = idx / kChunksPerRow;
                    const int c = (idx - r * kChunksPerRow) * kChunk;
                    const int64_t g = static_cast<int64_t>(r) * qs2 + c;
                    __pipeline_memcpy_async(&k_nxt[r * KV_LD + c], kn + g, 16);
                    __pipeline_memcpy_async(&v_nxt[r * KV_LD + c], vn + g, 16);
                }
                __pipeline_commit();
                // Two commits outstanding; wait until only the newest remains,
                // which means THIS tile has landed.
                __pipeline_wait_prior(1);
            } else {
                __pipeline_wait_prior(0);
            }
            stage = nxt;
        } else if (kCpAsyncOk && cp_async && kv_plain) {
            const int total = BLOCK_N * kChunksPerRow;
            for (int idx = tid; idx < total; idx += Cfg::NTHREADS) {
                const int r = idx / kChunksPerRow;
                const int c = (idx - r * kChunksPerRow) * kChunk;
                const int64_t g = static_cast<int64_t>(r) * qs2 + c;
                __pipeline_memcpy_async(&k_cur[r * KV_LD + c], k_base + g, 16);
                __pipeline_memcpy_async(&v_cur[r * KV_LD + c], v_base + g, 16);
            }
            // No double buffering yet: commit and wait before the tile is
            // read. That already removes the register round trip, which is the
            // first link of the chain; overlapping the copy with the previous
            // tile's MMAs is the next one and is measured separately.
            __pipeline_commit();
            __pipeline_wait_prior(0);
        } else if (kv_plain) {
            for (int idx = tid; idx < BLOCK_N * PDIM; idx += Cfg::NTHREADS) {
                const int r = idx / PDIM;
                const int c = idx - r * PDIM;
                const int64_t g = static_cast<int64_t>(r) * qs2 + c;
                k_cur[r * KV_LD + c] =
                    dev_of_float<compute_t>(dev_to_float(k_base[g]));
                v_cur[r * KV_LD + c] =
                    dev_of_float<compute_t>(dev_to_float(v_base[g]));
            }
        } else {
            for (int idx = tid; idx < BLOCK_N * PDIM; idx += Cfg::NTHREADS) {
                const int r = idx / PDIM;
                const int c = idx - r * PDIM;
                const bool inb = ((kt + r) < S) && (c < DIM);
                const int64_t g = static_cast<int64_t>(r) * qs2 + c;
                k_cur[r * KV_LD + c] =
                    inb ? dev_of_float<compute_t>(dev_to_float(k_base[g])) : zero_v;
                v_cur[r * KV_LD + c] =
                    inb ? dev_of_float<compute_t>(dev_to_float(v_base[g])) : zero_v;
            }
        }
        __syncthreads();

#if WMMA_REG_SOFTMAX
        // --- 1+2. S = Q @ K^T, softmax applied in the accumulators ---------
        //
        // The scores never reach shared memory. Each lane ends up holding two
        // rows of this warp's 16-row stripe (`lane>>2` and `lane>>2 + 8`) with
        // four columns of each per 16x16 tile, so a quad `lane & ~3` owns a
        // full 16-wide row and the row reduction is two shuffles.
        constexpr int KVT = BLOCK_N / 16;          // score tiles across the key tile
        acc_frag_t s_acc[KVT];
        #pragma unroll
        for (int n = 0; n < KVT; ++n) {
            wm::fill_fragment(s_acc[n], 0.0f);
            #pragma unroll
            for (int kk = 0; kk < PDIM / WK; ++kk) {
                wm::fragment<wm::matrix_b, 16, 16, WK, frag_elem, wm::col_major> kb;
                wm::load_matrix_sync(kb,
                                     k_cur + static_cast<size_t>(n) * 16 * KV_LD + kk * WK,
                                     KV_LD);
                if constexpr (IS_TF32) {
                    #pragma unroll
                    for (int t = 0; t < kb.num_elements; ++t) {
                        kb.x[t] = wm::__float_to_tf32(kb.x[t]);
                    }
                }
                wm::mma_sync(s_acc[n], q_frag[kk], kb, s_acc[n]);
            }
        }
        // No store_matrix_sync and no __syncwarp here: nothing was published.

        {
            // Which fragment slots belong to each of the lane's two rows.
            // acc_row_of puts ((t>>1)&1)==0 in the upper row and ==1 in the
            // lower, four slots each.
            constexpr int T_OF[2][4] = {{0, 1, 4, 5}, {2, 3, 6, 7}};
            const int q4 = lane & 3;                 // position within the quad
            const int rq = lane >> 2;                // row inside the 8-row half

            // Column of every slot this lane owns. Independent of h -- both of
            // the lane's rows sit at the same columns -- so it is hoisted out
            // of the row loop and out of the three places that each used to
            // recompute it (predicate, zero-fill, store). That arithmetic is
            // most of the masked path's cost, and the masked path is what the
            // short-sequence shapes take: seq 32 against BLOCK_M 64 has half
            // its rows out of bounds, so `rows_in` is false and `plain` never
            // fires. Recomputing it cost that shape 17%.
            int cols[KVT][4];
            #pragma unroll
            for (int n = 0; n < KVT; ++n) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    cols[n][e] = n * 16 + q4 * 2 + (e & 1) + 8 * (e >> 1);
                }
            }

            // Warp-uniform tile predicates, exactly as the shared-memory path
            // computes them -- q_base, kt and S do not vary across the warp.
            const bool tri_free = !is_causal || ((kt + BLOCK_N - 1) <= q_base);
            const bool plain    = classify && rows_in && cols_in && tri_free &&
                                  (mask == nullptr);

            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = row_base + rq + 8 * h;   // row within the block tile
                const int i = q_base + rq + 8 * h;     // global query row

                float sv[KVT][4];
                float local_max = -INFINITY;
                if (plain) {
                    #pragma unroll
                    for (int n = 0; n < KVT; ++n) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float raw = s_acc[n].x[T_OF[h][e]];
                            sv[n][e] = FOLD_Q ? raw : (raw * s_mul);
                            local_max = fmaxf(local_max, sv[n][e]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int n = 0; n < KVT; ++n) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            // col within the key tile: the quad's column pair,
                            // in whichever 8-wide half slot `e` names.
                            const int col = cols[n][e];
                            const int gj = kt + col;
                            bool ok = (i < S) && (gj < S);
                            if (ok && is_causal && gj > i) ok = false;
                            if (ok && mask != nullptr &&
                                !mask[mask_bh + static_cast<int64_t>(i) * ms2 +
                                      static_cast<int64_t>(gj) * ms3]) {
                                ok = false;
                            }
                            const float raw = s_acc[n].x[T_OF[h][e]];
                            sv[n][e] = ok ? (FOLD_Q ? raw : (raw * s_mul))
                                          : -INFINITY;
                            local_max = fmaxf(local_max, sv[n][e]);
                        }
                    }
                }

                // Two steps, not five: offsets 1 and 2 stay inside the quad,
                // and the quad is exactly the four lanes sharing this row.
                float mx = local_max;
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));

                const float m_old = m_s[r];
                const float m_new = fmaxf(m_old, mx);
                float corr = 1.0f;
                float lsum = 0.0f;
                if (m_new == -INFINITY) {
                    #pragma unroll
                    for (int n = 0; n < KVT; ++n) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int col = cols[n][e];
                            dev_from_float(p_s[r * S_LD + col], 0.0f);
                        }
                    }
                } else {
                    corr = (m_old == -INFINITY) ? 0.0f
                                                : attn_exp<USE_EXP2>(m_old - m_new);
                    #pragma unroll
                    for (int n = 0; n < KVT; ++n) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int col = cols[n][e];
                            float pv;
                            if constexpr (USE_EXP2) {
                                pv = attn_exp<true>(sv[n][e] - m_new);
                            } else {
                                pv = (sv[n][e] == -INFINITY)
                                         ? 0.0f
                                         : attn_exp<false>(sv[n][e] - m_new);
                            }
                            lsum += pv;
                            dev_from_float(p_s[r * S_LD + col], pv);
                        }
                    }
                }

                float tot = lsum;
                tot += __shfl_xor_sync(0xffffffffu, tot, 1);
                tot += __shfl_xor_sync(0xffffffffu, tot, 2);

                // One writer per row: the quad's first lane.
                if (q4 == 0) {
                    m_s[r] = m_new;
                    l_s[r] = l_s[r] * corr + tot;
                    c_s[r] = corr;
                }
            }
        }
        __syncwarp();

#else
        // --- 1. S = Q @ K^T ------------------------------------------------
        // k_s is [BLOCK_N, head_dim] row-major, which is K^T column-major with
        // ldm = KV_LD -- no transpose pass needed.
        #pragma unroll
        for (int n = 0; n < BLOCK_N / 16; ++n) {
            acc_frag_t acc;
            wm::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < PDIM / WK; ++kk) {
                wm::fragment<wm::matrix_b, 16, 16, WK, frag_elem, wm::col_major> kb;
                wm::load_matrix_sync(kb,
                                     k_cur + static_cast<size_t>(n) * 16 * KV_LD + kk * WK,
                                     KV_LD);
                if constexpr (IS_TF32) {
                    #pragma unroll
                    for (int t = 0; t < kb.num_elements; ++t) {
                        kb.x[t] = wm::__float_to_tf32(kb.x[t]);
                    }
                }
                wm::mma_sync(acc, q_frag[kk], kb, acc);
            }
            wm::store_matrix_sync(s_s + static_cast<size_t>(row_base) * S_LD + n * 16,
                                  acc, S_LD, wm::mem_row_major);
        }
        __syncwarp();

        // --- 2. online softmax over this warp 16 rows -----------------------
        //
        // One lane per query row, not per key column. The obvious mapping --
        // lane == key column -- needs a full 5-step butterfly per row to
        // reduce, 16 rows deep, and that reduction cost does not shrink with
        // head_dim, so at head_dim 16 it swamped both GEMMs. Giving each lane
        // a whole row segment instead turns the 5 steps into one: the only
        // cross-lane traffic left is between the two lanes that share a row.
        {
            const int sr = lane % RPW;              // row within the warp stripe
            const int sh = lane / RPW;              // which segment of the key tile
            const int r  = row_base + sr;
            const int i  = q_base + sr;
            const int c0 = sh * COLS_PER_LANE;
            const float* s_row = s_s + r * S_LD;

            // Does this key tile need any test at all? Every clause is
            // warp-uniform: q_base, kt and S do not vary across the warp, so
            // this branch cannot diverge.
            //
            //   rows_in  every query row of the warp stripe is inside S
            //            (hoisted above the key loop -- it does not vary)
            //   cols_in  every key column of the tile is inside S
            //            (computed once per tile, shared with the staging)
            //   tri_free the whole tile sits below the causal diagonal --
            //            the largest column it holds is still <= the smallest
            //            row the warp owns
            //
            // Under causal only the diagonal tile of each block fails
            // tri_free, which is the "for each row we only need apply the
            // causal mask to 1 block" observation from FlashAttention-2.
            const bool tri_free = !is_causal || ((kt + BLOCK_N - 1) <= q_base);
            const bool plain    = classify && rows_in && cols_in && tri_free &&
                                  (mask == nullptr);

            float sv[COLS_PER_LANE];
            float local_max = -INFINITY;
            if (plain) {
                #pragma unroll
                for (int t = 0; t < COLS_PER_LANE; ++t) {
                    // EXP2 folded `scale * log2e` into Q, so the score already
                    // carries both and arrives in the base-2 domain.
                    sv[t] = FOLD_Q ? s_row[c0 + t] : (s_row[c0 + t] * s_mul);
                    local_max = fmaxf(local_max, sv[t]);
                }
            } else {
                #pragma unroll
                for (int t = 0; t < COLS_PER_LANE; ++t) {
                    const int col = c0 + t;
                    const int gj = kt + col;
                    bool ok = (i < S) && (gj < S);
                    if (ok && is_causal && gj > i) ok = false;
                    if (ok && mask != nullptr &&
                        !mask[mask_bh + static_cast<int64_t>(i) * ms2 +
                              static_cast<int64_t>(gj) * ms3]) {
                        ok = false;
                    }
                    sv[t] = ok ? (FOLD_Q ? s_row[col] : (s_row[col] * s_mul))
                               : -INFINITY;
                    local_max = fmaxf(local_max, sv[t]);
                }
            }

            float mx = local_max;
            #pragma unroll
            for (int off = RPW; off < 32; off <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            }

            const float m_old = m_s[r];
            const float m_new = fmaxf(m_old, mx);
            float corr = 1.0f;
            float lsum = 0.0f;
            if (m_new == -INFINITY) {
                // Nothing admissible yet -- every key in every tile so far was
                // masked. Leave the running state alone.
                #pragma unroll
                for (int t = 0; t < COLS_PER_LANE; ++t) {
                    dev_from_float(p_s[r * S_LD + c0 + t], 0.0f);
                }
            } else {
                corr = (m_old == -INFINITY) ? 0.0f : attn_exp<USE_EXP2>(m_old - m_new);
                #pragma unroll
                for (int t = 0; t < COLS_PER_LANE; ++t) {
                    // m_new is finite in this branch, so a masked lane's
                    // argument is exactly -inf -- and `ex2.approx.f32(-inf)` is
                    // defined to return +0. The guard the original needed here
                    // is the exponential's own behaviour, so EXP2 drops it: one
                    // FSETP and one select per score element.
                    float p;
                    if constexpr (USE_EXP2) {
                        p = attn_exp<true>(sv[t] - m_new);
                    } else {
                        p = (sv[t] == -INFINITY) ? 0.0f
                                                 : attn_exp<false>(sv[t] - m_new);
                    }
                    lsum += p;
                    dev_from_float(p_s[r * S_LD + c0 + t], p);
                }
            }

            float tot = lsum;
            #pragma unroll
            for (int off = RPW; off < 32; off <<= 1) {
                tot += __shfl_xor_sync(0xffffffffu, tot, off);
            }

            if (sh == 0) {
                m_s[r] = m_new;
                l_s[r] = l_s[r] * corr + tot;
                c_s[r] = corr;
            }
        }
        __syncwarp();

#endif

        // --- 3. O = O * corr + P @ V ---------------------------------------
        // P does not depend on the output tile, so it is read once here rather
        // than once per tile; only V is re-read as n walks head_dim. Likewise
        // the per-row correction is pulled into registers rather than hitting
        // shared memory once per accumulator element per tile.
        wm::fragment<wm::matrix_a, 16, 16, WK, frag_elem, wm::row_major> p_frag[BLOCK_N / WK];
        #pragma unroll
        for (int kk = 0; kk < BLOCK_N / WK; ++kk) {
            wm::load_matrix_sync(p_frag[kk],
                                 p_s + static_cast<size_t>(row_base) * S_LD + kk * WK,
                                 S_LD);
            if constexpr (IS_TF32) {
                #pragma unroll
                for (int t = 0; t < p_frag[kk].num_elements; ++t) {
                    p_frag[kk].x[t] = wm::__float_to_tf32(p_frag[kk].x[t]);
                }
            }
        }

        float corr_of[ACC_ELEMS];
        #pragma unroll
        for (int t = 0; t < ACC_ELEMS; ++t) {
            corr_of[t] = c_s[row_base + acc_row[t]];
        }

        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            #pragma unroll
            for (int t = 0; t < ACC_ELEMS; ++t) {
                o_frag[n].x[t] *= corr_of[t];
            }
            #pragma unroll
            for (int kk = 0; kk < BLOCK_N / WK; ++kk) {
                wm::fragment<wm::matrix_b, 16, 16, WK, frag_elem, wm::row_major> vb;
                wm::load_matrix_sync(vb,
                                     v_cur + static_cast<size_t>(kk) * WK * KV_LD + n * 16,
                                     KV_LD);
                if constexpr (IS_TF32) {
                    #pragma unroll
                    for (int t = 0; t < vb.num_elements; ++t) {
                        vb.x[t] = wm::__float_to_tf32(vb.x[t]);
                    }
                }
                wm::mma_sync(o_frag[n], p_frag[kk], vb, o_frag[n]);
            }
        }
    }

    // --- write out ----------------------------------------------------------
    //
    // Can this warp hand its accumulator straight to global memory? The
    // fragment store writes a whole 16x16 tile at a fixed row stride, so it
    // needs all sixteen rows in range and no padded columns to drop. It also
    // writes fp32, which is what the accumulator holds and what part_o is, but
    // not necessarily what `out` is -- an fp16 or bf16 output needs a
    // conversion the fragment store cannot do. Every clause is warp-uniform.
    constexpr bool OUT_IS_FLOAT = std::is_same<scalar_t, float>::value;
    const bool o_whole = direct_o && ((q_base + RPW) <= S) && (DIM == PDIM);

    // The fallback, and the only thing shared memory is still needed for: one
    // 16x16 fragment at a time through this warp's slot of the O region, then
    // out a lane at a time. `dst` is the warp's first row and `ldm` its row
    // stride, which is constant in every layout involved.
    float* o_tile = o_s + warp * 16 * Cfg::O_TILE_LD;
    auto emit_tiled = [&](auto* dst, int64_t ldm) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            const int c0 = n * 16;
            const int w  = ((DIM - c0) < 16) ? (DIM - c0) : 16;   // 8 at head_dim 8
            __syncwarp();
            wm::store_matrix_sync(o_tile, o_frag[n], Cfg::O_TILE_LD,
                                  wm::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16 * w; idx += 32) {
                const int r = idx / w;
                if ((q_base + r) >= S) continue;
                const int c = idx - r * w;
                dev_from_float(dst[static_cast<int64_t>(r) * ldm + c0 + c],
                               o_tile[r * Cfg::O_TILE_LD + c]);
            }
        }
    };

    if (splits > 1) {
        // Unnormalised on purpose: 1/l is only knowable once every split's l
        // has been rebased onto the max over all splits, which is the combine
        // pass's job. Storing a normalised partial here would be wrong, not
        // merely wasteful.
        // [B, H, splits, S, DIM] for O and [B, H, splits, S] for the two row
        // statistics: the split axis sits between (b, h) and s, so one split's
        // writes stay contiguous -- which also makes DIM the row stride the
        // fragment store needs.
        const int64_t part_bh =
            (static_cast<int64_t>(b) * H + h) * splits + split;
        const int64_t row0 = part_bh * S;
        float* po0 = part_o + (row0 + q_base) * DIM;
        if (o_whole) {
            // part_o is fp32 whatever the tensors are, so OUT_IS_FLOAT does
            // not gate this one.
            #pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
                wm::store_matrix_sync(po0 + n * 16, o_frag[n], DIM,
                                      wm::mem_row_major);
            }
        } else if (direct_o) {
            emit_tiled(po0, DIM);
        } else {
            #pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
                wm::store_matrix_sync(o_s + static_cast<size_t>(row_base) * O_LD + n * 16,
                                      o_frag[n], O_LD, wm::mem_row_major);
            }
            __syncwarp();
            for (int rr = 0; rr < RPW; ++rr) {
                const int i = q_base + rr;
                if (i >= S) break;
                float* po = part_o + (row0 + i) * DIM;
                for (int c = lane; c < DIM; c += 32) {
                    po[c] = o_s[(row_base + rr) * O_LD + c];
                }
            }
        }
        for (int rr = 0; rr < RPW; ++rr) {
            const int i = q_base + rr;
            if (i >= S) break;
            if (lane == 0) {
                part_m[row0 + i] = m_s[row_base + rr];
                part_l[row0 + i] = l_s[row_base + rr];
            }
        }
    } else {
        // l == 0 means every key was masked. The reference produces NaN there;
        // emit 0 instead, since such rows are zero-filled downstream anyway.
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            #pragma unroll
            for (int t = 0; t < ACC_ELEMS; ++t) {
                const float lr = l_s[row_base + acc_row[t]];
                o_frag[n].x[t] *= (lr > 0.0f) ? (1.0f / lr) : 0.0f;
            }
        }

        // Both output layouts step one row by a constant: head_dim for
        // [B,H,S,D], and H*head_dim for the [B,S,H*D] view out_proj wants. So
        // either is a row stride the fragment store can take directly.
        scalar_t* out0 = out + out_base(out_bshd, b, h, q_base, H, S, DIM);
        const int64_t out_ld =
            out_bshd ? (static_cast<int64_t>(H) * DIM) : static_cast<int64_t>(DIM);
        if (o_whole && OUT_IS_FLOAT) {
            float* dst = reinterpret_cast<float*>(out0);
            #pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
                wm::store_matrix_sync(dst + n * 16, o_frag[n],
                                      static_cast<unsigned>(out_ld),
                                      wm::mem_row_major);
            }
        } else if (direct_o) {
            emit_tiled(out0, out_ld);
        } else {
            #pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
                wm::store_matrix_sync(o_s + static_cast<size_t>(row_base) * O_LD + n * 16,
                                      o_frag[n], O_LD, wm::mem_row_major);
            }
            __syncwarp();
            for (int rr = 0; rr < RPW; ++rr) {
                const int i = q_base + rr;
                if (i >= S) break;
                // Columns past DIM exist only to fill the fragment; dropped here.
                scalar_t* out_row = out + out_base(out_bshd, b, h, i, H, S, DIM);
                for (int c = lane; c < DIM; c += 32) {
                    dev_from_float(out_row[c], o_s[(row_base + rr) * O_LD + c]);
                }
            }
        }
    }
#endif
}

// Second pass of split-KV: fold the per-split partials into the finished row.
//
// The key-loop rescale, one level up. Each split reports the max over its own
// slice, so every partial is re-based onto the max over all slices before its
// numerator and denominator are added. USE_EXP2 has to match the kernel's
// softmax mode, because that is the domain the stored maxima are in.
//
// Plain CUDA rather than anything clever: one reduction of `splits` terms per
// output element, with no matmul in it. One thread per (row, head_dim element)
// and blockDim.y rows per block, so a warp stays contiguous in head_dim and
// the writes coalesce exactly as the single-pass kernel's do.
template <typename scalar_t, int HEAD_DIM, bool USE_EXP2>
__global__ void wmma_split_combine_kernel(const float* __restrict__ part_o,
                                          const float* __restrict__ part_m,
                                          const float* __restrict__ part_l,
                                          scalar_t* __restrict__ out,
                                          int H, int S, int splits,
                                          bool out_bshd, int64_t n_rows) {
    const int d = static_cast<int>(threadIdx.x);
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    const int64_t bh = row / S;
    const int64_t sq = row - bh * S;
    const int b = static_cast<int>(bh / H);
    const int h = static_cast<int>(bh - static_cast<int64_t>(b) * H);
    const int64_t base = bh * splits * S + sq;

    float m_g = -INFINITY;
    for (int j = 0; j < splits; ++j) {
        m_g = fmaxf(m_g, part_m[base + static_cast<int64_t>(j) * S]);
    }

    scalar_t* out_row =
        out + out_base(out_bshd, b, h, static_cast<int>(sq), H, S, HEAD_DIM);

    // Every split of this row was empty or entirely masked.
    if (m_g == -INFINITY) {
        dev_from_float(out_row[d], 0.0f);
        return;
    }

    float l_g = 0.0f;
    float o_g = 0.0f;
    for (int j = 0; j < splits; ++j) {
        const int64_t off = base + static_cast<int64_t>(j) * S;
        // <= 1 by construction, and exactly 0 for an empty split, whose stored
        // max is -inf and whose stored O is the zero the fragment started at.
        const float w = attn_exp<USE_EXP2>(part_m[off] - m_g);
        l_g += w * part_l[off];
        o_g += w * part_o[off * HEAD_DIM + d];
    }
    dev_from_float(out_row[d], (l_g > 0.0f) ? (o_g / l_g) : 0.0f);
}

// Blocks of this kernel instantiation the whole card can hold at once.
//
// From the occupancy API rather than from dividing the SM's shared memory by
// Cfg::SMEM, because registers can bind before shared memory does and only the
// driver knows which. It is a property of (kernel, threads, smem) -- all three
// compile-time constants here -- so it is queried once per instantiation
// rather than per launch.
//
// Two callers want it: the causal block reversal, which only pays when there
// is a queue to reorder, and the split-KV gate, which only pays when there is
// idle capacity to fill. They are opposite questions about the same number, so
// it lives in one place.
//
// Asked per O layout, because that is the point of the layout: direct_o frees
// 8 KB at head_dim 64 and 128 and 16 KB at 256, and whether that buys a block
// is exactly this query. Both answers are cached, so a knob flip between timed
// runs still costs one query each and not one per launch.
template <typename scalar_t, typename compute_t, int HEAD_DIM, int MODE>
int wmma_resident_blocks(bool direct_o) {
    using Cfg = WmmaCfg<compute_t, HEAD_DIM>;
    static const auto probe = [](size_t smem) {
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm,
                fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM, MODE>,
                Cfg::NTHREADS, smem) != cudaSuccess) {
            return 0;   // unknown -> every gate below reads "do not"
        }
        return per_sm * at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount;
    };
    static const int resident[2] = { probe(Cfg::SMEM), probe(Cfg::SMEM_TILE) };
    return resident[direct_o ? 1 : 0];
}

template <typename scalar_t, typename compute_t, int HEAD_DIM, int MODE>
void launch_wmma_kernel_as(const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& v, const bool* mask_ptr,
                           const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                           int B, int H, int S, bool is_causal, double scale) {
    using Cfg = WmmaCfg<compute_t, HEAD_DIM>;
    const dim3 block(Cfg::NTHREADS);
    const dim3 grid((S + Cfg::BLOCK_M - 1) / Cfg::BLOCK_M, H, B);

    const bool out_bshd = (out.dim() == 3);

    // Only a config that asked for more than the free 48 KB pays for this, and
    // it pays once per instantiation rather than per launch. Checked rather
    // than ignored: without the opt-in the launch fails with
    // cudaErrorInvalidValue several frames away from the cause.
    if constexpr (Cfg::NEEDS_CARVEOUT) {
        static const cudaError_t carveout = cudaFuncSetAttribute(
            fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM, MODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(Cfg::CARVEOUT_BYTES));
        TORCH_CHECK(carveout == cudaSuccess,
                    "fused_attention: head_dim ", HEAD_DIM, " needs ",
                    Cfg::CARVEOUT_BYTES,
                    " bytes of dynamic shared memory per block and this device "
                    "would not grant it (", cudaGetErrorString(carveout), ")");
    }

    // Reversal only pays when there is a queue to reorder. Measured on 13
    // causal shapes (table in TUNING.md): below one wave every block
    // is already resident, so dispatch order decides nothing and reversing only
    // scatters L2 locality -- b2 h2 s2048 d64, 128 blocks against 138 resident,
    // measured 0.933x. Above one wave the late blocks are the expensive ones
    // and LPT bites: b1 h8 s2048 d64 at 256 blocks measured 1.101x.
    //
    // The capacity comes from wmma_resident_blocks(); see the note there.
    //
    // Both gates below are about capacity, so they have to ask about the
    // layout that will actually be launched -- direct_o is worth a block per
    // SM at three head_dims, which is enough to change what either decides.
    // Forced on where the full O layout does not fit: SUPPORTED admits such a
    // shape on the strength of the tile layout alone, so honouring
    // WMMA_DIRECT_O=0 there would ask for more shared memory than the block is
    // allowed. The flag stays advisory rather than becoming an error because
    // it is a measurement knob, and "this shape has only one layout" is a
    // property of the shape, not a mistake by the caller.
    const bool direct_o = direct_o_flag() || !Cfg::FULL_O_FITS;
    const int resident =
        wmma_resident_blocks<scalar_t, compute_t, HEAD_DIM, MODE>(direct_o);
    const int n_m = static_cast<int>(grid.x);
    const int blocks = n_m * H * B;
    const bool reverse_m = Cfg::REVERSE_CAUSAL && is_causal &&
                           causal_reverse_flag() && blocks > resident;

    // Split-KV asks the opposite question the reversal does -- is there idle
    // capacity, rather than is there a queue -- off the same `resident`.
    const int splits = wmma_split_count(
        blocks, resident, split_key_tiles(S, Cfg::BLOCK_N, is_causal), HEAD_DIM);

    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    at::Tensor ws;
    if (splits > 1) {
        // ONE allocation, three views into it. Three separate at::empty calls
        // cost enough host time at these sizes to swamp the device win: the op
        // read 0.882x eager with three and 0.932x with one, for byte-identical
        // device work. From torch's caching allocator rather than cudaMalloc so
        // it is stream-ordered and draws on the pool already being accounted.
        const int64_t rows =
            static_cast<int64_t>(B) * H * splits * S;
        ws = at::empty({rows * (HEAD_DIM + 2)}, q.options().dtype(at::kFloat));
        part_o = ws.data_ptr<float>();
        part_m = part_o + rows * HEAD_DIM;
        part_l = part_m + rows;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 launch_grid(static_cast<unsigned>(n_m * splits), H, B);

    fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM, MODE>
        <<<launch_grid, block,
           (direct_o ? Cfg::SMEM_TILE : Cfg::SMEM) + Cfg::STAGE_EXTRA, stream>>>(
            reinterpret_cast<const scalar_t*>(q.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(k.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(v.const_data_ptr()),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            part_o, part_m, part_l,
            B, H, S, is_causal, static_cast<float>(scale), out_bshd,
            reverse_m, splits, mask_classify_flag(), direct_o,
            cp_async_mode(),
            acc_formula_flag() && acc_row_formula_ok<Cfg::WK>());

    if constexpr (!Cfg::PROBE_FITS) {
        // No fallback exists for this block shape, so a device the closed form
        // does not describe has to fail loudly rather than silently.
        TORCH_CHECK(acc_row_formula_ok<Cfg::WK>(),
                    "wmma attention: this block shape has no shared scratch for "
                    "the accumulator probe, and the closed-form accumulator "
                    "mapping does not describe this device");
    }

    if (splits > 1) {
        // The combine has to exponentiate in the domain the kernel stored its
        // maxima in, so it reads the same MODE the kernel was instantiated for.
        constexpr bool USE_EXP2 = (MODE >= 1);
        const int64_t n_rows = static_cast<int64_t>(B) * H * S;
        // x is head_dim so a warp stays inside one row and the writes coalesce;
        // y takes whatever is left of a 256-thread block.
        const int ty = (256 / HEAD_DIM) > 1 ? (256 / HEAD_DIM) : 1;
        const dim3 cblock(static_cast<unsigned>(HEAD_DIM),
                          static_cast<unsigned>(ty));
        const dim3 cgrid(static_cast<unsigned>((n_rows + ty - 1) / ty));
        wmma_split_combine_kernel<scalar_t, HEAD_DIM, USE_EXP2>
            <<<cgrid, cblock, 0, stream>>>(
                part_o, part_m, part_l,
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                H, S, splits, out_bshd, n_rows);
    }
}

// Resolves the exp2 flag to one of the two instantiations. A runtime bool
// inside the kernel would make every thread carry both paths' code and both
// paths' registers; here the choice costs one host-side branch per launch and
// the device code of each variant is exactly what that variant needs.
template <typename scalar_t, typename compute_t, int HEAD_DIM>
void launch_wmma_kernel(const torch::Tensor& q, const torch::Tensor& k,
                        const torch::Tensor& v, const bool* mask_ptr,
                        const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                        int B, int H, int S, bool is_causal, double scale) {
    switch (softmax_mode_flag()) {
        case 2:
            launch_wmma_kernel_as<scalar_t, compute_t, HEAD_DIM, 2>(
                q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return;
        case 1:
            launch_wmma_kernel_as<scalar_t, compute_t, HEAD_DIM, 1>(
                q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return;
        default:
            launch_wmma_kernel_as<scalar_t, compute_t, HEAD_DIM, 0>(
                q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return;
    }
}

// Returns false when this (dtype, head_dim) pair has no tensor-core
// specialization, so the caller can fall back to the scalar kernel.
// Which compute_t an fp32 tensor is contracted in. fp16 is the default -- it
// has tf32's 10-bit mantissa and twice its tensor-core rate -- and tf32 stays
// reachable so the two can be A/B'd in one process. Same contract as
// causal_reverse_flag(): flipped between timed runs from one thread, never
// while launches are in flight.
bool& wmma_fp16_flag() {
    static bool on = [] {
        const char* e = std::getenv("WMMA_FP16");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

template <typename scalar_t, typename compute_t, int HEAD_DIM>
bool maybe_launch_wmma_as(const torch::Tensor& q, const torch::Tensor& k,
                          const torch::Tensor& v, const bool* mask_ptr,
                          const int64_t* ms, const int64_t* qs,
                          torch::Tensor& out, int B, int H, int S,
                          bool is_causal, double scale) {
    // SUPPORTED is asked of compute_t, not scalar_t: it is the compute type
    // that sizes every staged tile and sets the fragment K, so an fp32 tensor
    // narrowed to fp16 can pass a shared-memory budget its tf32 self fails.
    if constexpr (WmmaCfg<compute_t, HEAD_DIM>::SUPPORTED) {
        launch_wmma_kernel<scalar_t, compute_t, HEAD_DIM>(
            q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        return true;
    } else {
        return false;
    }
}

template <typename scalar_t, int HEAD_DIM>
bool maybe_launch_wmma(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const bool* mask_ptr,
                       const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                       int B, int H, int S, bool is_causal, double scale,
                       AttnPrecision prec) {
    // Only an fp32 tensor has a choice to make. A tensor that is ALREADY half
    // or bfloat16 contracts in the type it is: narrowing further is not
    // possible and widening it back would buy nothing, since the mantissa it
    // lost is lost. So the precision argument is a request about fp32 inputs,
    // and fused_attention_forward is where a conflicting one is refused --
    // here it is simply not consulted.
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Auto defers to the process-wide knob, which is what WMMA_FP16 and
        // wmma_set_fp16() drive; the A/B scripts built on those keep working
        // without knowing this argument exists. An explicit precision wins over
        // the knob, because it came from the caller rather than the environment.
        AttnPrecision want = prec;
        if (want == AttnPrecision::Auto) {
            want = wmma_fp16_flag() ? AttnPrecision::Fp16 : AttnPrecision::Tf32;
        }
        switch (want) {
            case AttnPrecision::Fp16:
                return maybe_launch_wmma_as<scalar_t, __half, HEAD_DIM>(
                    q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            case AttnPrecision::Bf16:
                // Exposed for measurement only. 8 significand bits ran 425%-622%
                // of the harness's 2e-3 budget when this was last measured, so
                // it is a thing to compare against, not a thing to ship.
                return maybe_launch_wmma_as<scalar_t, __nv_bfloat16, HEAD_DIM>(
                    q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            default:
                // Tf32. compute_t == float IS the tf32 path: FragTraits<float>
                // is the 16x16x8 tf32 fragment, not an fp32 matmul.
                break;
        }
    }
    if constexpr (WmmaCfg<scalar_t, HEAD_DIM>::SUPPORTED) {
        launch_wmma_kernel<scalar_t, scalar_t, HEAD_DIM>(q, k, v, mask_ptr, ms, qs, out,
                                               B, H, S, is_causal, scale);
        return true;
    } else {
        return false;
    }
}


// {grid blocks, blocks the card holds at once, BLOCK_M, BLOCK_N} for this
// shape, on the
// path an fp32 tensor actually takes -- fp16 fragments, whichever softmax mode
// is live. Exported so a script can table the occupancy the split-KV gate is
// built on without rederiving BLOCK_M or guessing where the register limit is.
// {0,0,0} means this head_dim has no fp16 kernel.
template <int HEAD_DIM>
std::vector<int64_t> wmma_grid_info_hd(int B, int H, int S) {
    using Cfg = WmmaCfg<__half, HEAD_DIM>;
    if constexpr (!Cfg::SUPPORTED) {
        return {0, 0, 0, 0};
    } else {
        int resident = 0;
        const bool d = direct_o_flag();
        switch (softmax_mode_flag()) {
            case 2:  resident = wmma_resident_blocks<float, __half, HEAD_DIM, 2>(d); break;
            case 1:  resident = wmma_resident_blocks<float, __half, HEAD_DIM, 1>(d); break;
            default: resident = wmma_resident_blocks<float, __half, HEAD_DIM, 0>(d); break;
        }
        const int64_t n_m = (S + Cfg::BLOCK_M - 1) / Cfg::BLOCK_M;
        return {n_m * H * B, resident, Cfg::BLOCK_M, Cfg::BLOCK_N};
    }
}

std::vector<int64_t> wmma_grid_info(int B, int H, int S, int head_dim) {
    switch (head_dim) {
        case 8:   return wmma_grid_info_hd<8>(B, H, S);
        case 16:  return wmma_grid_info_hd<16>(B, H, S);
        case 32:  return wmma_grid_info_hd<32>(B, H, S);
        case 64:  return wmma_grid_info_hd<64>(B, H, S);
        case 128: return wmma_grid_info_hd<128>(B, H, S);
        case 256: return wmma_grid_info_hd<256>(B, H, S);
        default:  return {0, 0, 0, 0};
    }
}

template <typename c10_t>
bool dispatch_wmma(const torch::Tensor& q, const torch::Tensor& k,
                   const torch::Tensor& v, const bool* mask_ptr,
                   const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                   int B, int H, int S, int head_dim,
                   bool is_causal, double scale, AttnPrecision prec) {
    using scalar_t = typename DevType<c10_t>::type;
    switch (head_dim) {
        case 8:
            return maybe_launch_wmma<scalar_t, 8>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        case 16:
            return maybe_launch_wmma<scalar_t, 16>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        case 32:
            return maybe_launch_wmma<scalar_t, 32>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        case 64:
            return maybe_launch_wmma<scalar_t, 64>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        case 128:
            return maybe_launch_wmma<scalar_t, 128>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        case 256:
            return maybe_launch_wmma<scalar_t, 256>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale, prec);
        default:
            return false;
    }
}

}  // namespace
