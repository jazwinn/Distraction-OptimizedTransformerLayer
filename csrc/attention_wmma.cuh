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
#ifndef WMMA_M_32
#define WMMA_M_32  64
#define WMMA_N_32  32
#endif
#ifndef WMMA_M_64
#define WMMA_M_64  64
#define WMMA_N_64  16
#endif
#ifndef WMMA_M_128
#define WMMA_M_128 32
#define WMMA_N_128 16
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
    static constexpr bool P_ALIASES_S = std::is_same<scalar_t, float>::value;

    static constexpr int KV_LD = PDIM + PAD;       // k_s, v_s, and Q staging
    static constexpr int O_LD  = PDIM + 4;         // o_s is always fp32
    static constexpr int S_LD  = BLOCK_N + PAD;    // s_s and p_s

    static constexpr size_t Q_BYTES   = sizeof(scalar_t) * BLOCK_M * KV_LD;
    static constexpr size_t O_BYTES   = sizeof(float) * BLOCK_M * O_LD;
    static constexpr size_t QO_BYTES  = (Q_BYTES > O_BYTES) ? Q_BYTES : O_BYTES;
    static constexpr size_t KV_BYTES  = sizeof(scalar_t) * BLOCK_N * KV_LD;
    static constexpr size_t S_BYTES   = sizeof(float) * BLOCK_M * S_LD;
    static constexpr size_t P_BYTES   = P_ALIASES_S ? 0 : sizeof(scalar_t) * BLOCK_M * S_LD;
    static constexpr size_t ROW_BYTES = sizeof(float) * BLOCK_M;

    static constexpr size_t O_OFF = 0;
    static constexpr size_t K_OFF = O_OFF + QO_BYTES;
    static constexpr size_t V_OFF = K_OFF + KV_BYTES;
    static constexpr size_t S_OFF = V_OFF + KV_BYTES;
    static constexpr size_t P_OFF = S_OFF + S_BYTES;
    static constexpr size_t M_OFF = P_OFF + P_BYTES;
    static constexpr size_t L_OFF = M_OFF + ROW_BYTES;
    static constexpr size_t C_OFF = L_OFF + ROW_BYTES;
    static constexpr size_t SMEM  = C_OFF + ROW_BYTES;

    // The accumulator probe below needs 512 floats of scratch per warp. It runs
    // after Q has been hoisted into registers and before the first K/V tile is
    // staged, so the whole O/K/V/S span is dead and can host it -- but that
    // span has to actually be big enough.
    static constexpr size_t PROBE_BYTES = sizeof(float) * WARPS * 512;
    static constexpr size_t SCRATCH_BYTES = QO_BYTES + 2 * KV_BYTES + S_BYTES;

    // Whether causal blocks are worth dispatching longest-first. Per head_dim
    // for the same reason WmmaShape is: head_dim 128 runs a 32x16 block of two
    // warps at ~36 KB, so only two blocks and 128 threads land on an SM. With
    // that little in flight the kernel is bound by K/V locality rather than by
    // makespan, and reordering the dispatch costs more L2 reuse than it saves
    // tail -- measured 0.889x-0.966x over five shapes, against 1.02x-1.09x at
    // head_dim 16 through 64. scripts/ab_causal_reverse.py has the table.
    static constexpr bool REVERSE_CAUSAL = (HEAD_DIM <= 64);

    // GEMM1 contracts over the padded head_dim, GEMM2 over the key tile; both
    // must be a whole number of fragments. Staying under 48 KB keeps two blocks
    // resident per SM without having to opt in to the larger dynamic
    // shared-memory carveout.
    static constexpr bool SUPPORTED =
        FragTraits<scalar_t>::supported &&
        (PDIM % WK == 0) && (PDIM % 16 == 0) &&
        (BLOCK_N % WK == 0) && (BLOCK_M % 16 == 0) &&
        (SCRATCH_BYTES >= PROBE_BYTES) && (SMEM <= 48 * 1024);
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
                                 bool out_bshd, bool reverse_m, int splits) {
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
    float*    o_s = reinterpret_cast<float*>(smem_raw + Cfg::O_OFF);
    compute_t* k_s = reinterpret_cast<compute_t*>(smem_raw + Cfg::K_OFF);
    compute_t* v_s = reinterpret_cast<compute_t*>(smem_raw + Cfg::V_OFF);
    float*    s_s = reinterpret_cast<float*>(smem_raw + Cfg::S_OFF);
    float*    m_s = reinterpret_cast<float*>(smem_raw + Cfg::M_OFF);
    float*    l_s = reinterpret_cast<float*>(smem_raw + Cfg::L_OFF);
    float*    c_s = reinterpret_cast<float*>(smem_raw + Cfg::C_OFF);
    // P feeds the second GEMM as a matrix_a fragment, so it has to be in the
    // operand type. For fp32 that is the same type the scores were stored in,
    // so the softmax can overwrite the score tile in place.
    compute_t* p_s = Cfg::P_ALIASES_S ? reinterpret_cast<compute_t*>(s_s)
                                      : reinterpret_cast<compute_t*>(smem_raw + Cfg::P_OFF);

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
    static_assert(Cfg::SCRATCH_BYTES >= Cfg::PROBE_BYTES,
                  "shared scratch is too small to host the per-warp accumulator probe");
    int acc_row[ACC_ELEMS];
    {
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
    int kt_begin = 0;
    int kt_end   = key_limit;
    if (splits > 1) {
        const int n_kt = (key_limit + BLOCK_N - 1) / BLOCK_N;
        const int per  = (n_kt + splits - 1) / splits;
        kt_begin = min(split * per * BLOCK_N, key_limit);
        kt_end   = min(kt_begin + per * BLOCK_N, key_limit);
    }

    for (int kt = kt_begin; kt < kt_end; kt += BLOCK_N) {
        __syncthreads();  // everyone is done reading the previous k_s/v_s

        // head_dim is stride-1 whatever the caller's layout, so a key row is
        // one flat span and the global reads stay coalesced; qs2 is only the
        // spacing between rows. Rows past S are zeroed: they are masked out of
        // the scores anyway, but a NaN in v_s would survive `0 * v` in GEMM2.
        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * qs2;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * qs2;
        for (int idx = tid; idx < BLOCK_N * PDIM; idx += Cfg::NTHREADS) {
            const int r = idx / PDIM;
            const int c = idx - r * PDIM;
            const bool inb = ((kt + r) < S) && (c < DIM);
            const int64_t g = static_cast<int64_t>(r) * qs2 + c;
            k_s[r * KV_LD + c] =
                inb ? dev_of_float<compute_t>(dev_to_float(k_base[g])) : zero_v;
            v_s[r * KV_LD + c] =
                inb ? dev_of_float<compute_t>(dev_to_float(v_base[g])) : zero_v;
        }
        __syncthreads();

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
                                     k_s + static_cast<size_t>(n) * 16 * KV_LD + kk * WK,
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

            float sv[COLS_PER_LANE];
            float local_max = -INFINITY;
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
                // EXP2 folded `scale * log2e` into Q, so the score already
                // carries both and arrives in the base-2 domain.
                sv[t] = ok ? (FOLD_Q ? s_row[col] : (s_row[col] * s_mul))
                           : -INFINITY;
                local_max = fmaxf(local_max, sv[t]);
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
                                     v_s + static_cast<size_t>(kk) * WK * KV_LD + n * 16,
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
    if (splits > 1) {
        // Unnormalised on purpose: 1/l is only knowable once every split's l
        // has been rebased onto the max over all splits, which is the combine
        // pass's job. Storing a normalised partial here would be wrong, not
        // merely wasteful.
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            wm::store_matrix_sync(o_s + static_cast<size_t>(row_base) * O_LD + n * 16,
                                  o_frag[n], O_LD, wm::mem_row_major);
        }
        __syncwarp();

        // [B, H, splits, S, DIM] for O and [B, H, splits, S] for the two row
        // statistics: the split axis sits between (b, h) and s, so one split's
        // writes stay contiguous.
        const int64_t part_bh =
            (static_cast<int64_t>(b) * H + h) * splits + split;
        const int64_t row0 = part_bh * S;
        for (int rr = 0; rr < RPW; ++rr) {
            const int i = q_base + rr;
            if (i >= S) break;
            const int r = row_base + rr;
            float* po = part_o + (row0 + i) * DIM;
            for (int c = lane; c < DIM; c += 32) {
                po[c] = o_s[r * O_LD + c];
            }
            if (lane == 0) {
                part_m[row0 + i] = m_s[r];
                part_l[row0 + i] = l_s[r];
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
            wm::store_matrix_sync(o_s + static_cast<size_t>(row_base) * O_LD + n * 16,
                                  o_frag[n], O_LD, wm::mem_row_major);
        }
        __syncwarp();

        for (int rr = 0; rr < RPW; ++rr) {
            const int i = q_base + rr;
            if (i >= S) break;
            const int r = row_base + rr;
            // Columns past DIM exist only to fill the fragment; dropped here.
            scalar_t* out_row = out + out_base(out_bshd, b, h, i, H, S, DIM);
            for (int c = lane; c < DIM; c += 32) {
                dev_from_float(out_row[c], o_s[r * O_LD + c]);
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
template <typename scalar_t, typename compute_t, int HEAD_DIM, int MODE>
int wmma_resident_blocks() {
    using Cfg = WmmaCfg<compute_t, HEAD_DIM>;
    static const int resident = [] {
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm,
                fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM, MODE>,
                Cfg::NTHREADS, Cfg::SMEM) != cudaSuccess) {
            return 0;   // unknown -> every gate below reads "do not"
        }
        return per_sm * at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount;
    }();
    return resident;
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

    // Reversal only pays when there is a queue to reorder. Measured on 13
    // causal shapes (scripts/ab_causal_reverse.py): below one wave every block
    // is already resident, so dispatch order decides nothing and reversing only
    // scatters L2 locality -- b2 h2 s2048 d64, 128 blocks against 138 resident,
    // measured 0.933x. Above one wave the late blocks are the expensive ones
    // and LPT bites: b1 h8 s2048 d64 at 256 blocks measured 1.101x.
    //
    // The capacity comes from wmma_resident_blocks(); see the note there.
    const int resident =
        wmma_resident_blocks<scalar_t, compute_t, HEAD_DIM, MODE>();
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
        <<<launch_grid, block, Cfg::SMEM, stream>>>(
            reinterpret_cast<const scalar_t*>(q.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(k.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(v.const_data_ptr()),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            part_o, part_m, part_l,
            B, H, S, is_causal, static_cast<float>(scale), out_bshd,
            reverse_m, splits);

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
                       int B, int H, int S, bool is_causal, double scale) {
    // Only an fp32 tensor has a choice to make; half and bfloat16 contract in
    // the type they already are.
    if constexpr (std::is_same<scalar_t, float>::value) {
        if (wmma_fp16_flag()) {
            return maybe_launch_wmma_as<scalar_t, __half, HEAD_DIM>(
                q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
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
        switch (softmax_mode_flag()) {
            case 2:  resident = wmma_resident_blocks<float, __half, HEAD_DIM, 2>(); break;
            case 1:  resident = wmma_resident_blocks<float, __half, HEAD_DIM, 1>(); break;
            default: resident = wmma_resident_blocks<float, __half, HEAD_DIM, 0>(); break;
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
                   bool is_causal, double scale) {
    using scalar_t = typename DevType<c10_t>::type;
    switch (head_dim) {
        case 8:
            return maybe_launch_wmma<scalar_t, 8>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 16:
            return maybe_launch_wmma<scalar_t, 16>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 32:
            return maybe_launch_wmma<scalar_t, 32>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 64:
            return maybe_launch_wmma<scalar_t, 64>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 128:
            return maybe_launch_wmma<scalar_t, 128>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        default:
            return false;
    }
}

}  // namespace
