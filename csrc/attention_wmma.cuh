// Tensor-core kernel.
//
// Block owns BLOCK_M=64 query rows; 4 warps, one 16-row stripe each. Keys are
// walked in BLOCK_N=32 tiles. Per tile a warp does:
//
//   1. S = Q @ K^T           16x32 via wmma, accumulate fp32   -> smem
//   2. online softmax on those 16 rows                          -> P in smem
//   3. O = O * corr + P @ V  16xD  via wmma, straight into the O fragments
//
// Q and O both stay in registers for the whole key loop, so the only traffic
// in the inner loop is the K/V tile, the score tile, and the fragment reads
// that feed the MMA units.
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
template <typename scalar_t, typename compute_t, int HEAD_DIM>
__global__ __launch_bounds__(WmmaCfg<compute_t, HEAD_DIM>::NTHREADS)
void fused_attention_wmma_kernel(const scalar_t* __restrict__ q,
                                 const scalar_t* __restrict__ k,
                                 const scalar_t* __restrict__ v,
                                 int64_t qs0, int64_t qs1, int64_t qs2,
                                 const bool* __restrict__ mask,
                                 int64_t ms0, int64_t ms1,
                                 int64_t ms2, int64_t ms3,
                                 scalar_t* __restrict__ out,
                                 int B, int H, int S,
                                 bool is_causal, float scale,
                                 bool out_bshd, bool reverse_m) {
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
    const int m_tile = reverse_m ? (static_cast<int>(gridDim.x) - 1 -
                                    static_cast<int>(blockIdx.x))
                                 : static_cast<int>(blockIdx.x);
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
        for (int idx = tid; idx < BLOCK_M * PDIM; idx += Cfg::NTHREADS) {
            const int r = idx / PDIM;
            const int c = idx - r * PDIM;
            const int gr = m_tile * BLOCK_M + r;
            q_s[r * KV_LD + c] =
                (gr < S && c < DIM)
                    ? dev_of_float<compute_t>(dev_to_float(
                          q[bh_off + static_cast<int64_t>(gr) * qs2 + c]))
                    : zero_v;
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

    for (int kt = 0; kt < key_limit; kt += BLOCK_N) {
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
                sv[t] = ok ? (s_row[col] * scale) : -INFINITY;
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
                corr = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                #pragma unroll
                for (int t = 0; t < COLS_PER_LANE; ++t) {
                    const float p = (sv[t] == -INFINITY) ? 0.0f : __expf(sv[t] - m_new);
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

    // --- normalize and write out --------------------------------------------
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
        // Columns past DIM exist only to fill the fragment; they are dropped.
        scalar_t* out_row = out + out_base(out_bshd, b, h, i, H, S, DIM);
        for (int c = lane; c < DIM; c += 32) {
            dev_from_float(out_row[c], o_s[r * O_LD + c]);
        }
    }
#endif
}

template <typename scalar_t, typename compute_t, int HEAD_DIM>
void launch_wmma_kernel(const torch::Tensor& q, const torch::Tensor& k,
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
    // The capacity comes from the occupancy API rather than from dividing
    // 100 KB by Cfg::SMEM, because registers can bind before shared memory does
    // and only the driver knows which. It is a property of (kernel, threads,
    // smem) -- all three compile-time constants here -- so it is queried once
    // per instantiation, not per launch.
    static const int resident = [] {
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM>,
                Cfg::NTHREADS, Cfg::SMEM) != cudaSuccess) {
            return 0;   // unknown -> never reverse, i.e. keep today's mapping
        }
        return per_sm * at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount;
    }();
    const int blocks = static_cast<int>(grid.x) * H * B;
    const bool reverse_m = Cfg::REVERSE_CAUSAL && is_causal &&
                           causal_reverse_flag() && blocks > resident;

    fused_attention_wmma_kernel<scalar_t, compute_t, HEAD_DIM>
        <<<grid, block, Cfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const scalar_t*>(q.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(k.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(v.const_data_ptr()),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            B, H, S, is_causal, static_cast<float>(scale), out_bshd,
            reverse_m);
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
