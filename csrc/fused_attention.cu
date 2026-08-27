// Custom fused attention extension.
//
// FlashAttention-style: the [B,H,S,S] score matrix is never written to global
// memory. K/V are streamed through shared memory a tile at a time and the
// softmax is accumulated online (running max + running sum).
//
// Three implementations of the same math:
//
//   wmma    tensor cores via nvcuda::wmma, 64 queries x 32 keys per block, 4
//           warps. fp32 goes through TF32 fragments with fp32 accumulate --
//           what cuBLAS does for the baseline under allow_tf32, so this path is
//           *closer* to the reference than the scalar one, not further.
//           half/bfloat16 use native 16x16x16 fragments. SM 8.0+, head_dim in
//           {8,16,32,64,128}.
//
//   scalar  one thread per query row, plain fp32 FMA. No tensor cores and no
//           TF32 rounding, so it is the pre-Ampere path and the exact-arithmetic
//           reference for A/B comparison (impl=1). head_dim in {8,16,32,64}.
//
//   tile    the same math on the CUDA tile programming model, in
//           tile_attention.cu -- a separate translation unit because it needs
//           -std=c++20 -enable-tile. float32 and head_dim in {8,16,32,64}, and
//           only when the build found CUDA 13.3+. impl=3/4/5 select fp32, bf16
//           and tf32 operands; none is picked by impl=0.
//
// A head_dim outside those sets -- 24 or 48, say -- falls back to the ATen
// implementation at the bottom, which mirrors BaselineSelfAttention.forward
// exactly. head_dim is a template parameter, so each supported value is a
// separately compiled kernel and the set cannot be open-ended.

#include "tile_attention.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <type_traits>
#include <vector>

namespace {

__global__ void identity_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Where row (b, h, i) of the output starts, for either output layout.
//
//   out_bshd == false  ->  [B, H, S, D], the natural per-head layout.
//   out_bshd == true   ->  [B, S, H*D], what out_proj consumes. Otherwise the
//                          caller pays a transpose(1,2) + reshape, which cannot
//                          be a view -- a full strided repack per layer. Free
//                          here: only the destination index changes, and
//                          head_dim stays the fastest-varying axis either way,
//                          so the stores stay just as coalesced.
__device__ __forceinline__ int64_t out_base(bool out_bshd, int b, int h, int i,
                                            int H, int S, int D) {
    const int64_t bi = b;
    const int64_t hi = h;
    const int64_t ii = i;
    return out_bshd
        ? (((bi * S + ii) * H + hi) * D)
        : (((bi * H + hi) * S + ii) * D);
}

// ---------------------------------------------------------------------------
// Scalar kernel: one thread owns one query row.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_DIM>
__global__ void fused_attention_kernel(const scalar_t* __restrict__ q,
                                       const scalar_t* __restrict__ k,
                                       const scalar_t* __restrict__ v,
                                       int64_t qs0, int64_t qs1, int64_t qs2,
                                       const bool* __restrict__ mask,
                                       int64_t ms0, int64_t ms1,
                                       int64_t ms2, int64_t ms3,
                                       scalar_t* __restrict__ out,
                                       int B, int H, int S,
                                       bool is_causal, float scale,
                                       bool out_bshd) {
    constexpr int BLOCK_M = 64;                              // query rows per block
    constexpr int BLOCK_N = (HEAD_DIM >= 64) ? 64 : 128;     // keys per shared tile

    extern __shared__ __align__(16) char smem_raw[];
    scalar_t* k_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* v_s = k_s + BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int m_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int i = m_tile * BLOCK_M + tid;
    const bool active = (i < S);

    const int64_t bh_off =
        static_cast<int64_t>(b) * qs0 + static_cast<int64_t>(h) * qs1;

    // q is read once and reused against every key, so it earns registers.
    float q_reg[HEAD_DIM];
    if (active) {
        const scalar_t* q_row = q + bh_off + static_cast<int64_t>(i) * qs2;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_reg[d] = static_cast<float>(q_row[d]);
        }
    }

    // Running softmax state, accumulated in float even for half inputs -- the
    // reference does its softmax in fp32 and we have to match that.
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc[d] = 0.0f;
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int64_t mask_bh =
        (mask != nullptr) ? (static_cast<int64_t>(b) * ms0 + static_cast<int64_t>(h) * ms1) : 0;

    // Under causal masking no thread in this block looks past the block's own
    // last query row, so whole tiles beyond it are skipped rather than computed
    // and thrown away. This is where the causal speedup actually comes from.
    const int key_limit = is_causal ? min(S, m_tile * BLOCK_M + BLOCK_M) : S;

    for (int kt = 0; kt < key_limit; kt += BLOCK_N) {
        const int n_keys = min(BLOCK_N, S - kt);

        // head_dim is stride-1 in every layout the caller can pass, so one key
        // row is still a flat span and the loads stay coalesced; only the
        // spacing between rows is a runtime value. Shared memory stays packed,
        // hence the flat destination index.
        //
        // Rows are the outer loop so the 64-bit row offset is computed once per
        // row rather than once per element. Flattening the two into a single
        // strided loop costs a divide, a modulo and a 64-bit multiply per
        // element, which at head_dim 128 -- where q_reg and acc alone are 256
        // registers and the kernel is already spilling -- measured 1.5x slower.
        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * qs2;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * qs2;
        for (int r = 0; r < n_keys; ++r) {
            const int64_t g = static_cast<int64_t>(r) * qs2;
            for (int c = tid; c < HEAD_DIM; c += BLOCK_M) {
                k_s[r * HEAD_DIM + c] = k_base[g + c];
                v_s[r * HEAD_DIM + c] = v_base[g + c];
            }
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < n_keys; ++j) {
                const int gj = kt + j;
                if (is_causal && gj > i) {
                    break;
                }
                if (mask != nullptr &&
                    !mask[mask_bh + static_cast<int64_t>(i) * ms2 +
                          static_cast<int64_t>(gj) * ms3]) {
                    continue;  // exp(-inf) == 0, so a masked key contributes nothing
                }

                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    s += q_reg[d] * static_cast<float>(k_s[j * HEAD_DIM + d]);
                }
                s *= scale;

                // Online softmax: rescale only when a new max appears, which
                // after the first few keys is rare.
                if (s > m_run) {
                    const float corr = __expf(m_run - s);
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        acc[d] *= corr;
                    }
                    l_run *= corr;
                    m_run = s;
                }

                const float p = __expf(s - m_run);
                l_run += p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[d] += p * static_cast<float>(v_s[j * HEAD_DIM + d]);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        scalar_t* out_row = out + out_base(out_bshd, b, h, i, H, S, HEAD_DIM);
        // l_run == 0 means every key was masked. The reference would produce
        // NaN there; emit 0 instead, since such rows are zero-filled downstream
        // anyway and NaN would only risk contaminating something else.
        const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            out_row[d] = static_cast<scalar_t>(acc[d] * inv);
        }
    }
}

template <typename scalar_t, int HEAD_DIM>
void launch_kernel(const torch::Tensor& q, const torch::Tensor& k,
                   const torch::Tensor& v, const bool* mask_ptr,
                   const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                   int B, int H, int S, bool is_causal, double scale) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = (HEAD_DIM >= 64) ? 64 : 128;

    const dim3 block(BLOCK_M);
    const dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    const size_t smem = 2 * BLOCK_N * HEAD_DIM * sizeof(scalar_t);

    // A 3-D output is [B, S, H*head_dim]; a 4-D one is [B, H, S, head_dim].
    // Reading the layout off the tensor keeps it out of every dispatch
    // signature between here and fused_attention_forward.
    const bool out_bshd = (out.dim() == 3);

    fused_attention_kernel<scalar_t, HEAD_DIM>
        <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            out.data_ptr<scalar_t>(), B, H, S, is_causal,
            static_cast<float>(scale), out_bshd);
}

// Returns false when this head_dim has no specialization compiled.
template <typename scalar_t>
bool dispatch_head_dim(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const bool* mask_ptr,
                       const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                       int B, int H, int S, int head_dim,
                       bool is_causal, double scale) {
    switch (head_dim) {
        case 8:
            launch_kernel<scalar_t, 8>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return true;
        case 16:
            launch_kernel<scalar_t, 16>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return true;
        case 32:
            launch_kernel<scalar_t, 32>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return true;
        case 64:
            launch_kernel<scalar_t, 64>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------

namespace wm = nvcuda::wmma;

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

__device__ __forceinline__ void dev_from_float(float& d, float s)         { d = s; }
__device__ __forceinline__ void dev_from_float(__half& d, float s)        { d = __float2half(s); }
__device__ __forceinline__ void dev_from_float(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }

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
// One shape serves all three dtypes: WmmaShape is not parameterised on
// scalar_t, so a sweep that finds different winners per dtype is evidence for
// adding that parameter, not something these macros can express.
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
#define WMMA_N_64  32
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

template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(WmmaCfg<scalar_t, HEAD_DIM>::NTHREADS)
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
                                 bool out_bshd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Cfg = WmmaCfg<scalar_t, HEAD_DIM>;
    using frag_elem = typename FragTraits<scalar_t>::elem;
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
    constexpr bool IS_TF32 = std::is_same<scalar_t, float>::value;

    extern __shared__ __align__(16) char smem_raw[];
    float*    o_s = reinterpret_cast<float*>(smem_raw + Cfg::O_OFF);
    scalar_t* k_s = reinterpret_cast<scalar_t*>(smem_raw + Cfg::K_OFF);
    scalar_t* v_s = reinterpret_cast<scalar_t*>(smem_raw + Cfg::V_OFF);
    float*    s_s = reinterpret_cast<float*>(smem_raw + Cfg::S_OFF);
    float*    m_s = reinterpret_cast<float*>(smem_raw + Cfg::M_OFF);
    float*    l_s = reinterpret_cast<float*>(smem_raw + Cfg::L_OFF);
    float*    c_s = reinterpret_cast<float*>(smem_raw + Cfg::C_OFF);
    // P feeds the second GEMM as a matrix_a fragment, so it has to be in the
    // operand type. For fp32 that is the same type the scores were stored in,
    // so the softmax can overwrite the score tile in place.
    scalar_t* p_s = Cfg::P_ALIASES_S ? reinterpret_cast<scalar_t*>(s_s)
                                     : reinterpret_cast<scalar_t*>(smem_raw + Cfg::P_OFF);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int m_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int b      = blockIdx.z;

    const int64_t bh_off =
        static_cast<int64_t>(b) * qs0 + static_cast<int64_t>(h) * qs1;
    const int row_base = warp * RPW;                  // this warp stripe in the block
    const int q_base   = m_tile * BLOCK_M + row_base; // ... and in the sequence

    scalar_t zero_v;
    dev_from_float(zero_v, 0.0f);

    // --- stage Q, then hoist it into registers for the whole key loop -------
    {
        scalar_t* q_s = reinterpret_cast<scalar_t*>(smem_raw + Cfg::O_OFF);
        // Walk the padded width so columns past DIM are explicitly zeroed:
        // they feed GEMM1's contraction, where a stale value would corrupt the
        // score rather than contribute nothing.
        for (int idx = tid; idx < BLOCK_M * PDIM; idx += Cfg::NTHREADS) {
            const int r = idx / PDIM;
            const int c = idx - r * PDIM;
            const int gr = m_tile * BLOCK_M + r;
            q_s[r * KV_LD + c] =
                (gr < S && c < DIM)
                    ? q[bh_off + static_cast<int64_t>(gr) * qs2 + c]
                    : zero_v;
        }
        __syncthreads();
    }

    wm::fragment<wm::matrix_a, 16, 16, WK, frag_elem, wm::row_major> q_frag[PDIM / WK];
    {
        const scalar_t* q_s = reinterpret_cast<const scalar_t*>(smem_raw + Cfg::O_OFF);
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
            k_s[r * KV_LD + c] = inb ? k_base[g] : zero_v;
            v_s[r * KV_LD + c] = inb ? v_base[g] : zero_v;
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

template <typename scalar_t, int HEAD_DIM>
void launch_wmma_kernel(const torch::Tensor& q, const torch::Tensor& k,
                        const torch::Tensor& v, const bool* mask_ptr,
                        const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                        int B, int H, int S, bool is_causal, double scale) {
    using Cfg = WmmaCfg<scalar_t, HEAD_DIM>;
    const dim3 block(Cfg::NTHREADS);
    const dim3 grid((S + Cfg::BLOCK_M - 1) / Cfg::BLOCK_M, H, B);

    const bool out_bshd = (out.dim() == 3);

    fused_attention_wmma_kernel<scalar_t, HEAD_DIM>
        <<<grid, block, Cfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const scalar_t*>(q.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(k.const_data_ptr()),
            reinterpret_cast<const scalar_t*>(v.const_data_ptr()),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            B, H, S, is_causal, static_cast<float>(scale), out_bshd);
}

// Returns false when this (dtype, head_dim) pair has no tensor-core
// specialization, so the caller can fall back to the scalar kernel.
template <typename scalar_t, int HEAD_DIM>
bool maybe_launch_wmma(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const bool* mask_ptr,
                       const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                       int B, int H, int S, bool is_causal, double scale) {
    if constexpr (WmmaCfg<scalar_t, HEAD_DIM>::SUPPORTED) {
        launch_wmma_kernel<scalar_t, HEAD_DIM>(q, k, v, mask_ptr, ms, qs, out,
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

// ---------------------------------------------------------------------------
// Kernel selection.
//
// Every launcher below answers "can you handle this case?": true having
// launched, false having done nothing. That keeps the choice of kernel (here)
// separate from each kernel's coverage rules (inside it), so the caller is a
// plain list of preferences.
// ---------------------------------------------------------------------------

enum class Impl : int64_t {
    Auto     = 0,
    Scalar   = 1,
    Wmma     = 2,
    Tile     = 3,   // cuTile, fp32 operands -- CUDA cores,   ~1e-6
    TileBf16 = 4,   // cuTile, bf16 operands -- tensor cores, ~4e-3
    TileTf32 = 5,   // cuTile, tf32 operands -- tensor cores, ~1e-3
};

const char* impl_name(Impl impl) {
    switch (impl) {
        case Impl::Auto:     return "auto";
        case Impl::Scalar:   return "scalar";
        case Impl::Wmma:     return "wmma";
        case Impl::Tile:     return "tile";
        case Impl::TileBf16: return "tile-bf16";
        case Impl::TileTf32: return "tile-tf32";
    }
    return "?";
}

// One bundle so the launchers do not each take a dozen positional arguments.
struct AttnArgs {
    const torch::Tensor& q;
    const torch::Tensor& k;
    const torch::Tensor& v;
    const bool* mask_ptr;
    const int64_t* ms;
    // Batch, head and sequence strides of q/k/v, in elements. The head_dim
    // stride is not carried: it is always 1, because fused_attention_forward
    // makes a contiguous copy rather than pass a layout where it is not.
    // Contiguous inputs give {H*S*head_dim, S*head_dim, head_dim}, so a kernel
    // reading these needs no separate contiguous path.
    const int64_t* qs;
    torch::Tensor& out;
    int B;
    int H;
    int S;
    int head_dim;
    bool is_causal;
    double scale;
};

bool launch_scalar(const AttnArgs& a) {
    bool launched = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a.q.scalar_type(), "launch_scalar", [&] {
            launched = dispatch_head_dim<scalar_t>(
                a.q, a.k, a.v, a.mask_ptr, a.ms, a.qs, a.out,
                a.B, a.H, a.S, a.head_dim, a.is_causal, a.scale);
        });
    return launched;
}

bool launch_wmma(const AttnArgs& a) {
    // wmma fragments need SM 8.0+; below that the kernel body is compiled away
    // and would silently write zeros, so gate on the actual device.
    if (at::cuda::getCurrentDeviceProperties()->major < 8) {
        return false;
    }
    bool launched = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a.q.scalar_type(), "launch_wmma", [&] {
            launched = dispatch_wmma<scalar_t>(
                a.q, a.k, a.v, a.mask_ptr, a.ms, a.qs, a.out,
                a.B, a.H, a.S, a.head_dim, a.is_causal, a.scale);
        });
    return launched;
}

bool launch_tile(const AttnArgs& a, tile_attn::MathMode math) {
    // Unlike the other two, these are hard requirements rather than coverage
    // gaps: a caller asking for the tile kernel on a build or dtype that cannot
    // have it wants to hear so, not to be quietly given a different kernel.
    TORCH_CHECK(tile_attn::available(),
                "fused_attention_forward: impl=3 (tile) requested but this build "
                "has no tile support. It needs CUDA 13.3+ with -std=c++20 "
                "-enable-tile; see kernel_ext.py.");
    TORCH_CHECK(a.q.scalar_type() == torch::kFloat32,
                "fused_attention_forward: the tile kernels take float32 in and "
                "out -- the math mode narrows only the GEMM operands. Got ",
                a.q.scalar_type());
    TORCH_CHECK(tile_attn::supports(math),
                "fused_attention_forward: this build has no tile kernel for that "
                "math mode. tf32 needs a toolkit shipping <cuda_tf32.h> (CUDA "
                "13.3+); see csrc/tile_attention.h.");
    // Scratch for split-KV's per-split partials, zero on most shapes. From
    // torch's caching allocator rather than cudaMalloc: it is stream-ordered
    // (freed to the same stream the kernel runs on when `ws` dies) and it draws
    // on the pool the benchmark is already accounted against.
    const size_t ws_bytes = tile_attn::workspace_bytes(
        a.B, a.H, a.S, a.head_dim, a.is_causal, math);
    at::Tensor ws;
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        ws = at::empty({static_cast<int64_t>(ws_bytes)},
                       a.q.options().dtype(at::kByte));
        ws_ptr = ws.data_ptr();
    }
    return tile_attn::launch(
        a.q.const_data_ptr<float>(), a.k.const_data_ptr<float>(),
        a.v.const_data_ptr<float>(), a.mask_ptr, a.ms, a.qs,
        a.out.data_ptr<float>(), ws_ptr, ws_bytes,
        a.B, a.H, a.S, a.head_dim, a.is_causal,
        static_cast<float>(a.scale), math, at::cuda::getCurrentCUDAStream());
}

// Runs the first kernel that covers this case, honouring what the caller asked
// for. Auto is the only mode that tries a second kernel; forcing an impl means
// that kernel or nothing. What happens when nothing covers the case is decided
// by the caller, not here.
bool run_kernel(Impl impl, const AttnArgs& a) {
    switch (impl) {
        case Impl::Scalar:   return launch_scalar(a);
        case Impl::Wmma:     return launch_wmma(a);
        case Impl::Tile:     return launch_tile(a, tile_attn::MathMode::Fp32);
        case Impl::TileBf16: return launch_tile(a, tile_attn::MathMode::Bf16);
        case Impl::TileTf32: return launch_tile(a, tile_attn::MathMode::Tf32);
        // Tile is deliberately absent here: it covers only float32 and is a
        // separate programming model whose performance the caller should opt
        // into deliberately rather than inherit.
        case Impl::Auto:     return launch_wmma(a) || launch_scalar(a);
    }
    return false;
}

// [B,H,S,D] -> [B,S,H*D]. reshape() cannot view across the transpose, so this
// is a real repack -- which is exactly the cost the layout-1 kernels avoid.
torch::Tensor to_bshd(const torch::Tensor& t) {
    return t.transpose(1, 2).reshape({t.size(0), t.size(2), t.size(1) * t.size(3)});
}

// Mirrors BaselineSelfAttention.forward exactly. Used for shapes/dtypes no
// kernel specializes.
torch::Tensor attention_aten(const torch::Tensor& q, const torch::Tensor& k,
                             const torch::Tensor& v,
                             const c10::optional<torch::Tensor>& attn_mask,
                             bool is_causal, double scale) {
    const int64_t S = q.size(2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

    if (is_causal) {
        auto blocked = torch::ones({S, S},
                                   torch::TensorOptions().dtype(torch::kBool).device(q.device()))
                           .triu(/*diagonal=*/1);
        scores = scores.masked_fill(blocked, -std::numeric_limits<float>::infinity());
    }
    if (attn_mask.has_value()) {
        scores = scores.masked_fill(attn_mask.value().logical_not(),
                                    -std::numeric_limits<float>::infinity());
    }

    auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(q.scalar_type());
    return torch::matmul(probs, v);
}


// ---------------------------------------------------------------------------
// Fused residual add + LayerNorm.
//
//   x_new  = x + sub                     the residual stream, still needed by
//                                        the next sublayer's skip connection
//   normed = LayerNorm(x_new) * w + b    what the next sublayer consumes
//
// Unfused this is two kernels and five passes over the tensor. Keeping the row
// on chip cuts that to four -- x and sub in, x_new and normed out -- and
// removes a launch, which at these sizes costs about as much as the bandwidth.
//
// The row lives in shared memory rather than registers so d_model can stay a
// runtime value; at 512 floats that is 2 KB per block. Reading it back from
// shared is also what makes the corrected two-pass statistics free -- see the
// note on precision inside the kernel.
// ---------------------------------------------------------------------------

// Sum of v across the whole block, broadcast back to every thread. scratch
// needs one float per warp. It is clobbered, so callers must not keep live data
// there, and must __syncthreads() between two calls that share it.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
        scratch[warp] = v;
    }
    __syncthreads();

    // nwarps is at most 32 for any legal block size, so warp 0 alone finishes.
    if (warp == 0) {
        v = (lane < nwarps) ? scratch[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            scratch[0] = v;
        }
    }
    __syncthreads();
    return scratch[0];
}

template <typename scalar_t>
__global__ void fused_add_layernorm_kernel(const scalar_t* __restrict__ x,
                                           const scalar_t* __restrict__ sub,
                                           const scalar_t* __restrict__ w,
                                           const scalar_t* __restrict__ beta,
                                           scalar_t* __restrict__ x_new,
                                           scalar_t* __restrict__ normed,
                                           int D, float eps) {
    extern __shared__ __align__(16) char smem_raw[];
    float* s_row = reinterpret_cast<float*>(smem_raw);   // D floats
    float* scratch = s_row + D;                          // one float per warp

    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * D;

    // Statistics accumulate in float even for half inputs: F.layer_norm does
    // its mean and variance in float, and this has to match that.
    //
    // The sum is rounded through scalar_t before being kept for the normalise
    // step. The reference adds x + sub in the tensor dtype and then normalises
    // that rounded value, so keeping the wider intermediate here would be more
    // accurate than the reference rather than equal to it -- and equal is what
    // the harness measures. For float the round-trip is the identity.
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        const scalar_t v = static_cast<scalar_t>(
            static_cast<float>(x[base + d]) + static_cast<float>(sub[base + d]));
        const float vf = static_cast<float>(v);
        s_row[d] = vf;
        x_new[base + d] = v;
        local_sum += vf;
    }
    __syncthreads();

    // Corrected two-pass statistics. A plain sum-then-mean is accurate only
    // while the row's mean is small relative to its spread: once activations
    // grow (a deep residual stream, or --input-scale up) the sum loses low bits
    // and every (x - mean) inherits that error. Against F.layer_norm, which
    // uses Welford, a naive mean drifted to 1.5e-3 at mean 1e4 -- past atol on
    // its own.
    //
    // So the first mean is an estimate only, and the second pass sums the
    // residuals (x - mean_est), which are O(spread) however large the mean is.
    // One extra reduction over shared memory, no global traffic.
    const float mean_est = block_reduce_sum(local_sum, scratch) / static_cast<float>(D);

    float local_c = 0.0f;
    float local_cc = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        const float c = s_row[d] - mean_est;
        local_c += c;
        local_cc += c * c;
    }
    __syncthreads();

    const float delta = block_reduce_sum(local_c, scratch) / static_cast<float>(D);
    __syncthreads();  // scratch is reused by the next reduction
    const float mean_sq = block_reduce_sum(local_cc, scratch) / static_cast<float>(D);

    const float mean = mean_est + delta;
    // delta is the small correction, so subtracting its square cannot cancel
    // the way E[x^2] - mean^2 does. Clamp anyway: rounding can leave a
    // zero-variance row very slightly negative, and rsqrtf would return NaN.
    const float var = fmaxf(mean_sq - delta * delta, 0.0f);
    const float rstd = rsqrtf(var + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        const float v = (s_row[d] - mean) * rstd;
        normed[base + d] = static_cast<scalar_t>(
            v * static_cast<float>(w[d]) + static_cast<float>(beta[d]));
    }
}


// ---------------------------------------------------------------------------
// GEMM with a fused bias + GELU epilogue:   C = GELU(A @ W^T + bias)
//
//   A     [M, K]  row-major. The layer input, M = batch * seq_len.
//   W     [N, K]  row-major -- nn.Linear's own weight layout, which makes W^T
//                 [K, N] column-major with ldm = K, so matrix_b reads it with
//                 no transpose pass. The trick the attention kernel uses for
//                 K^T.
//   bias  [N]
//   C     [M, N]  row-major.
//
// 128x128 block tile, 8 warps in a 4x2 grid, each owning a 32x64 quadrant as
// 2x4 wmma tiles. The tile size is forced by DRAM traffic, not chosen.
//
// DEAD CODE, DELIBERATELY: this kernel is correct, is exported as linear_gelu,
// and loses to cuBLAS at every shape the model uses (0.88x on the FFN's first
// GEMM), so nothing calls it. The fused activation is worth what it was
// predicted to be worth; the GEMM underneath is not competitive. It is kept as
// the evidence for that conclusion -- csrc/TUNING.md has the numbers and what
// closing the gap would take. Delete both if the file space matters more.
// ---------------------------------------------------------------------------

// Exact GELU, matching F.gelu(approximate="none") rather than the tanh form.
// The harness's accuracy gate is tight enough that the tanh approximation
// fails atol on its own, so this has to be the erf version.
__device__ __forceinline__ float gelu_exact(float v) {
    // 1/sqrt(2), to more digits than float carries.
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

template <int BM, int BN, int BK>
__global__ __launch_bounds__(256)
void gemm_bias_gelu_kernel(const float* __restrict__ A,
                           const float* __restrict__ W,
                           const float* __restrict__ bias,
                           float* __restrict__ C,
                           int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;   // 256
    constexpr int WM = BM / WARPS_M;                   // 32 rows per warp
    constexpr int WN = BN / WARPS_N;                   // 64 cols per warp
    constexpr int MT = WM / 16;                        // 2 wmma tiles down
    constexpr int NT = WN / 16;                        // 4 wmma tiles across
    constexpr int WK = 8;                              // tf32 fragment K
    constexpr int VEC = 4;                             // float4 staging

    // Padded leading dimension, for the same reason as in the attention
    // kernel: a fragment load walks a column of the tile, and an unpadded
    // stride of 32 floats puts all 16 rows in one bank. +4 rotates them onto 8
    // distinct bank groups (2-way instead of 16-way), and keeps the stride a
    // multiple of 4 floats so the float4 staging stores stay 16-byte aligned.
    constexpr int LD = BK + 4;

    extern __shared__ __align__(16) char smem_raw[];
    float* A_s = reinterpret_cast<float*>(smem_raw);
    float* W_s = A_s + BM * LD;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    wm::fragment<wm::accumulator, 16, 16, WK, float> acc[MT][NT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            wm::fill_fragment(acc[mt][nt], 0.0f);
        }
    }

    constexpr int A_SLOTS = BM * (BK / VEC);
    constexpr int W_SLOTS = BN * (BK / VEC);
    constexpr int COLS4 = BK / VEC;
    constexpr int A_PER_THREAD = A_SLOTS / NTHREADS;   // 4 float4
    constexpr int W_PER_THREAD = W_SLOTS / NTHREADS;   // 4 float4
    static_assert(A_SLOTS % NTHREADS == 0 && W_SLOTS % NTHREADS == 0,
                  "staging must divide evenly over the block for the prefetch");

    // Software pipelining: the next tile is fetched into registers *before* the
    // current tile's MMAs are issued, so the loads are in flight while the
    // tensor cores run. Without it the MMA units sit idle for the whole
    // global-memory latency of every K tile (9.6 TFLOPS, and the two
    // __syncthreads per iteration bracket a stall rather than work).
    //
    // Registers rather than a second shared buffer: double-buffering the tiles
    // would want 73.7 KB against the 48 KB budget, where 8 float4 per thread
    // costs 32 registers, which the accumulators can afford.
    float4 pa[A_PER_THREAD];
    float4 pw[W_PER_THREAD];

    // Rows past M/N and columns past K are zero-filled rather than skipped:
    // they feed the contraction, where a stale value would corrupt the dot
    // product instead of contributing nothing.
    #define FETCH_TILES(kbase)                                                       \
        {                                                                            \
            _Pragma("unroll")                                                        \
            for (int j = 0; j < A_PER_THREAD; ++j) {                                 \
                const int i = tid + j * NTHREADS;                                    \
                const int r = i / COLS4;                                             \
                const int c = (i - r * COLS4) * VEC;                                 \
                pa[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                         \
                if (m0 + r < M && (kbase) + c < K) {                                 \
                    pa[j] = *reinterpret_cast<const float4*>(                        \
                        A + static_cast<int64_t>(m0 + r) * K + (kbase) + c);         \
                }                                                                    \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int j = 0; j < W_PER_THREAD; ++j) {                                 \
                const int i = tid + j * NTHREADS;                                    \
                const int r = i / COLS4;                                             \
                const int c = (i - r * COLS4) * VEC;                                 \
                pw[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                         \
                if (n0 + r < N && (kbase) + c < K) {                                 \
                    pw[j] = *reinterpret_cast<const float4*>(                        \
                        W + static_cast<int64_t>(n0 + r) * K + (kbase) + c);         \
                }                                                                    \
            }                                                                        \
        }

    #define STORE_TILES()                                                            \
        {                                                                            \
            _Pragma("unroll")                                                        \
            for (int j = 0; j < A_PER_THREAD; ++j) {                                 \
                const int i = tid + j * NTHREADS;                                    \
                const int r = i / COLS4;                                             \
                const int c = (i - r * COLS4) * VEC;                                 \
                *reinterpret_cast<float4*>(&A_s[r * LD + c]) = pa[j];                \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int j = 0; j < W_PER_THREAD; ++j) {                                 \
                const int i = tid + j * NTHREADS;                                    \
                const int r = i / COLS4;                                             \
                const int c = (i - r * COLS4) * VEC;                                 \
                *reinterpret_cast<float4*>(&W_s[r * LD + c]) = pw[j];                \
            }                                                                        \
        }

    FETCH_TILES(0);
    STORE_TILES();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int knext = k0 + BK;
        // Issued before the MMAs below, so the memory latency overlaps them.
        if (knext < K) {
            FETCH_TILES(knext);
        }

        #pragma unroll
        for (int ks = 0; ks < BK / WK; ++ks) {
            wm::fragment<wm::matrix_a, 16, 16, WK, wm::precision::tf32, wm::row_major> af[MT];
            wm::fragment<wm::matrix_b, 16, 16, WK, wm::precision::tf32, wm::col_major> bf[NT];

            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                wm::load_matrix_sync(
                    af[mt], A_s + static_cast<size_t>(wm * WM + mt * 16) * LD + ks * WK, LD);
                #pragma unroll
                for (int t = 0; t < af[mt].num_elements; ++t) {
                    af[mt].x[t] = wm::__float_to_tf32(af[mt].x[t]);
                }
            }
            // W_s holds [n][k]; col_major with ldm = LD presents element (k, n)
            // at W_s[n * LD + k], which is W^T without moving anything.
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                wm::load_matrix_sync(
                    bf[nt], W_s + static_cast<size_t>(wn * WN + nt * 16) * LD + ks * WK, LD);
                #pragma unroll
                for (int t = 0; t < bf[nt].num_elements; ++t) {
                    bf[nt].x[t] = wm::__float_to_tf32(bf[nt].x[t]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    wm::mma_sync(acc[mt][nt], af[mt], bf[nt], acc[mt][nt]);
                }
            }
        }

        if (knext < K) {
            __syncthreads();   // every warp is done reading this tile
            STORE_TILES();
            __syncthreads();   // ... and the next one is fully staged
        }
    }

    #undef FETCH_TILES
    #undef STORE_TILES

    // --- epilogue: bias, GELU, store ---------------------------------------
    //
    // One 16x16 tile at a time through a small shared scratch, rather than
    // applying the activation to accumulator elements directly. The
    // element-to-row mapping inside an accumulator fragment is
    // architecture-defined, and the attention kernel had to probe it at
    // runtime to work around that. Here nothing needs the row *during* the
    // loop -- the bias depends only on the column and GELU is elementwise --
    // so storing the tile first and reading it back with known indexing avoids
    // the whole problem for the cost of one 1 KB round trip per warp per tile.
    // The K loop is over, so the staging tiles are dead and host the scratch.
    __syncthreads();
    float* scratch = A_s + warp * 256;

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            wm::store_matrix_sync(scratch, acc[mt][nt], 16, wm::mem_row_major);
            __syncwarp();

            const int tile_m = m0 + wm * WM + mt * 16;
            const int tile_n = n0 + wn * WN + nt * 16;
            for (int idx = lane; idx < 256; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                const int gr = tile_m + r;
                const int gc = tile_n + c;
                if (gr < M && gc < N) {
                    C[static_cast<int64_t>(gr) * N + gc] =
                        gelu_exact(scratch[idx] + bias[gc]);
                }
            }
            __syncwarp();
        }
    }
#endif
}

}  // namespace


torch::Tensor smoke_test_identity(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "smoke_test_identity: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "smoke_test_identity: only float32 is wired up for the smoke test");

    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);

    const int64_t n = x_contig.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;

    identity_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_contig.data_ptr<float>(), out.data_ptr<float>(), n);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "smoke_test_identity: kernel launch failed: ", cudaGetErrorString(err));
    return out;
}


// Linear followed by exact GELU, in one kernel:  GELU(x @ weight^T + bias)
//
//   x       [..., K]  contiguous; every leading dimension is flattened into M
//   weight  [N, K]    nn.Linear layout
//   bias    [N]
//
// returns   [..., N]
//
// Declines (returns an undefined tensor) rather than raising when the shape or
// dtype is outside what the kernel covers, so the caller can fall back to
// F.linear + F.gelu instead of failing. Requirements: float32, SM 8.0+, and
// K divisible by 4 for the float4 staging path.
torch::Tensor linear_gelu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "linear_gelu: all inputs must be CUDA tensors");
    TORCH_CHECK(weight.dim() == 2, "linear_gelu: weight must be 2-D [N, K]");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == weight.size(1),
                "linear_gelu: x's last dimension must match weight's, got ",
                x.sizes(), " and ", weight.sizes());
    TORCH_CHECK(bias.numel() == weight.size(0),
                "linear_gelu: bias must have ", weight.size(0), " elements");

    const int64_t K = weight.size(1);
    const int64_t N = weight.size(0);

    const bool covered =
        x.scalar_type() == torch::kFloat32 &&
        weight.scalar_type() == torch::kFloat32 &&
        bias.scalar_type() == torch::kFloat32 &&
        (K % 4) == 0 &&
        at::cuda::getCurrentDeviceProperties()->major >= 8;
    if (!covered) {
        return torch::Tensor();   // caller falls back
    }

    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto bc = bias.contiguous();

    const int64_t M = xc.numel() / K;

    auto out_sizes = xc.sizes().vec();
    out_sizes.back() = N;
    auto out = torch::empty(out_sizes, xc.options());

    constexpr int BM = 128, BN = 128, BK = 32;
    constexpr int LD = BK + 4;
    const size_t smem = sizeof(float) * static_cast<size_t>(BM + BN) * LD;
    static_assert((BM + BN) * (BK + 4) * sizeof(float) <= 48 * 1024,
                  "staging tiles exceed the 48 KB shared-memory budget");

    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((M + BM - 1) / BM));

    gemm_bias_gelu_kernel<BM, BN, BK>
        <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
            xc.const_data_ptr<float>(), wc.const_data_ptr<float>(),
            bc.const_data_ptr<float>(), out.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "linear_gelu: kernel launch failed: ", cudaGetErrorString(err));
    return out;
}


// Fused residual add + LayerNorm over the last dimension.
//
//   x, sub        : identical shapes, CUDA. Normalisation is over the last
//                   dimension, whose size must match weight and bias.
//   weight, bias  : [D], the LayerNorm affine parameters.
//   eps           : as nn.LayerNorm.
//
// returns         : {x + sub, LayerNorm(x + sub) * weight + bias}
//
// The un-normalised sum is returned because the caller still needs it for its
// own skip connection. That is what makes the pair fusable without changing
// any of the model arithmetic.
std::vector<torch::Tensor> fused_add_layernorm(torch::Tensor x,
                                               torch::Tensor sub,
                                               torch::Tensor weight,
                                               torch::Tensor bias,
                                               double eps) {
    TORCH_CHECK(x.is_cuda() && sub.is_cuda(), "fused_add_layernorm: x/sub must be CUDA");
    TORCH_CHECK(x.sizes() == sub.sizes(),
                "fused_add_layernorm: x and sub must have identical shapes, got ",
                x.sizes(), " and ", sub.sizes());
    TORCH_CHECK(x.scalar_type() == sub.scalar_type(),
                "fused_add_layernorm: x and sub must share a dtype");
    TORCH_CHECK(x.dim() >= 1, "fused_add_layernorm: x needs at least one dimension");

    const int64_t D = x.size(-1);
    TORCH_CHECK(weight.numel() == D && bias.numel() == D,
                "fused_add_layernorm: weight/bias must have ", D, " elements");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() &&
                    bias.scalar_type() == x.scalar_type(),
                "fused_add_layernorm: weight/bias must share x's dtype");

    auto xc = x.contiguous();
    auto sc = sub.contiguous();
    auto wc = weight.contiguous();
    auto bc = bias.contiguous();

    auto x_new = torch::empty_like(xc);
    auto normed = torch::empty_like(xc);

    const int64_t rows = xc.numel() / D;
    // 256 threads keeps the block reduction short while still coalescing the
    // loads; each thread strides by blockDim, so any D spreads over it.
    const int threads = 256;
    const int nwarps = (threads + 31) / 32;
    const size_t smem = sizeof(float) * static_cast<size_t>(D + nwarps);

    // The row has to fit in shared memory. 48 KB is the limit that needs no
    // opt-in carveout, which covers d_model up to 12280 -- far past anything
    // the harness produces, but check rather than corrupt memory if it is not.
    TORCH_CHECK(smem <= 48 * 1024,
                "fused_add_layernorm: last dimension ", D,
                " needs ", smem, " bytes of shared memory, over the 48 KB budget");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        xc.scalar_type(), "fused_add_layernorm", [&] {
            fused_add_layernorm_kernel<scalar_t>
                <<<static_cast<int>(rows), threads, smem,
                   at::cuda::getCurrentCUDAStream()>>>(
                    xc.const_data_ptr<scalar_t>(), sc.const_data_ptr<scalar_t>(),
                    wc.const_data_ptr<scalar_t>(), bc.const_data_ptr<scalar_t>(),
                    x_new.data_ptr<scalar_t>(), normed.data_ptr<scalar_t>(),
                    static_cast<int>(D), static_cast<float>(eps));
        });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_add_layernorm: kernel launch failed: ", cudaGetErrorString(err));
    return {x_new, normed};
}


// Fused scaled-dot-product attention.
//
//   q, k, v          : [B, H, S, head_dim], CUDA
//   attn_mask        : optional bool, broadcastable to [B, H, S, S].
//                      SDPA convention -- true means ALLOWED to attend.
//   is_causal        : lower-triangular masking. Never combined with attn_mask;
//                      the caller folds causal into attn_mask when both apply.
//   scale            : score multiplier, normally 1/sqrt(head_dim).
//   out_layout       : 0 -> [B, H, S, head_dim], the natural per-head layout
//                      1 -> [B, S, H*head_dim], ready for out_proj
//
// returns            : the requested layout, always. The wmma and scalar
//                      kernels write layout 1 directly; the tile kernels and
//                      the ATen fallback produce layout 0 and are converted
//                      here, so the caller never has to ask which it got.
torch::Tensor fused_attention_forward(torch::Tensor q,
                                      torch::Tensor k,
                                      torch::Tensor v,
                                      c10::optional<torch::Tensor> attn_mask,
                                      bool is_causal,
                                      double scale,
                                      int64_t impl,
                                      int64_t out_layout) {
    TORCH_CHECK(out_layout == 0 || out_layout == 1,
                "fused_attention_forward: out_layout must be 0 ([B,H,S,head_dim]) "
                "or 1 ([B,S,H*head_dim])");
    TORCH_CHECK(impl >= 0 && impl <= 5,
                "fused_attention_forward: impl must be 0 (auto), 1 (scalar), "
                "2 (wmma), 3 (tile), 4 (tile-bf16) or 5 (tile-tf32)");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "fused_attention_forward: q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "fused_attention_forward: q/k/v must be 4-D [B, H, S, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "fused_attention_forward: q/k/v must have identical shapes");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "fused_attention_forward: q/k/v must share a dtype");
    TORCH_CHECK(!(is_causal && attn_mask.has_value()),
                "fused_attention_forward: pass either is_causal or attn_mask, not both");

    const Impl mode = static_cast<Impl>(impl);
    const bool tile_mode =
        (mode == Impl::Tile || mode == Impl::TileBf16 || mode == Impl::TileTf32);

    // Every kernel addresses q/k/v through explicit (batch, head, sequence)
    // strides and assumes only that head_dim is stride-1. That is exactly what
    // MySelfAttention's fused QKV projection produces: one [B, S, 3*d_model]
    // tensor viewed and permuted into three [B,H,S,head_dim] views whose last
    // axis is still contiguous, with rows spaced 3*d_model apart instead of
    // head_dim. Taking those views as they are removes three clone kernels per
    // layer -- the copy was only ever paying for a row pitch.
    //
    // What still copies is a layout this ABI cannot describe: head_dim not
    // stride-1, or q/k/v that are not three slices of one tensor. Those fall
    // back rather than widening the ABI -- a clone costs microseconds, a wrong
    // address costs a wrong answer.
    const bool pass_strided =
        q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 &&
        q.strides().equals(k.strides()) && q.strides().equals(v.strides());

    auto qc = pass_strided ? q : q.contiguous();
    auto kc = pass_strided ? k : k.contiguous();
    auto vc = pass_strided ? v : v.contiguous();
    const int64_t qs[3] = {qc.stride(0), qc.stride(1), qc.stride(2)};

    const int B = static_cast<int>(qc.size(0));
    const int H = static_cast<int>(qc.size(1));
    const int S = static_cast<int>(qc.size(2));
    const int head_dim = static_cast<int>(qc.size(3));

    // expand() gives stride-0 dims instead of copying, so a [B,1,1,S] mask and
    // a [B,1,S,S] mask are both just a stride pattern to the kernel.
    const bool* mask_ptr = nullptr;
    int64_t ms[4] = {0, 0, 0, 0};
    torch::Tensor mask_expanded;
    if (attn_mask.has_value()) {
        auto m = attn_mask.value();
        TORCH_CHECK(m.scalar_type() == torch::kBool,
                    "fused_attention_forward: attn_mask must be a bool tensor");
        mask_expanded = m.contiguous().expand({B, H, S, S});
        mask_ptr = mask_expanded.data_ptr<bool>();
        for (int d = 0; d < 4; ++d) {
            ms[d] = mask_expanded.stride(d);
        }
    }

    // The tile kernels index their output as [B,H,S,head_dim] internally and
    // have no layout switch, so they always get a 4-D buffer and are converted
    // afterwards. Everything else writes the requested layout directly, and the
    // launchers read which one off out.dim().
    const bool want_bshd = (out_layout == 1);
    const bool kernel_writes_bshd = want_bshd && !tile_mode;

    // Spelled out rather than empty_like(qc): qc is now often a non-contiguous
    // view, and empty_like preserves its strides. Every kernel writes its
    // output as a packed [B,H,S,head_dim] -- out_base() and the tile epilogue
    // both derive the row pitch from head_dim, not from a stride argument -- so
    // inheriting q's pitch here would scatter the result into the wrong rows.
    auto out = kernel_writes_bshd
                   ? torch::empty({static_cast<int64_t>(B), static_cast<int64_t>(S),
                                   static_cast<int64_t>(H) * head_dim},
                                  qc.options())
                   : torch::empty({static_cast<int64_t>(B), static_cast<int64_t>(H),
                                   static_cast<int64_t>(S),
                                   static_cast<int64_t>(head_dim)},
                                  qc.options());

    const AttnArgs args{qc, kc, vc, mask_ptr, ms, qs, out,
                        B, H, S, head_dim, is_causal, scale};

    if (!run_kernel(mode, args)) {
        // Nothing covered this case, so ATen finishes the job -- it takes any
        // head_dim. wmma and tile decline loudly instead: asking for one of
        // them specifically and quietly getting ATen would let a benchmark
        // time one kernel and report it as another.
        //
        // scalar does *not* raise here, only because it never has. That is an
        // inconsistency rather than a decision -- it can time ATen and label it
        // "scalar" -- but changing it is a behaviour change, not a cleanup.
        TORCH_CHECK(mode != Impl::Wmma && mode != Impl::Tile &&
                        mode != Impl::TileBf16 && mode != Impl::TileTf32,
                    "fused_attention_forward: impl=", impl, " (", impl_name(mode),
                    ") does not cover dtype=", qc.scalar_type(),
                    ", head_dim=", head_dim, " on compute capability ",
                    at::cuda::getCurrentDeviceProperties()->major, ".",
                    at::cuda::getCurrentDeviceProperties()->minor,
                    ". wmma needs SM 8.0+ and head_dim in {8,16,32,64,128}; "
                    "the tile kernels need float32 and head_dim in "
                    "{8,16,32,64}.");
        auto aten_out = attention_aten(qc, kc, vc, attn_mask, is_causal, scale);
        return want_bshd ? to_bshd(aten_out) : aten_out;
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_attention_forward: kernel launch failed: ", cudaGetErrorString(err));

    // Only reached by the tile kernels, which could not write layout 1 natively.
    return (want_bshd && !kernel_writes_bshd) ? to_bshd(out) : out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_test_identity", &smoke_test_identity,
          "Copies input to output via a trivial CUDA kernel (build smoke test)");
    m.def("linear_gelu", &linear_gelu,
          "GELU(x @ weight^T + bias) in one kernel; undefined tensor if unsupported",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"));
    m.def("fused_add_layernorm", &fused_add_layernorm,
          "Fused residual add + LayerNorm; returns {x + sub, norm(x + sub)}",
          pybind11::arg("x"),
          pybind11::arg("sub"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps") = 1e-5);
    m.def("tile_workspace_bytes",
          [](int B, int H, int S, int head_dim, bool is_causal, int impl) {
              // impl codes match fused_attention_forward's: 3 tile-fp32,
              // 4 tile-bf16, 5 tile-tf32. Non-zero means the launcher intends
              // to split the key range for this shape, so a test can assert
              // that it is really exercising the split path rather than
              // silently passing on the single-pass one.
              tile_attn::MathMode math = tile_attn::MathMode::Fp32;
              if (impl == 4) math = tile_attn::MathMode::Bf16;
              if (impl == 5) math = tile_attn::MathMode::Tf32;
              return static_cast<int64_t>(tile_attn::workspace_bytes(
                  B, H, S, head_dim, is_causal, math));
          },
          "Scratch bytes the tile kernel would take for this shape; 0 when it "
          "will not split the key range",
          pybind11::arg("B"), pybind11::arg("H"), pybind11::arg("S"),
          pybind11::arg("head_dim"), pybind11::arg("is_causal"),
          pybind11::arg("impl"));
    m.def("tile_set_split_kv", &tile_attn::set_split_kv,
          "Enable/disable the tile kernel's split-KV (Flash-Decoding) path. "
          "Runtime-settable so both paths can be timed in one process.",
          pybind11::arg("enabled"));
    m.def("tile_split_kv_enabled", &tile_attn::split_kv_enabled,
          "Whether the tile kernel's split-KV path is currently enabled");
    m.def("fused_attention_forward", &fused_attention_forward,
          "Fused scaled-dot-product attention",
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("attn_mask") = c10::nullopt,
          pybind11::arg("is_causal") = false,
          pybind11::arg("scale"),
          pybind11::arg("impl") = 0,
          pybind11::arg("out_layout") = 0);
}
