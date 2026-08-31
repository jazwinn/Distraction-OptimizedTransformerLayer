// GEMM with a fused bias epilogue, with or without GELU:
//
//     C = A @ W^T + bias            (linear_bias)
//     C = GELU(A @ W^T + bias)      (linear_gelu)
//
// One kernel templated on the activation rather than two: everything before
// the epilogue -- staging, the software pipeline, the fragment loop -- is the
// same GEMM, and the only difference is one erff in the store loop. The
// bias-only form is what the QKV projection and the two output projections
// use, which is where most of this model's remaining cuBLAS time sat.
//
//   A     [M, K]  row-major. The layer input, M = batch * seq_len.
//   W     [N, K]  row-major -- nn.Linear's own weight layout, which makes W^T
//                 [K, N] column-major with ldm = K, so matrix_b reads it with
//                 no transpose pass. The trick the attention kernel uses for
//                 K^T.
//   bias  [N]
//   C     [M, N]  row-major.
//
// 8 warps in a 4x2 grid over a BMxBN block tile, each warp owning a
// (BM/4)x(BN/2) quadrant as wmma tiles. BM must be a multiple of 64 and BN a
// multiple of 32, so that every warp gets at least one 16x16 tile in each
// direction.
//
// The tile is CHOSEN PER SHAPE, not fixed -- see pick_gemm_tile(). A DRAM-traffic
// argument says bigger is always better (a BMxBN tile does BM*BN*K MACs per
// (BM+BN)*K floats loaded, so 128x128 sustains twice the MAC/float of 64x64),
// and that argument is wrong whenever the operands fit in L2, which they do at
// every d_model this model uses. What actually binds at those shapes is the
// grid: at M=128, N=256 a 128x128 tile is 2 blocks on a 46-SM card. Measured
// under graph replay, 128x128 runs 0.198x of cuBLAS+GELU there.

#pragma once

#include "kernel_common.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

// Fragment precision for the GEMM below. TF32 and FP16 carry the SAME 10-bit
// mantissa, so they produce the same error against an fp64 reference -- measured
// identical to three significant figures at the attention op, the FFN GEMM and
// the whole six-layer model. What differs is throughput: on this card fp16
// tensor cores measure 2.0x-2.25x tf32 (39.7 vs 17.7 TFLOPS at N=2048), and an
// fp16 fragment contracts 16 elements of K against tf32's 8, so it also halves
// the mma instruction count. bf16 is NOT here: 8 mantissa bits put it at
// 425%-622% of the harness's 2e-3 budget, tens of thousands of failing elements.
//
// The conversion also moves. tf32 fragments are loaded as float and narrowed
// element-by-element *after* every load_matrix_sync, so A is converted MT times
// per k-step and W NT times. fp16 narrows once, on the way into shared memory.
//
// `store_t` is what the staging tiles hold; `frag_t` is what the fragment
// declares. They differ for tf32, where wmma stores tf32 as float.
struct Tf32Math {
    using store_t = float;
    using frag_t  = wm::precision::tf32;
    static constexpr int K = 8;
    static constexpr int LD_PAD = 4;    // wmma wants ldm % 4 == 0 for float
    static constexpr bool convert_frag = true;
};

struct Fp16Math {
    using store_t = __half;
    using frag_t  = __half;
    static constexpr int K = 16;
    static constexpr int LD_PAD = 8;    // ... and ldm % 8 == 0 for 16-bit
    static constexpr bool convert_frag = false;
};

// Shared-memory plan, shared by the kernel and its launcher so the two cannot
// disagree about how many bytes were requested.
//
// The epilogue reuses the (dead) staging tiles as scratch, one 16x16 fp32 tile
// per warp. In fp32 that always fit; in fp16 the staging area halves and the
// smallest tile no longer covers it -- 64x32 stages 7.5 KB against 8 KB of
// scratch -- so the request is the larger of the two rather than the staging
// size alone.
template <typename Math, int BM, int BN, int BK>
struct GemmSmem {
    static constexpr int LD = BK + Math::LD_PAD;
    static constexpr size_t staging =
        sizeof(typename Math::store_t) * static_cast<size_t>(BM + BN) * LD;
    // 8 warps, each materialising a 16 x SEG strip of its own row block.
    // SEG is capped at 32 columns so this stays at 16 KB for every block tile:
    // at 8 warps a block is capped to 6 residents by the SM's 48-warp limit
    // anyway, and 100 KB / 16 KB is exactly 6 -- so the wider strip that makes
    // the stores 128 bytes wide costs no occupancy. For BN 128 the staging area
    // is 20 KB and already dominates, so `bytes` does not move there at all.
    static constexpr int SEG_WANT = (BN / 2 < 32) ? (BN / 2) : 32;
    static constexpr size_t scratch_want = sizeof(float) * 8 * 16 * SEG_WANT;
    // The wide strip is only taken when it fits inside the staging area it
    // aliases. Growing `bytes` is NOT free even when resident blocks are
    // unchanged: at 8 warps a block is capped to 6 residents by the SM's
    // 48-warp limit, but 6 x 16 KB = 98 KB forces the driver onto the 100 KB
    // shared carveout instead of 64 KB, which leaves ~28 KB of L1 instead of
    // ~64 KB. Measured cost of getting that wrong: shape 8's QKV GEMM (K=1024,
    // N=3072, 64x64 tile) went 11.5 ms against ~8 ms expected and the shape
    // regressed 15.9%.
    static constexpr int SEG = (scratch_want <= staging) ? SEG_WANT : 16;
    static constexpr size_t scratch = sizeof(float) * 8 * 16 * SEG;
    static constexpr size_t bytes = staging > scratch ? staging : scratch;
};

// Four floats into four staging elements. The source is always fp32 -- the model
// hands this kernel fp32 activations and weights -- so the narrowing happens
// here, once per element, rather than per fragment load.
__device__ __forceinline__ void store4(float* dst, const float4& v) {
    *reinterpret_cast<float4*>(dst) = v;
}
__device__ __forceinline__ void store4(__half* dst, const float4& v) {
    // 4 halves = 8 bytes, written as one store. dst is 8-byte aligned: LD is a
    // multiple of 8 halves and the column offset is a multiple of VEC == 4.
    //
    // __float2half four times rather than __floats2half2_rn twice: the half2
    // intrinsics pull <cuda/std/...> into this translation unit, and CCCL
    // refuses to compile under MSVC's traditional preprocessor, which is what
    // the JIT build passes. Same instruction count either way.
    __half h[4];
    h[0] = __float2half(v.x);
    h[1] = __float2half(v.y);
    h[2] = __float2half(v.z);
    h[3] = __float2half(v.w);
    *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(h);
}

// Epilogue store. The fp16 overload is the SAME __float2half the attention
// kernel's staging path applies (dev_from_float in attention_wmma.cuh), so
// moving the conversion here changes no value -- it only moves it upstream of a
// global round trip.
// Four elements of A, widened to float4 whatever they were stored as. The
// vector width stays 4 ELEMENTS rather than 16 bytes, so an fp16 A reads 8
// bytes per lane instead of 16 -- which is the point -- and STORE_TILES below
// needs no change at all. Alignment holds: K % 4 == 0 is already required, so
// a row start is a multiple of 4 halves, and the column offset is a multiple
// of VEC == 4.
__device__ __forceinline__ float4 load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 load4(const __half* p) {
    const float2 raw = *reinterpret_cast<const float2*>(p);
    const __half* h = reinterpret_cast<const __half*>(&raw);
    return make_float4(__half2float(h[0]), __half2float(h[1]),
                       __half2float(h[2]), __half2float(h[3]));
}

__device__ __forceinline__ void store_out(float& d, float s)  { d = s; }
__device__ __forceinline__ void store_out(__half& d, float s)  { d = __float2half(s); }

// Four elements in one instruction, for the epilogue's coalesced path. fp32 is
// a 16-byte float4; fp16 is four halves, which is 8 bytes and so a float2-sized
// store. Both need the destination aligned to their own width, which the caller
// guarantees by only taking this path when N % 4 == 0 and the column offset is
// a multiple of 4 -- torch's base pointers are 16-byte aligned.
__device__ __forceinline__ void store_out4(float* p, const float (&v)[4]) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void store_out4(__half* p, const float (&v)[4]) {
    const __half h[4] = {__float2half(v[0]), __float2half(v[1]),
                         __float2half(v[2]), __float2half(v[3])};
    *reinterpret_cast<float2*>(p) = *reinterpret_cast<const float2*>(h);
}

// Exact GELU, matching F.gelu(approximate="none") rather than the tanh form.
// The harness's accuracy gate is tight enough that the tanh approximation
// fails atol on its own, so this has to be the erf version.
__device__ __forceinline__ float gelu_exact(float v) {
    // 1/sqrt(2), to more digits than float carries.
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

// OutT is the type C is STORED as, not the type anything is computed in: the
// accumulator and the epilogue are fp32 either way, and only the final store
// narrows. Writing __half is what lets the QKV projection hand the attention
// kernel a tensor it does not have to narrow itself -- see the note on
// linear_bias's out_half argument.
// InT is how A arrives, OutT how C leaves. Only A varies: W and bias are
// nn.Linear parameters and stay fp32, and the accumulator is fp32 throughout.
// An fp16 A is what the attention output already is once the QKV projection
// writes fp16 -- reading it wide would mean widening it only to narrow it
// again on the way into shared memory.
template <typename Math, int BM, int BN, int BK, bool ACT,
          typename InT, typename OutT>
__global__ __launch_bounds__(256)
void gemm_bias_act_kernel(const InT* __restrict__ A,
                          const float* __restrict__ W,
                          const float* __restrict__ bias,
                          OutT* __restrict__ C,
                          int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;   // 256
    constexpr int WM = BM / WARPS_M;                   // 32 rows per warp
    constexpr int WN = BN / WARPS_N;                   // 64 cols per warp
    constexpr int MT = WM / 16;                        // 2 wmma tiles down
    constexpr int NT = WN / 16;                        // 4 wmma tiles across
    constexpr int WK = Math::K;                        // fragment K extent
    constexpr int VEC = 4;                             // float4 global loads
    using store_t = typename Math::store_t;
    static_assert(BK % WK == 0, "BK must be a whole number of fragment K steps");

    // Padded leading dimension, for the same reason as in the attention
    // kernel: a fragment load walks a column of the tile, and an unpadded
    // stride of 32 floats puts all 16 rows in one bank. +4 rotates them onto 8
    // distinct bank groups (2-way instead of 16-way), and keeps the stride a
    // multiple of 4 floats so the float4 staging stores stay 16-byte aligned.
    constexpr int LD = GemmSmem<Math, BM, BN, BK>::LD;

    extern __shared__ __align__(16) char smem_raw[];
    store_t* A_s = reinterpret_cast<store_t*>(smem_raw);
    store_t* W_s = A_s + BM * LD;

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
                    pa[j] = load4(                        \
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
                store4(&A_s[r * LD + c], pa[j]);                                     \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int j = 0; j < W_PER_THREAD; ++j) {                                 \
                const int i = tid + j * NTHREADS;                                    \
                const int r = i / COLS4;                                             \
                const int c = (i - r * COLS4) * VEC;                                 \
                store4(&W_s[r * LD + c], pw[j]);                                     \
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
            wm::fragment<wm::matrix_a, 16, 16, WK, typename Math::frag_t, wm::row_major> af[MT];
            wm::fragment<wm::matrix_b, 16, 16, WK, typename Math::frag_t, wm::col_major> bf[NT];

            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                wm::load_matrix_sync(
                    af[mt], A_s + static_cast<size_t>(wm * WM + mt * 16) * LD + ks * WK, LD);
                // fp16 fragments arrive already narrowed, from store4 above.
                // `if constexpr`, not `if`: a runtime branch still type-checks
                // its dead body, and __float_to_tf32 returns a float that an
                // fp16 fragment element cannot be assigned from.
                if constexpr (Math::convert_frag) {
                    #pragma unroll
                    for (int t = 0; t < af[mt].num_elements; ++t) {
                        af[mt].x[t] = wm::__float_to_tf32(af[mt].x[t]);
                    }
                }
            }
            // W_s holds [n][k]; col_major with ldm = LD presents element (k, n)
            // at W_s[n * LD + k], which is W^T without moving anything.
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                wm::load_matrix_sync(
                    bf[nt], W_s + static_cast<size_t>(wn * WN + nt * 16) * LD + ks * WK, LD);
                if constexpr (Math::convert_frag) {
                    #pragma unroll
                    for (int t = 0; t < bf[nt].num_elements; ++t) {
                        bf[nt].x[t] = wm::__float_to_tf32(bf[nt].x[t]);
                    }
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
    // Through a shared scratch strip, rather than applying the activation to
    // accumulator elements directly. The element-to-row mapping inside an
    // accumulator fragment is architecture-defined, and the attention kernel
    // had to probe it at runtime to work around that. Here nothing needs the
    // row *during* the loop -- the bias depends only on the column and GELU is
    // elementwise -- so storing the tile first and reading it back with known
    // indexing avoids the whole problem. The K loop is over, so the staging
    // tiles are dead and host the scratch.
    //
    // The strip is SEG columns wide, not 16, and that width is the point. This
    // used to round-trip one 16x16 tile at a time and store it with one float
    // per lane, which put 16 consecutive floats -- 64 bytes, half a sector --
    // in each half-warp's store. No lane remapping can fix that: a row of a
    // 16-wide tile IS 64 bytes. Materialising SEG_T adjacent tiles into one
    // 16 x SEG strip makes a row SEG floats of *contiguous* global memory, so
    // float4 stores from 8 lanes cover 128 bytes.
    //
    // Row 1a of docs/OPTIMIZATION_LEDGER.md is the motivation: at shape 6, whose
    // working set does not fit in 8 GB, the whole forward runs at ~86 GB/s
    // against the 350 GB/s these kernels reach at shape 1, and a partial sector
    // costs disproportionately more in that regime -- which is why cuBLAS, with
    // a vectorized epilogue, degrades less there.
    __syncthreads();
    constexpr int SEG = GemmSmem<Math, BM, BN, BK>::SEG;   // columns per strip
    constexpr int SEG_T = SEG / 16;                        // wmma tiles in one
    constexpr int SEGS = NT / SEG_T;                       // strips across WN
    static_assert(SEG % 16 == 0 && NT % SEG_T == 0,
                  "the strip must be a whole number of fragment tiles");
    static_assert(GemmSmem<Math, BM, BN, BK>::bytes
                      >= NTHREADS / 32 * 16 * SEG * sizeof(float),
                  "epilogue scratch overruns the shared allocation");
    float* scratch = reinterpret_cast<float*>(smem_raw) + warp * (16 * SEG);

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int sg = 0; sg < SEGS; ++sg) {
            // Lay the segment's tiles side by side in one strip. The leading
            // dimension is SEG, so tile t lands at column t * 16.
            #pragma unroll
            for (int t = 0; t < SEG_T; ++t) {
                wm::store_matrix_sync(scratch + t * 16, acc[mt][sg * SEG_T + t],
                                      SEG, wm::mem_row_major);
            }
            __syncwarp();

            const int tile_m = m0 + wm * WM + mt * 16;
            const int tile_n = n0 + wn * WN + sg * SEG;

            // The vector path needs the whole strip in bounds and a row stride
            // that keeps every 4-element group aligned. Everything else falls
            // back to the scalar loop, which is also what handles a ragged M or
            // N -- so correctness never depends on the fast path being taken.
            const bool whole = (tile_m + 16 <= M) && (tile_n + SEG <= N)
                               && ((N & 3) == 0);
            if (whole) {
                constexpr int V = SEG / 4;             // float4s per row
                for (int idx = lane; idx < 16 * V; idx += 32) {
                    const int r = idx / V;
                    const int c = (idx - r * V) * 4;
                    float v[4];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        v[e] = scratch[r * SEG + c + e] + bias[tile_n + c + e];
                    }
                    // if constexpr: the bias-only form must not pay for a
                    // branch, and must not even instantiate erff.
                    if constexpr (ACT) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) v[e] = gelu_exact(v[e]);
                    }
                    // One RNE narrowing per element, in exactly the place the
                    // consumer would otherwise do it. store_out4 is a plain
                    // float4 for fp32 out.
                    store_out4(&C[static_cast<int64_t>(tile_m + r) * N
                                  + tile_n + c], v);
                }
            } else {
                for (int idx = lane; idx < 16 * SEG; idx += 32) {
                    const int r = idx / SEG;
                    const int c = idx - r * SEG;
                    const int gr = tile_m + r;
                    const int gc = tile_n + c;
                    if (gr < M && gc < N) {
                        const float v = scratch[idx] + bias[gc];
                        const float o = ACT ? gelu_exact(v) : v;
                        store_out(C[static_cast<int64_t>(gr) * N + gc], o);
                    }
                }
            }
            __syncwarp();
        }
    }
#endif
}


// Block tiles gemm_bias_gelu_kernel is instantiated for, smallest last. Only
// shapes that keep 8 warps in a 4x2 grid busy are here: BM % 64 == 0 (four warps
// down, one 16-row wmma tile each) and BN % 32 == 0 (two warps across).
constexpr int kGemmTileAuto    = -1;
constexpr int kGemmTile128x128 = 0;
constexpr int kGemmTile64x64   = 1;
constexpr int kGemmTile64x32   = 2;

// Fragment precision. Auto is fp16 -- same mantissa as tf32, twice the tensor-core
// rate -- and tf32 stays selectable so the comparison can be re-run.
constexpr int kGemmMathAuto = -1;
constexpr int kGemmMathTf32 = 0;
constexpr int kGemmMathFp16 = 1;

template <typename Math, int BM, int BN, int BK, bool ACT,
          typename InT, typename OutT>
void launch_gemm_bias_act(const torch::Tensor& x, const torch::Tensor& w,
                          const torch::Tensor& bias, torch::Tensor& out,
                          int64_t M, int64_t N, int64_t K) {
    using Plan = GemmSmem<Math, BM, BN, BK>;
    static_assert(Plan::bytes <= 48 * 1024,
                  "staging tiles exceed the 48 KB shared-memory budget");
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((M + BM - 1) / BM));
    gemm_bias_act_kernel<Math, BM, BN, BK, ACT, InT, OutT>
        <<<grid, block, Plan::bytes, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const InT*>(x.const_data_ptr()),
            w.const_data_ptr<float>(),
            bias.const_data_ptr<float>(),
            reinterpret_cast<OutT*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

// Tile and precision are independent axes, so the dispatch is one function of
// both rather than a switch duplicated per precision.
template <typename Math, bool ACT, typename InT, typename OutT>
void launch_gemm_tile(int tile, const torch::Tensor& x, const torch::Tensor& w,
                      const torch::Tensor& bias, torch::Tensor& out,
                      int64_t M, int64_t N, int64_t K) {
    switch (tile) {
        case kGemmTile128x128:
            launch_gemm_bias_act<Math, 128, 128, 32, ACT, InT, OutT>(x, w, bias, out, M, N, K);
            break;
        case kGemmTile64x64:
            launch_gemm_bias_act<Math, 64, 64, 32, ACT, InT, OutT>(x, w, bias, out, M, N, K);
            break;
        case kGemmTile64x32:
            launch_gemm_bias_act<Math, 64, 32, 32, ACT, InT, OutT>(x, w, bias, out, M, N, K);
            break;
        default:
            TORCH_CHECK(false, "linear_gelu: unknown tile id ", tile);
    }
}

// Which block tile to use, and why there is no longer a speed gate beside it.
//
// Measured with scripts/tune_linear_gelu.py, graph-timed against cuBLAS + a
// separate GELU, best tile per precision, control +/-1.0% to +/-2.9%:
//
//   M        K     N    tiles64   tf32     fp16    fp16/tf32   best tile
//   32       256   256        4   0.99x   1.634x     1.599x    64x32
//   128      64    64         2   1.36x   1.665x     1.221x    64x32
//   128      256   256        8   0.91x   1.327x     1.464x    64x32
//   512      256   256       32   0.71x   1.240x     1.758x    64x32
//   1024     32    32        16   2.10x   2.402x     1.127x    64x32
//   1024     256   256       64   0.86x   1.425x     1.650x    64x32
//   2048     256   256      128   1.66x   3.188x     1.921x    64x64
//   8192     256   256      512   1.34x   2.534x     1.892x    64x64
//   16384    256   256     1024   1.18x   2.139x     1.817x    64x64
//   320000   256   256    20000   1.03x   1.919x     1.861x    64x64
//   1024     512   512      128   1.19x   2.448x     2.058x    64x64
//   1024    1024  1024      256   0.86x   1.846x     2.147x    64x64
//   512     2048  2048      256   0.78x   1.638x     2.112x    64x64
//   1024     512  2048      512   0.82x   1.707x     2.093x    64x64
//   2048    4096  4096     2048   0.62x   1.371x     2.227x    128x128
//
// **fp16 wins at every one of the sixteen**, 1.24x to 2.40x, across grids from
// 2 tiles to 20000 and K/N from 32 to 4096 -- so the shape gate that tf32
// needed is gone. It existed because tf32 lost below two full waves of blocks
// and above K=N=512; fp16 loses at neither. What is left is the coverage check
// in linear_gelu() and the precondition below.
//
// PRECONDITION: operands must fit fp16's range. The model's activations are
// post-LayerNorm and its weights are O(1/sqrt(d)), so this holds by
// construction here; a caller feeding values past 65504, or clustered under
// fp16's 6.1e-5 smallest normal, would silently lose them. LINEAR_GELU="off"
// is the escape hatch, and the harness's accuracy gate is what would catch it.
//
// Two thresholds decide the tile, and they are different constraints:
//
//   * **Grid, below two full waves.** The crossover sits between 64 tiles
//     (64x32 wins) and 128 tiles (64x64 wins): under two waves a 64x64 grid
//     leaves SMs idle, and halving BN doubles the grid for a tile that is only
//     half as load-efficient. Both precisions cross over in the same place.
//   * **Contraction length, past L2.** A BMxBN tile does BM*BN*K MACs per
//     (BM+BN)*K floats loaded, so a bigger tile is more load-efficient -- an
//     argument that only bites once the operands stop fitting in the 3070's
//     4 MB L2. At K=N=2048 the weight is 16 MB and 64x64 still wins; at
//     K=N=4096 it is 64 MB and 128x128 wins by 1.29x over 64x64. That is one
//     measured point, past every shape this model issues (d_model tops out at
//     2048), so it is a guard rather than a tuned threshold.
int pick_gemm_tile(int64_t M, int64_t N, int64_t K) {
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t tiles_64 = ((M + 63) / 64) * ((N + 63) / 64);
    if (tiles_64 < 2 * sms) {
        return kGemmTile64x32;   // fill the card first; load efficiency second
    }
    if (K >= 4096) {
        return kGemmTile128x128;
    }
    return kGemmTile64x64;
}

}  // namespace


// Shared body of linear_gelu and linear_bias:  C = act(x @ weight^T + bias)
//
//   x       [..., K]  contiguous; every leading dimension is flattened into M
//   weight  [N, K]    nn.Linear layout
//   bias    [N]
//
// returns   [..., N]
//
// Declines (returns an undefined tensor) rather than raising when the shape or
// dtype is outside what the kernel covers, so the caller can fall back to
// F.linear instead of failing. Requirements: float32, SM 8.0+, and K divisible
// by 4 for the float4 staging path.
//
// `tile` selects the block tile: kGemmTileAuto asks pick_gemm_tile(), and an
// explicit id forces one, which is what scripts/tune_linear_gelu.py sweeps.
// `math` selects the fragment precision the same way; both default to auto.
static torch::Tensor gemm_bias_act(const torch::Tensor& x, const torch::Tensor& weight,
                                   const torch::Tensor& bias, int64_t tile, int64_t math,
                                   bool act, bool out_half, const char* who) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                who, ": all inputs must be CUDA tensors");
    TORCH_CHECK(weight.dim() == 2, who, ": weight must be 2-D [N, K]");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == weight.size(1),
                who, ": x's last dimension must match weight's, got ",
                x.sizes(), " and ", weight.sizes());
    TORCH_CHECK(bias.numel() == weight.size(0),
                who, ": bias must have ", weight.size(0), " elements");

    const int64_t K = weight.size(1);
    const int64_t N = weight.size(0);

    // A may be fp32 or fp16; W and bias are nn.Linear parameters and are always
    // fp32. An fp16 A is the attention output once the QKV projection writes
    // fp16 -- see linear_bias's out_half.
    const bool in_half = x.scalar_type() == torch::kFloat16;
    const bool covered =
        (x.scalar_type() == torch::kFloat32 || in_half) &&
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
    // Spelled from x's options with the dtype REPLACED rather than inherited:
    // an fp16 A must still be able to produce an fp32 C, which is exactly what
    // out_proj needs -- it consumes the fp16 attention output and feeds an fp32
    // residual add.
    auto out = torch::empty(
        out_sizes,
        xc.options().dtype(out_half ? torch::kFloat16 : torch::kFloat32));

    const int chosen = (tile == kGemmTileAuto) ? pick_gemm_tile(M, N, K) : tile;
    // The activation and the store type are both template parameters: the
    // bias-only form then has no erff in its store loop at all, and the fp16
    // store is a single __float2half rather than a branch.
    //
    // out_half is instantiated for the bias-only form ONLY. It exists for the
    // QKV projection, whose consumer narrows anyway; nothing consuming a GELU
    // output wants fp16, so pairing them would double the instantiation count
    // for a combination no caller asks for.
    // Four (InT, OutT, ACT) combinations are instantiated, not all eight, because
    // only four occur in this model:
    //
    //   (float, float, GELU)   linear_gelu -- the FFN's first projection
    //   (float, float, -)      ffn_out, and any fp32 fallback
    //   (float, __half, -)     the QKV projection, fp32 normed in
    //   (__half, float, -)     out_proj: fp16 attention output in, fp32 out
    //   (__half, __half, -)    the QKV projection once the LayerNorm feeding it
    //                          also emits fp16 -- see config.NORMED_FP16
    //
    // The last of those was refused when out_half landed, on the grounds that
    // no chain in this model needed it. That was true then and stopped being
    // true one cycle later, and the TORCH_CHECK is what said so -- loudly, at
    // the first forward pass, rather than by quietly mis-dispatching.
    //
    //   (__half, float, GELU)  linear_gelu once the norm2 LayerNorm feeding it
    //                          also emits fp16
    //
    // Still refused: (anything, __half, GELU). Nothing consuming a GELU output
    // wants fp16 -- ffn_out reads it and the GELU result is the one value here
    // that is computed rather than copied, so narrowing it would be a real
    // precision loss rather than a moved rounding. That combination costs
    // 2 math modes x 3 tiles and buys an error.
    TORCH_CHECK(!(out_half && act), who,
                ": out_half is not instantiated with an activation -- nothing "
                "consuming a GELU output wants fp16");

    if (in_half && act) {
        // An fp16 A into linear_gelu. Only bit-identical under fp16 fragments:
        // Fp16Math narrows A to __half when staging, so pre-narrowing changes
        // nothing, whereas Tf32Math keeps it float and would genuinely lose the
        // mantissa. The caller gates on that -- see _gelu_input_wants_fp16 --
        // and this refuses the pair rather than trusting it to.
        TORCH_CHECK(math != kGemmMathTf32, who,
                    ": an fp16 input with tf32 fragments would lose mantissa "
                    "rather than move a rounding -- tf32 stages A as float");
        launch_gemm_tile<Fp16Math, true, __half, float>(chosen, xc, wc, bc, out, M, N, K);
    } else if (in_half && out_half) {
        if (math == kGemmMathTf32) {
            launch_gemm_tile<Tf32Math, false, __half, __half>(chosen, xc, wc, bc, out, M, N, K);
        } else {
            launch_gemm_tile<Fp16Math, false, __half, __half>(chosen, xc, wc, bc, out, M, N, K);
        }
    } else if (in_half) {
        if (math == kGemmMathTf32) {
            launch_gemm_tile<Tf32Math, false, __half, float>(chosen, xc, wc, bc, out, M, N, K);
        } else {
            launch_gemm_tile<Fp16Math, false, __half, float>(chosen, xc, wc, bc, out, M, N, K);
        }
    } else if (out_half) {
        if (math == kGemmMathTf32) {
            launch_gemm_tile<Tf32Math, false, float, __half>(chosen, xc, wc, bc, out, M, N, K);
        } else {
            launch_gemm_tile<Fp16Math, false, float, __half>(chosen, xc, wc, bc, out, M, N, K);
        }
    } else if (math == kGemmMathTf32) {
        if (act) launch_gemm_tile<Tf32Math, true,  float, float>(chosen, xc, wc, bc, out, M, N, K);
        else     launch_gemm_tile<Tf32Math, false, float, float>(chosen, xc, wc, bc, out, M, N, K);
    } else {
        if (act) launch_gemm_tile<Fp16Math, true,  float, float>(chosen, xc, wc, bc, out, M, N, K);
        else     launch_gemm_tile<Fp16Math, false, float, float>(chosen, xc, wc, bc, out, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                who, ": kernel launch failed: ", cudaGetErrorString(err));
    return out;
}


// Linear followed by exact GELU, in one kernel:  GELU(x @ weight^T + bias).
// Matches F.gelu(F.linear(...), approximate="none"), not the tanh form -- the
// harness's accuracy gate is tight enough that the approximation fails atol on
// its own.
torch::Tensor linear_gelu(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor bias, int64_t tile, int64_t math) {
    return gemm_bias_act(x, weight, bias, tile, math, /*act=*/true,
                         /*out_half=*/false, "linear_gelu");
}


// Plain linear, in one kernel:  x @ weight^T + bias. The same GEMM without the
// activation, for the three projections that have none -- QKV, out_proj and
// ffn_out. Those were the last cuBLAS calls in the forward pass, and cuBLAS
// serves them in TF32 (or, at the small-K grading shapes, in *SIMT fp32* --
// shapes 7 and 12 land on cutlass_80_simt_sgemm), where fp16 fragments carry
// the same 10-bit mantissa at twice the tensor-core rate.
//
// `out_half` stores C as fp16 instead of fp32. It is for the QKV projection and
// the reason is that the conversion is already happening: the attention kernel
// stages q/k/v into shared memory through dev_from_float, narrowing fp32 to
// __half with RNE, so an fp32 qkv buffer is written wide, read wide, and then
// thrown away one mantissa at a time. Doing the same RNE here instead makes the
// values the attention kernel contracts BIT-IDENTICAL while halving both the
// write and the read: 12 MB to 6 MB each at grading shape 1.
//
// Only valid when the consumer really does contract in fp16. An fp32 or tf32
// attention precision, or the scalar kernel, needs the wide operands, so the
// caller gates this -- see optimized/kernels.py.
torch::Tensor linear_bias(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor bias, int64_t tile, int64_t math,
                          bool out_half) {
    return gemm_bias_act(x, weight, bias, tile, math, /*act=*/false,
                         out_half, "linear_bias");
}
