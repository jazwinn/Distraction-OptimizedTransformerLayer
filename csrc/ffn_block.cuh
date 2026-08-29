// Fused post-attention block: add + LayerNorm, Linear+GELU, Linear, add +
// LayerNorm -- one kernel, one block per tile of token rows.
//
// What this replaces, per layer:
//
//   x1, n1 = fused_add_layernorm(x, attn_out, norm2)   writes x1 and n1
//   h      = linear_gelu(n1, W_in, b_in)               writes h  [rows, F]
//   y      = ffn_out(h)                                writes y  [rows, D]
//   x2, n2 = fused_add_layernorm(x1, y, next_norm)     writes x2 and n2
//
// h, y and n1 exist only to be handed to the next kernel. Keeping them on chip
// removes those round trips to HBM. At 448 GB/s that is 9%-13% of the forward
// on the mid-size grading shapes, 3% on shape 6.
//
// Why the ffn_dim <= 128 bound is real and not a tuning choice: the second GEMM
// reduces over ffn_dim, so a block cannot produce ANY output column until it
// holds the whole intermediate row. BN is therefore pinned to F rather than
// tuned. At F = 1024 the [ROWS, F] tile does not fit in 48 KB at any useful
// ROWS, and shrinking ROWS to make it fit multiplies the per-block weight
// re-reads -- which stop being free once the weights (2*D*F*4 bytes) leave the
// 4 MB L2. D, F <= 128 keeps both the tile on chip and the weights L2-resident.
//
// Layout: one warp owns one 16-wide column tile of each GEMM, so the warp count
// follows max(D, F)/16 rather than being chosen separately. Rows per block is
// 16 -- one wmma m-tile -- which keeps the three shared tiles to 24 KB at
// D = F = 128.

#pragma once

#include "kernel_common.cuh"

namespace {

// GELU must be the erf form: the tanh approximation misses the harness atol on
// its own. Identical to linear_gelu.cuh deliberately -- the fused and unfused
// paths have to agree bit for bit on this or the A/B is measuring two answers.
__device__ __forceinline__ float ffn_gelu_exact(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

// Warp-wide sum. The row belongs to one warp, so a butterfly is the whole
// reduction -- no shared scratch, no __syncthreads, and every lane ends with
// the total rather than only lane 0.
__device__ __forceinline__ float ffn_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

constexpr int kFfnRows  = 16;   // one wmma m-tile
constexpr int kFfnWmmaK = 8;    // tf32 fragment K
constexpr int kFfnPad   = 4;    // wmma wants a float ldm divisible by 4

template <int D, int F>
struct FfnCfg {
    static constexpr int ROWS    = kFfnRows;
    static constexpr int LD_D    = D + kFfnPad;
    static constexpr int LD_F    = F + kFfnPad;
    static constexpr int NT_D    = D / 16;   // column tiles of the 2nd GEMM
    static constexpr int NT_F    = F / 16;   // column tiles of the 1st GEMM
    static constexpr int WARPS   = (NT_D > NT_F ? NT_D : NT_F);
    static constexpr int THREADS = WARPS * 32;
    // s_x1 [ROWS, LD_D], s_h [ROWS, LD_F], s_y [ROWS, LD_D]
    static constexpr int SMEM =
        static_cast<int>(sizeof(float)) *
        (ROWS * LD_D + ROWS * LD_F + ROWS * LD_D);
    static constexpr bool SUPPORTED =
        D >= 16 && F >= 16 && D % 16 == 0 && F % 16 == 0 &&
        D <= 128 && F <= 128 && SMEM <= 48 * 1024 && THREADS <= 1024;
};

// One tf32 GEMM: acc = A[16, K_DIM] * W[n0:n0+16, :]^T, with W held row-major
// as [N, K] exactly as nn.Linear stores it. Reading it col_major with ldm =
// K_DIM gives B[k][n] = W[n0+n][k], so no host-side transpose is needed.
template <int K_DIM>
__device__ __forceinline__ void ffn_gemm_tile(
        const float* __restrict__ a_tile, int lda,
        const float* __restrict__ w, int n0,
        wm::fragment<wm::accumulator, 16, 16, kFfnWmmaK, float>& acc) {
    #pragma unroll
    for (int k0 = 0; k0 < K_DIM; k0 += kFfnWmmaK) {
        wm::fragment<wm::matrix_a, 16, 16, kFfnWmmaK, wm::precision::tf32, wm::row_major> af;
        wm::fragment<wm::matrix_b, 16, 16, kFfnWmmaK, wm::precision::tf32, wm::col_major> bf;
        wm::load_matrix_sync(af, a_tile + k0, lda);
        wm::load_matrix_sync(bf, w + static_cast<int64_t>(n0) * K_DIM + k0, K_DIM);
        #pragma unroll
        for (int t = 0; t < af.num_elements; ++t) af.x[t] = wm::__float_to_tf32(af.x[t]);
        #pragma unroll
        for (int t = 0; t < bf.num_elements; ++t) bf.x[t] = wm::__float_to_tf32(bf.x[t]);
        wm::mma_sync(acc, af, bf, acc);
    }
}

template <int D, int F>
__global__ __launch_bounds__(FfnCfg<D, F>::THREADS)
void fused_ffn_block_kernel(const float* __restrict__ x,
                            const float* __restrict__ attn_out,
                            const float* __restrict__ w1,
                            const float* __restrict__ b1,
                            const float* __restrict__ W_in,
                            const float* __restrict__ b_in,
                            const float* __restrict__ W_out,
                            const float* __restrict__ b_out,
                            const float* __restrict__ w2,
                            const float* __restrict__ b2,
                            float* __restrict__ x_out,
                            float* __restrict__ normed_out,
                            int rows, float eps) {
    using Cfg = FfnCfg<D, F>;
    constexpr int VPL = D / 32 + 1;   // per-lane slots for one row of width D

    extern __shared__ __align__(16) char ffn_smem[];
    float* s_x1 = reinterpret_cast<float*>(ffn_smem);
    float* s_h  = s_x1 + Cfg::ROWS * Cfg::LD_D;
    float* s_y  = s_h + Cfg::ROWS * Cfg::LD_F;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int row0     = static_cast<int>(blockIdx.x) * Cfg::ROWS;
    const int row_span = min(Cfg::ROWS, rows - row0);

    // ---- stage 1: x1 = x + attn_out, n1 = LayerNorm(x1) * w1 + b1 ----------
    // One warp per row, so mean and variance are butterflies. n1 lands in s_y,
    // dead until stage 3 writes it -- the alternative is a fourth [ROWS, D]
    // tile the shared budget cannot spare.
    for (int r = warp; r < row_span; r += Cfg::WARPS) {
        const int64_t base = static_cast<int64_t>(row0 + r) * D;
        float v[VPL];
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            const int d = lane + t * 32;
            const float s = (d < D) ? (x[base + d] + attn_out[base + d]) : 0.0f;
            v[t] = s;
            if (d < D) s_x1[r * Cfg::LD_D + d] = s;
            sum += s;
        }
        const float mean = ffn_warp_sum(sum) / static_cast<float>(D);
        float cc = 0.0f;
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            if (lane + t * 32 < D) { const float c = v[t] - mean; cc += c * c; }
        }
        const float rstd = rsqrtf(ffn_warp_sum(cc) / static_cast<float>(D) + eps);
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            const int d = lane + t * 32;
            if (d < D) s_y[r * Cfg::LD_D + d] = (v[t] - mean) * rstd * w1[d] + b1[d];
        }
    }
    // Zero the tail rows: mma_sync has no masking, and one uninitialised value
    // contaminates the whole 16x16 accumulator, not just the row it came from.
    for (int i = tid; i < (Cfg::ROWS - row_span) * Cfg::LD_D; i += Cfg::THREADS) {
        s_y[row_span * Cfg::LD_D + i] = 0.0f;
    }
    __syncthreads();

    // ---- stage 2: h = GELU(n1 @ W_in^T + b_in), [ROWS, F] ------------------
    if (warp < Cfg::NT_F) {
        wm::fragment<wm::accumulator, 16, 16, kFfnWmmaK, float> acc;
        wm::fill_fragment(acc, 0.0f);
        ffn_gemm_tile<D>(s_y, Cfg::LD_D, W_in, warp * 16, acc);
        wm::store_matrix_sync(s_h + warp * 16, acc, Cfg::LD_F, wm::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < Cfg::ROWS * F; i += Cfg::THREADS) {
        const int r = i / F;
        const int c = i - r * F;
        s_h[r * Cfg::LD_F + c] = ffn_gelu_exact(s_h[r * Cfg::LD_F + c] + b_in[c]);
    }
    __syncthreads();

    // ---- stage 3: y = h @ W_out^T, [ROWS, D] -------------------------------
    if (warp < Cfg::NT_D) {
        wm::fragment<wm::accumulator, 16, 16, kFfnWmmaK, float> acc;
        wm::fill_fragment(acc, 0.0f);
        ffn_gemm_tile<F>(s_h, Cfg::LD_F, W_out, warp * 16, acc);
        wm::store_matrix_sync(s_y + warp * 16, acc, Cfg::LD_D, wm::mem_row_major);
    }
    __syncthreads();

    // ---- stage 4: x2 = x1 + y + b_out, n2 = LayerNorm(x2) * w2 + b2 --------
    for (int r = warp; r < row_span; r += Cfg::WARPS) {
        const int64_t base = static_cast<int64_t>(row0 + r) * D;
        float v[VPL];
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            const int d = lane + t * 32;
            const float s = (d < D)
                ? (s_x1[r * Cfg::LD_D + d] + s_y[r * Cfg::LD_D + d] + b_out[d])
                : 0.0f;
            v[t] = s;
            if (d < D) x_out[base + d] = s;
            sum += s;
        }
        const float mean = ffn_warp_sum(sum) / static_cast<float>(D);
        float cc = 0.0f;
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            if (lane + t * 32 < D) { const float c = v[t] - mean; cc += c * c; }
        }
        const float rstd = rsqrtf(ffn_warp_sum(cc) / static_cast<float>(D) + eps);
        #pragma unroll
        for (int t = 0; t * 32 < D; ++t) {
            const int d = lane + t * 32;
            if (d < D) normed_out[base + d] = (v[t] - mean) * rstd * w2[d] + b2[d];
        }
    }
}

template <int D, int F>
bool launch_ffn_block(const torch::Tensor& x, const torch::Tensor& attn_out,
                      const torch::Tensor& w1, const torch::Tensor& b1,
                      const torch::Tensor& W_in, const torch::Tensor& b_in,
                      const torch::Tensor& W_out, const torch::Tensor& b_out,
                      const torch::Tensor& w2, const torch::Tensor& b2,
                      torch::Tensor& x_out, torch::Tensor& normed_out,
                      int rows, double eps) {
    using Cfg = FfnCfg<D, F>;
    if (!Cfg::SUPPORTED) return false;

    const dim3 grid((rows + Cfg::ROWS - 1) / Cfg::ROWS);
    fused_ffn_block_kernel<D, F><<<grid, Cfg::THREADS, Cfg::SMEM,
                                   at::cuda::getCurrentCUDAStream()>>>(
        x.const_data_ptr<float>(), attn_out.const_data_ptr<float>(),
        w1.const_data_ptr<float>(), b1.const_data_ptr<float>(),
        W_in.const_data_ptr<float>(), b_in.const_data_ptr<float>(),
        W_out.const_data_ptr<float>(), b_out.const_data_ptr<float>(),
        w2.const_data_ptr<float>(), b2.const_data_ptr<float>(),
        x_out.data_ptr<float>(), normed_out.data_ptr<float>(),
        rows, static_cast<float>(eps));
    return true;
}

// Dispatch on the two widths. Only the (D, F) pairs the grading shapes use are
// instantiated: every extra pair is a full expansion of two wmma GEMMs and the
// .pyd is already 12 MB.
bool dispatch_ffn_block(int D, int F,
                        const torch::Tensor& x, const torch::Tensor& attn_out,
                        const torch::Tensor& w1, const torch::Tensor& b1,
                        const torch::Tensor& W_in, const torch::Tensor& b_in,
                        const torch::Tensor& W_out, const torch::Tensor& b_out,
                        const torch::Tensor& w2, const torch::Tensor& b2,
                        torch::Tensor& x_out, torch::Tensor& normed_out,
                        int rows, double eps) {
#define FFN_CASE(DD, FF)                                                       \
    if (D == (DD) && F == (FF))                                                \
        return launch_ffn_block<DD, FF>(x, attn_out, w1, b1, W_in, b_in,       \
                                        W_out, b_out, w2, b2, x_out,           \
                                        normed_out, rows, eps);
    FFN_CASE(128, 128)
    FFN_CASE(64, 64)
    FFN_CASE(48, 48)
    FFN_CASE(32, 32)
    FFN_CASE(16, 16)
#undef FFN_CASE
    return false;
}

}  // namespace

// Fused residual add + LayerNorm + FFN + residual add + LayerNorm.
// Returns {x2, normed2}, or an empty vector when the shape is not covered, so
// the caller runs the unfused chain instead.
std::vector<torch::Tensor> fused_ffn_block(torch::Tensor x, torch::Tensor attn_out,
                                           torch::Tensor norm1_w, torch::Tensor norm1_b,
                                           torch::Tensor ffn_in_w, torch::Tensor ffn_in_b,
                                           torch::Tensor ffn_out_w, torch::Tensor ffn_out_b,
                                           torch::Tensor norm2_w, torch::Tensor norm2_b,
                                           double eps) {
    TORCH_CHECK(x.is_cuda(), "fused_ffn_block: x must be CUDA");
    TORCH_CHECK(x.sizes() == attn_out.sizes(),
                "fused_ffn_block: x and attn_out must match, got ",
                x.sizes(), " and ", attn_out.sizes());
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "fused_ffn_block: float32 only, got ", x.scalar_type());
    TORCH_CHECK(ffn_in_w.dim() == 2 && ffn_out_w.dim() == 2,
                "fused_ffn_block: ffn weights must be 2-D");

    const int64_t D = x.size(-1);
    const int64_t F = ffn_in_w.size(0);
    TORCH_CHECK(ffn_in_w.size(1) == D && ffn_out_w.size(0) == D && ffn_out_w.size(1) == F,
                "fused_ffn_block: weight shapes disagree with d_model=", D,
                " ffn_dim=", F);

    auto xc = x.contiguous();
    auto ac = attn_out.contiguous();
    auto wi = ffn_in_w.contiguous();
    auto wo = ffn_out_w.contiguous();
    const int64_t rows = xc.numel() / D;

    auto x_out = torch::empty_like(xc);
    auto normed_out = torch::empty_like(xc);

    if (!dispatch_ffn_block(static_cast<int>(D), static_cast<int>(F),
                            xc, ac, norm1_w.contiguous(), norm1_b.contiguous(),
                            wi, ffn_in_b.contiguous(), wo, ffn_out_b.contiguous(),
                            norm2_w.contiguous(), norm2_b.contiguous(),
                            x_out, normed_out, static_cast<int>(rows), eps)) {
        return {};
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_ffn_block: launch failed: ", cudaGetErrorString(err));
    return {x_out, normed_out};
}
