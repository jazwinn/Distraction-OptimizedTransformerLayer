// Custom fused attention extension.
//
// FlashAttention-style scalar kernel: one thread per query row, K/V streamed
// through shared memory a tile at a time, softmax accumulated online, so the
// [B,H,S,S] score matrix is never written to global memory. Plain fp32 FMA,
// which makes it more precise than the baseline -- it never rounds through
// TF32 the way cuBLAS does.
//
// Coverage: head_dim in {8,16,32,64}, float/half/bfloat16. Anything else
// (e.g. head_dim 128) falls back to the ATen implementation at the bottom,
// which mirrors BaselineSelfAttention.forward exactly.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void identity_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// ---------------------------------------------------------------------------
// Scalar kernel: one thread owns one query row.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_DIM>
__global__ void fused_attention_kernel(const scalar_t* __restrict__ q,
                                       const scalar_t* __restrict__ k,
                                       const scalar_t* __restrict__ v,
                                       const bool* __restrict__ mask,
                                       int64_t ms0, int64_t ms1,
                                       int64_t ms2, int64_t ms3,
                                       scalar_t* __restrict__ out,
                                       int B, int H, int S,
                                       bool is_causal, float scale) {
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

    const int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_DIM;

    // q is read once and reused against every key, so it earns registers.
    float q_reg[HEAD_DIM];
    if (active) {
        const scalar_t* q_row = q + bh_off + static_cast<int64_t>(i) * HEAD_DIM;
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

        // Contiguous copy: [B,H,S,head_dim] means a run of keys is one flat
        // span, so this stays coalesced without any index gymnastics.
        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * HEAD_DIM;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * HEAD_DIM;
        for (int idx = tid; idx < n_keys * HEAD_DIM; idx += BLOCK_M) {
            k_s[idx] = k_base[idx];
            v_s[idx] = v_base[idx];
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
        scalar_t* out_row = out + bh_off + static_cast<int64_t>(i) * HEAD_DIM;
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
                   const int64_t* ms, torch::Tensor& out,
                   int B, int H, int S, bool is_causal, double scale) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = (HEAD_DIM >= 64) ? 64 : 128;

    const dim3 block(BLOCK_M);
    const dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    const size_t smem = 2 * BLOCK_N * HEAD_DIM * sizeof(scalar_t);

    fused_attention_kernel<scalar_t, HEAD_DIM>
        <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            out.data_ptr<scalar_t>(), B, H, S, is_causal,
            static_cast<float>(scale));
}

// Returns false when this head_dim has no specialization compiled.
template <typename scalar_t>
bool dispatch_head_dim(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const bool* mask_ptr,
                       const int64_t* ms, torch::Tensor& out,
                       int B, int H, int S, int head_dim,
                       bool is_causal, double scale) {
    switch (head_dim) {
        case 8:
            launch_kernel<scalar_t, 8>(q, k, v, mask_ptr, ms, out, B, H, S, is_causal, scale);
            return true;
        case 16:
            launch_kernel<scalar_t, 16>(q, k, v, mask_ptr, ms, out, B, H, S, is_causal, scale);
            return true;
        case 32:
            launch_kernel<scalar_t, 32>(q, k, v, mask_ptr, ms, out, B, H, S, is_causal, scale);
            return true;
        case 64:
            launch_kernel<scalar_t, 64>(q, k, v, mask_ptr, ms, out, B, H, S, is_causal, scale);
            return true;
        default:
            return false;
    }
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


// Fused scaled-dot-product attention.
//
//   q, k, v          : [B, H, S, head_dim], CUDA
//   attn_mask        : optional bool, broadcastable to [B, H, S, S].
//                      SDPA convention -- true means ALLOWED to attend.
//   is_causal        : lower-triangular masking. Never combined with attn_mask;
//                      the caller folds causal into attn_mask when both apply.
//   scale            : score multiplier, normally 1/sqrt(head_dim).
//
// returns            : [B, H, S, head_dim]
torch::Tensor fused_attention_forward(torch::Tensor q,
                                      torch::Tensor k,
                                      torch::Tensor v,
                                      c10::optional<torch::Tensor> attn_mask,
                                      bool is_causal,
                                      double scale) {
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

    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();

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

    auto out = torch::empty_like(qc);

    bool handled = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        qc.scalar_type(), "fused_attention_forward", [&] {
            handled = dispatch_head_dim<scalar_t>(
                qc, kc, vc, mask_ptr, ms, out, B, H, S, head_dim, is_causal, scale);
        });

    if (!handled) {
        return attention_aten(qc, kc, vc, attn_mask, is_causal, scale);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_attention_forward: kernel launch failed: ", cudaGetErrorString(err));
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_test_identity", &smoke_test_identity,
          "Copies input to output via a trivial CUDA kernel (build smoke test)");
    m.def("fused_attention_forward", &fused_attention_forward,
          "Fused scaled-dot-product attention",
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("attn_mask") = c10::nullopt,
          pybind11::arg("is_causal") = false,
          pybind11::arg("scale"));
}
