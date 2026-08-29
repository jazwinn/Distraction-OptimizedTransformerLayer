// Custom CUDA extension for the transformer layer: the module definition, and
// a map of the directory.
//
// This .cu is the only translation unit that builds these kernels. Each .cuh
// below is a vertical slice -- the kernels, their launch rules, and the public
// entry point Python calls -- so a reader chasing one operation opens one file.
// The internals of each slice sit in an unnamed namespace; see
// kernel_common.cuh for why.
//
//   kernel_common.cuh       out_base(), the nvcuda::wmma alias. Shared bits.
//   attention_scalar.cuh    attention, one thread per query row (impl=1)
//   attention_wmma.cuh      attention on tensor cores via nvcuda::wmma (impl=2)
//   attention_dispatch.cuh  which attention kernel runs, and
//                           fused_attention_forward()
//   add_layernorm.cuh       fused residual add + LayerNorm, and
//                           fused_add_layernorm()
//   linear_gelu.cuh         GEMM with a bias + GELU epilogue, and linear_gelu()
//   tile_attention.h/.cu    attention on the CUDA tile programming model
//                           (impl=3/4/5/6). A SEPARATE translation unit: it
//                           needs -std=c++20 -enable-tile, which the torch
//                           headers here do not survive. tile_attention.h is
//                           the plain-pointer boundary between the two.
//   TUNING.md               every measurement behind every constant above
//
// The attention op, since it is spread over four of those files:
//
// FlashAttention-style -- the [B,H,S,S] score matrix is never written to global
// memory. K/V are streamed through shared memory a tile at a time and the
// softmax is accumulated online (running max + running sum). Three
// implementations of the same math:
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
//   tile    the same math on the CUDA tile programming model. float32 and
//           head_dim in {8,16,32,64}, and only when the build found CUDA 13.3+.
//           impl=3/4/5/6 select fp32, bf16, tf32 and fp16 operands; none is
//           picked by impl=0.
//
// A head_dim outside those sets -- 24 or 48, say, or the 256 the grading set
// contains -- falls back to SDPA. head_dim is a template parameter, so each
// supported value is a separately compiled kernel and the set cannot be
// open-ended.
//
// impl=0 (auto) does not simply take the first of the three that covers a case:
// at head_dim 128 the wmma kernel is correct and slower than SDPA, so auto
// declines it there and the fallback serves the call instead. Coverage and
// preference are separate questions -- see run_kernel() in
// attention_dispatch.cuh.

#include "attention_dispatch.cuh"
#include "add_layernorm.cuh"
#include "linear_gelu.cuh"
#include "ffn_block.cuh"
#include "tile_attention.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


// Build smoke test: a copy kernel, so a caller can prove the extension
// compiled, loaded and can reach the GPU without depending on any of the real
// kernels above.
namespace {

__global__ void identity_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_test_identity", &smoke_test_identity,
          "Copies input to output via a trivial CUDA kernel (build smoke test)");
    m.def("fused_ffn_block", &fused_ffn_block,
          "Fused add+LayerNorm, Linear+GELU, Linear, add+LayerNorm. "
          "Returns {x2, normed2}, or an empty list when the shape is not covered.",
          pybind11::arg("x"), pybind11::arg("attn_out"),
          pybind11::arg("norm1_w"), pybind11::arg("norm1_b"),
          pybind11::arg("ffn_in_w"), pybind11::arg("ffn_in_b"),
          pybind11::arg("ffn_out_w"), pybind11::arg("ffn_out_b"),
          pybind11::arg("norm2_w"), pybind11::arg("norm2_b"),
          pybind11::arg("eps"));
    m.def("linear_gelu", &linear_gelu,
          "GELU(x @ weight^T + bias) in one kernel; undefined tensor if unsupported",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("tile") = kGemmTileAuto,
          pybind11::arg("math") = kGemmMathAuto);
    m.def("fused_add_layernorm", &fused_add_layernorm,
          "Fused residual add + LayerNorm; returns {x + sub, norm(x + sub)}",
          pybind11::arg("x"),
          pybind11::arg("sub"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps") = 1e-5);
    m.def("layernorm_set_block_threads",
          [](int threads) { layernorm_threads_override() = threads; },
          "Force fused_add_layernorm's block size, or 0 to restore the "
          "width-scaled rule. Runtime-settable so candidates can be timed in "
          "one process. Clamped to a whole number of warps, at most 1024.",
          pybind11::arg("threads"));
    m.def("layernorm_set_warp_width",
          [](int width) { layernorm_warp_width_override() = width; },
          "Widest row that fused_add_layernorm sends to the warp-per-row "
          "kernel. 0 disables it (block-per-row everywhere), -1 restores the "
          "measured default. Capped at 256. Runtime-settable so both kernels "
          "can be timed in one process.",
          pybind11::arg("width"));
    m.def("layernorm_warp_width",
          []() { return layernorm_warp_width(); },
          "The current warp-per-row width threshold");
    m.def("layernorm_set_warp_rows",
          [](int rows) { layernorm_warp_rows_override() = rows; },
          "Rows per block for the warp-per-row kernel; 0 restores the measured "
          "default.",
          pybind11::arg("rows"));
    m.def("layernorm_warp_rows",
          []() { return layernorm_warp_rows(); },
          "The current rows-per-block for the warp-per-row kernel");
    m.def("layernorm_set_fused_reduce",
          [](bool enabled) { layernorm_fused_reduce_flag() = enabled; },
          "Whether the block-per-row add+LayerNorm folds its two second-pass "
          "sums into one reduction. Identical arithmetic either way; "
          "runtime-settable so both can be timed in one process.",
          pybind11::arg("enabled"));
    m.def("layernorm_fused_reduce_enabled",
          []() { return layernorm_fused_reduce_flag(); },
          "Whether the fused second-pass reduction is currently on");
    m.def("layernorm_blocks_per_sm",
          [](int64_t D) { return layernorm_blocks_per_sm(D); },
          "Blocks of the float32 fused_add_layernorm kernel one SM can hold at "
          "this row width, from the occupancy API. Multiply by the block's warp "
          "count to get warps in flight against the SM's 48.",
          pybind11::arg("D"));
    m.def("layernorm_block_threads",
          [](int64_t D) { return layernorm_block_threads(D); },
          "The block size fused_add_layernorm would launch for a row of D "
          "elements, override included",
          pybind11::arg("D"));
    m.def("tile_workspace_bytes",
          [](int B, int H, int S, int head_dim, bool is_causal, int impl) {
              // impl codes match fused_attention_forward's: 3 tile-fp32,
              // 4 tile-bf16, 5 tile-tf32, 6 tile-fp16. Non-zero means the launcher intends
              // to split the key range for this shape, so a test can assert
              // that it is really exercising the split path rather than
              // silently passing on the single-pass one.
              tile_attn::MathMode math = tile_attn::MathMode::Fp32;
              if (impl == 4) math = tile_attn::MathMode::Bf16;
              if (impl == 5) math = tile_attn::MathMode::Tf32;
              if (impl == 6) math = tile_attn::MathMode::Fp16;
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
    m.def("wmma_set_fp16",
          [](bool on) { wmma_fp16_flag() = on; },
          "Contract fp32 q/k/v in fp16 fragments (true, the default) or tf32 "
          "(false). Same 10-bit mantissa either way; fp16 runs at twice the "
          "tensor-core rate and halves every staged tile. Runtime-settable so "
          "both can be timed in one process.",
          pybind11::arg("on"));
    m.def("wmma_fp16",
          []() { return wmma_fp16_flag(); },
          "Whether fp32 tensors are currently contracted in fp16 fragments");
    m.def("wmma_set_softmax_mode",
          [](int mode) {
              TORCH_CHECK(mode >= 0 && mode <= 2,
                          "wmma_set_softmax_mode: mode must be 0, 1 or 2");
              softmax_mode_flag() = mode;
          },
          "Which softmax the wmma attention kernel runs. 0 = the original "
          "(score * scale, then __expf, plus an explicit -inf test). 1 = the "
          "base-2 domain: one fused score multiply, a bare exp2f, and no -inf "
          "test, at identical accuracy (the default). 2 = as 1 with "
          "scale*log2(e) also folded into Q, which drops the score-side "
          "multiply but narrows both constants to fp16. Selects among three "
          "kernel instantiations; runtime-settable so all three can be timed "
          "in one process.",
          pybind11::arg("mode"));
    m.def("wmma_softmax_mode",
          []() { return softmax_mode_flag(); },
          "Which softmax mode the wmma attention kernel is currently using");
    m.def("wmma_set_mask_classify",
          [](bool on) { mask_classify_flag() = on; },
          "Classify each key tile once and skip the per-element bounds, "
          "causal and mask tests on the interior ones (true, the default), or "
          "test every score element the old way (false). Identical results "
          "either way; runtime-settable so both can be timed in one process.",
          pybind11::arg("on"));
    m.def("wmma_mask_classify_enabled",
          []() { return mask_classify_flag(); },
          "Whether per-tile mask classification is currently on");
    m.def("wmma_set_direct_o",
          [](bool on) { direct_o_flag() = on; },
          "Have the wmma attention epilogue store O from the accumulator "
          "fragments straight to global memory, which also shrinks the block's "
          "shared O tile to one fragment per warp (true, the default), or "
          "stage the whole block tile through shared memory the old way "
          "(false). The flag picks the shared-memory layout and the launch "
          "size as well as the code path, so an A/B of it covers both halves "
          "of the change in one process.",
          pybind11::arg("on"));
    m.def("wmma_direct_o_enabled",
          []() { return direct_o_flag(); },
          "Whether the wmma epilogue is currently storing O direct to global");
    m.def("wmma_set_split_kv",
          [](bool on) { split_kv_flag() = on; },
          "Enable/disable the wmma kernel split-KV (Flash-Decoding) path. "
          "Runtime-settable so both paths can be timed in one process.",
          pybind11::arg("on"));
    m.def("wmma_split_kv_enabled",
          []() { return split_kv_flag(); },
          "Whether the wmma kernel split-KV path is currently enabled");
    m.def("wmma_set_split_count",
          [](int n) { split_count_override() = n; },
          "Force the wmma split count; 0 restores the measured rule. Used to "
          "sweep counts a shape would not otherwise be given.",
          pybind11::arg("n"));
    m.def("wmma_split_count",
          [](int B, int H, int S, int head_dim, bool is_causal) {
              // BLOCK_N comes back from wmma_grid_info rather than being
              // rederived here: it is a build-line macro that
              // scripts/tune_block_shapes.py overrides, so a second copy of
              // the rule would quietly disagree with the launcher.
              auto info = wmma_grid_info(B, H, S, head_dim);
              if (info[0] == 0) return 1;
              return wmma_split_count(
                  static_cast<int>(info[0]), static_cast<int>(info[1]),
                  split_key_tiles(S, static_cast<int>(info[3]), is_causal),
                  head_dim);
          },
          "How many ways the launcher would split the key range for this "
          "shape. 1 means it declines to split.",
          pybind11::arg("B"), pybind11::arg("H"), pybind11::arg("S"),
          pybind11::arg("head_dim"), pybind11::arg("is_causal"));
    m.def("wmma_grid_info",
          [](int B, int H, int S, int head_dim) {
              return wmma_grid_info(B, H, S, head_dim);
          },
          "{grid blocks, blocks the card holds at once, BLOCK_M, BLOCK_N} "
          "for this "
          "attention shape on the fp16 compute path. The occupancy the "
          "split-KV gate is built on; blocks << resident means the grid "
          "cannot fill the card. {0,0,0} if this head_dim has no kernel.",
          pybind11::arg("B"), pybind11::arg("H"), pybind11::arg("S"),
          pybind11::arg("head_dim"));
    m.def("wmma_set_causal_reverse",
          [](bool enabled) { causal_reverse_flag() = enabled; },
          "Enable/disable the wmma kernel's causal block-index reversal "
          "(longest-processing-time-first). Runtime-settable so both mappings "
          "can be timed in one process. No effect on dense shapes.",
          pybind11::arg("enabled"));
    m.def("wmma_causal_reverse_enabled",
          []() { return causal_reverse_flag(); },
          "Whether the wmma kernel's causal block-index reversal is on");
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
