// Interface to the cuTile attention kernel in tile_attention.cu.
//
// Deliberately free of torch and cuda_tile headers. tile_attention.cu needs
// -std=c++20 -enable-tile and CUDA 13.3's <cuda_tile.h>; fused_attention.cu
// needs the torch headers. Keeping the two in separate translation units that
// meet at this plain-pointer boundary means neither has to compile under the
// other's requirements.

#pragma once

#include <cuda_runtime.h>

namespace tile_attn {

// Precision of the two GEMM operands, and the only thing that decides whether
// the tensor cores run at all. Everything else -- the softmax, the running
// max/sum, the accumulator, and the tensors in and out -- stays fp32 in every
// mode.
//
// No tensor core on any shipped architecture performs a full fp32 matmul, so
// Fp32 lands on the general-purpose CUDA cores everywhere. Narrowing the
// operands is what moves the work onto the MMA units. A newer GPU does not
// change that; it changes how much faster the MMA path is once you are on it.
//
// There is deliberately no Fp16 mode: cuTile accumulates a __half matmul into
// __half, and attention sums hundreds of products per output, so the running
// total would lose precision as it grows. bf16 is the only narrow type cuTile
// accumulates into float.
enum class MathMode : int {
    Fp32 = 0,  // exact. CUDA cores.    ~1e-6 against an exact reference.
    Bf16 = 1,  // tensor cores.         ~4e-3 -- 8 significand bits.
    Tf32 = 2,  // tensor cores.         ~1e-3 -- see supports().
};

// True when this build actually contains the tile kernel (CUDA 13.3+ and
// -enable-tile). False builds still link; launch() just declines.
bool available();

// True when `mode` was compiled into this build.
//
// Tf32 is the one that moves. <cuda_tile.h> forward-declares __nv_tf32 but no
// CUDA 13.3 header defines it, so ct::tile<__nv_tf32, ...> is an incomplete
// type and will not compile -- bf16 works only because <cuda_bf16.h> completes
// its forward declaration the same way. When a toolkit ships the defining
// header, build with -DTILE_HAVE_TF32 and this starts returning true for Tf32
// with no other change; the kernel is already written for it.
bool supports(MathMode mode);

// Fused scaled-dot-product attention. float32 in and out regardless of mode.
//
//   q, k, v, out  [B, H, S, head_dim], contiguous, device pointers
//   mask          optional [B,H,S,S] bool, true == may attend. ms holds its
//                 four strides, which may be 0 on broadcast dimensions.
//                 Pass nullptr for no mask.
//   is_causal     lower-triangular masking. Never combined with mask.
//   mode          operand precision; see MathMode.
//
// Returns false without launching anything when this head_dim or mode has no
// specialisation compiled, so the caller can fall back.
bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms,
            float* out, int B, int H, int S, int head_dim,
            bool is_causal, float scale, MathMode mode, cudaStream_t stream);

}  // namespace tile_attn
