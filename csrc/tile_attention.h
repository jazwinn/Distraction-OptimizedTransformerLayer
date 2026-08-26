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

// True when this build actually contains the tile kernel (CUDA 13.3+ and
// -enable-tile). False builds still link; launch() just declines.
bool available();

// Fused scaled-dot-product attention, float32 only.
//
//   q, k, v, out  [B, H, S, head_dim], contiguous, device pointers
//   mask          optional [B,H,S,S] bool, true == may attend. ms holds its
//                 four strides, which may be 0 on broadcast dimensions.
//                 Pass nullptr for no mask.
//   is_causal     lower-triangular masking. Never combined with mask.
//
// Returns false without launching anything when head_dim has no
// specialisation compiled, so the caller can fall back.
bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms,
            float* out, int B, int H, int S, int head_dim,
            bool is_causal, float scale, cudaStream_t stream);

}  // namespace tile_attn
