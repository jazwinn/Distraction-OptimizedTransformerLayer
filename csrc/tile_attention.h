// Interface to the cuTile attention kernel in tile_attention.cu.
//
// Deliberately free of torch and cuda_tile headers: tile_attention.cu needs
// -std=c++20 -enable-tile and CUDA 13.3's <cuda_tile.h>, fused_attention.cu
// needs the torch headers, and this plain-pointer boundary means neither has to
// compile under the other's requirements.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace tile_attn {

// Precision of the two GEMM operands, and the only thing that decides whether
// the tensor cores run at all. The softmax, running max/sum, accumulator and
// the tensors in and out stay fp32 in every mode.
//
// No shipped tensor core does a full fp32 matmul, so Fp32 lands on the CUDA
// cores everywhere; narrowing the operands is what moves work onto the MMA
// units. There is no Fp16 mode: cuTile accumulates a __half matmul into
// __half, and attention sums hundreds of products per output.
enum class MathMode : int {
    Fp32 = 0,  // exact. CUDA cores.    ~1e-6 against an exact reference.
    Bf16 = 1,  // tensor cores.         ~4e-3 -- 8 significand bits.
    Tf32 = 2,  // tensor cores.         ~1e-3 -- see supports().
};

// True when this build actually contains the tile kernel (CUDA 13.3+ and
// -enable-tile). False builds still link; launch() just declines.
bool available();

// True when `mode` was compiled into this build. Tf32 is the one that moves:
// it needs a toolkit whose headers define __nv_tf32 (CUDA 13.3+ ships
// <cuda_tf32.h>), or -DTILE_HAVE_TF32 to force it.
bool supports(MathMode mode);

// Turn the split-KV path on or off for subsequent launches. Starts from the
// TILE_SPLIT_KV environment variable (0 means off) and defaults to on.
//
// Settable at run time so the two paths can be timed against each other inside
// one process -- run-to-run variance on the causal long-sequence cases is
// larger than the effect being measured. See csrc/TUNING.md.
//
// Turning it off makes workspace_bytes() return 0 too, so a caller that queries
// then launches stays consistent. Do not flip it between the two.
void set_split_kv(bool enabled);
bool split_kv_enabled();

// Device scratch launch() wants for this problem, or 0 if it needs none. It is
// non-zero only when the launcher decides to split the key range across blocks
// (Flash-Decoding); the scratch holds the per-split partial outputs and their
// running max / sum until a second pass folds them together.
//
// Query with exactly the arguments you will pass to launch(): the split count
// depends on the block shape, which depends on head_dim, mode and is_causal.
// Both functions resolve it through the same code, so they cannot disagree.
//
// Passing launch() nullptr, or a smaller buffer, costs performance rather than
// correctness -- it falls back to the single-pass kernel.
size_t workspace_bytes(int B, int H, int S, int head_dim,
                       bool is_causal, MathMode mode);

// Fused scaled-dot-product attention. float32 in and out regardless of mode.
//
//   q, k, v, out  [B, H, S, head_dim], contiguous, device pointers
//   mask          optional [B,H,S,S] bool, true == may attend. ms holds its
//                 four strides, which may be 0 on broadcast dimensions.
//                 Pass nullptr for no mask.
//   ws, ws_bytes  device scratch of at least workspace_bytes(); nullptr/0 is
//                 valid and selects the single-pass kernel.
//   is_causal     lower-triangular masking. Never combined with mask.
//   mode          operand precision; see MathMode.
//
// Returns false without launching when this head_dim or mode has no
// specialisation compiled, so the caller can fall back.
bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms,
            float* out, void* ws, size_t ws_bytes,
            int B, int H, int S, int head_dim,
            bool is_causal, float scale, MathMode mode, cudaStream_t stream);

}  // namespace tile_attn
