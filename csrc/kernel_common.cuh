// Pieces shared by more than one kernel in this extension.
//
// Everything in these headers lives in an unnamed namespace. The extension is
// one translation unit -- fused_attention.cu includes every .cuh here, and
// tile_attention.cu is compiled separately and includes none of them -- so
// internal linkage costs nothing, keeps the kernels out of the .pyd's export
// table, and lets nvcc see every launcher and its kernel together.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace {

namespace wm = nvcuda::wmma;

// The arithmetic a kernel contracts q/k/v in, independent of what dtype the
// tensors themselves are. The two used to be tangled together: `tile-fp16` named
// a kernel and a precision at once, while wmma's precision lived in a separate
// --attn-fp16 flag that no other backend shared. They are orthogonal, so they
// are two arguments now.
//
// Not every pair exists, and the ones that do not are refused rather than
// quietly rounded to a neighbour:
//
//              Fp32           Tf32          Fp16          Bf16
//   scalar     yes            -             -             -
//   wmma       -              yes           yes           yes (testing only)
//   tile       yes            yes           yes           yes
//
// wmma has no Fp32 because no shipped tensor core does a full fp32 matmul --
// asking for it there is a category error, not a missing instantiation. scalar
// has only Fp32 because its accumulators are float and that is the whole point
// of it; feeding it narrower TENSORS is a separate axis (--dtype) and works.
enum class AttnPrecision : int64_t {
    Auto = 0,   // each kernel's own preference -- wmma fp16, scalar and tile fp32
    Fp32 = 1,
    Tf32 = 2,
    Fp16 = 3,
    Bf16 = 4,
};

inline const char* precision_name(AttnPrecision p) {
    switch (p) {
        case AttnPrecision::Auto: return "auto";
        case AttnPrecision::Fp32: return "fp32";
        case AttnPrecision::Tf32: return "tf32";
        case AttnPrecision::Fp16: return "fp16";
        case AttnPrecision::Bf16: return "bf16";
    }
    return "?";
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

}  // namespace
