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
