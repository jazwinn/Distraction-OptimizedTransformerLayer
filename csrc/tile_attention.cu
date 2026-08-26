// Fused attention on the CUDA tile programming model (cuTile).
//
// This is a third implementation of the same math already covered by the
// scalar and wmma kernels in fused_attention.cu. The difference is the level
// the code is written at:
//
//   scalar / wmma   the kernel is written per *thread*. The author picks the
//                   block size, stages K/V through shared memory by hand,
//                   pads leading dimensions to dodge bank conflicts, places
//                   __syncthreads(), and drives the MMA units through
//                   explicit 16x16x8 fragments.
//
//   tile (here)     the kernel is written per *block*, once. A tile is a
//                   fixed-size array that the whole block owns collectively;
//                   `ct::matmul` is a matrix multiply of two such arrays, not
//                   a fragment dance. Register/shared allocation, the load
//                   schedule, bank-conflict avoidance and the intra-block
//                   synchronisation are all the compiler's job. There is no
//                   threadIdx in this file, and the launch uses one "thread"
//                   per block because the block *is* the unit of work.
//
// The algorithm is still FlashAttention: the [B,H,S,S] score matrix is never
// materialised in global memory, K/V are streamed a tile at a time, and the
// softmax is accumulated online with a running max and running sum.
//
// Scope: float32 only, head_dim in {8,16,32,64}. half/bfloat16 fall back to
// the existing kernels -- cuTile's matmul accumulates __half in __half, which
// would not hold the accuracy budget the harness checks against, and the
// tf32/bf16 paths that do accumulate in float are already what wmma covers.
//
// Requires CUDA 13.3+ (for <cuda_tile.h>), -std=c++20 and -enable-tile.
// TRANSFORMER_HAVE_TILE is defined by the build only when all of that is
// present; without it this file still compiles, into a launcher that reports
// "not available" so the caller falls back.

#include "tile_attention.h"

#ifdef TRANSFORMER_HAVE_TILE

#include <cuda_tile.h>

namespace ct = cuda::tiles;

namespace {

// Finite stand-in for -inf on masked-out scores. A true -inf would make the
// rescale term (m_run - m_new) evaluate to nan on a row where *every* key is
// masked, and that nan then spreads through the accumulator. A large negative
// float keeps every intermediate finite; the final reciprocal is where a dead
// row is turned into zeros.
constexpr float NEG = -1e30f;
constexpr float NEG_THRESH = -1e29f;

// Masking modes. Causal and an explicit mask are never combined -- the torch
// wrapper folds causal into attn_mask when a caller wants both -- so these are
// three separate specialisations rather than two independent flags.
enum class MaskMode { None, Causal, Explicit };

template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MaskMode MODE>
__tile_global__ void tile_attention_kernel(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const bool* __restrict__ mask,
                                           long long ms0, long long ms1,
                                           long long ms2, long long ms3,
                                           float* __restrict__ out,
                                           int B, int H, int S,
                                           float scale) {
    using OTile   = ct::tile<float, ct::shape<BLOCK_M, HEAD_DIM>>;
    using STile   = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using RowTile = ct::tile<float, ct::shape<BLOCK_M, 1>>;
    using RowBool = ct::tile<bool, ct::shape<BLOCK_M, 1>>;
    using MaskTile = ct::tile<bool, ct::shape<BLOCK_M, BLOCK_N>>;
    // Index tiles are int64 so they can be scaled by mask strides directly,
    // with no element-type conversion in the inner loop.
    using RowIdx = ct::tile<long long, ct::shape<BLOCK_M, 1>>;
    using ColIdx = ct::tile<long long, ct::shape<1, BLOCK_N>>;

    const auto bid = ct::bid();
    const int m_tile = static_cast<int>(bid.x);
    const int h      = static_cast<int>(bid.y);
    const int b      = static_cast<int>(bid.z);

    // [B,H,S,head_dim] is contiguous, so one (b,h) slice is a flat [S,head_dim]
    // matrix and the views below need no stride gymnastics.
    const long long bh_off = static_cast<long long>(b * H + h) * S * HEAD_DIM;

    auto q_view = ct::partition_view{ct::tensor_span{q + bh_off, ct::extents{S, HEAD_DIM}},
                                     ct::shape<BLOCK_M, HEAD_DIM>{}};
    auto k_view = ct::partition_view{ct::tensor_span{k + bh_off, ct::extents{S, HEAD_DIM}},
                                     ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto v_view = ct::partition_view{ct::tensor_span{v + bh_off, ct::extents{S, HEAD_DIM}},
                                     ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto o_view = ct::partition_view{ct::tensor_span{out + bh_off, ct::extents{S, HEAD_DIM}},
                                     ct::shape<BLOCK_M, HEAD_DIM>{}};

    // Q is read once and reused against every key tile. load_masked zero-fills
    // rows past S; those rows are computed but never stored.
    auto qq = q_view.load_masked(m_tile, 0);

    auto acc   = ct::zeros<OTile>();
    auto m_run = ct::full<RowTile>(NEG);
    auto l_run = ct::zeros<RowTile>();

    auto row_idx = ct::iota<RowIdx>() + static_cast<long long>(m_tile) * BLOCK_M;

    // Under causal masking no query in this block looks past the block's own
    // last row, so whole key tiles beyond it are skipped rather than computed
    // and discarded. This is where the causal speedup comes from.
    const int causal_end = m_tile * BLOCK_M + BLOCK_M;
    const int key_limit =
        (MODE == MaskMode::Causal) ? (causal_end < S ? causal_end : S) : S;

    for (int kt = 0; kt < key_limit; kt += BLOCK_N) {
        auto kk = k_view.load_masked(kt / BLOCK_N, 0);
        auto vv = v_view.load_masked(kt / BLOCK_N, 0);

        // S = Q @ K^T, scaled. One expression per block; no fragments.
        auto s = ct::matmul(qq, ct::transpose(kk)) * scale;

        // The bounds test alone is [1, BLOCK_N]. AND-ing an all-true column
        // widens it to [BLOCK_M, BLOCK_N] up front, so the causal/explicit
        // terms below refine a tile of the right shape instead of changing it.
        auto col_idx = ct::iota<ColIdx>() + static_cast<long long>(kt);
        auto valid = ct::full<RowBool>(true) && (col_idx < static_cast<long long>(S));

        if constexpr (MODE == MaskMode::Causal) {
            valid = valid && (col_idx <= row_idx);
        } else if constexpr (MODE == MaskMode::Explicit) {
            // The mask arrives already expanded to [B,H,S,S], so a [B,1,1,S]
            // or [B,1,S,S] mask reaches us as stride-0 dimensions. That rules
            // out a partition_view, whose indexing assumes a dense tile grid;
            // gathering through a pointer tile is what copes with an arbitrary
            // stride pattern. `valid` gates the gather, so out-of-range rows
            // and columns are never dereferenced.
            const bool* mask_bh = mask + b * ms0 + h * ms1;
            auto ptrs = mask_bh + (row_idx * ms2 + col_idx * ms3);
            valid = valid && ct::load_masked(ptrs, valid, ct::full<MaskTile>(false));
        }

        s = ct::select(valid, s, ct::full<STile>(NEG));

        // Online softmax. Unlike the scalar kernel, which rescales only when a
        // new max actually appears, the tile form rescales every iteration:
        // the whole [BLOCK_M, BLOCK_N] tile is evaluated as one expression, so
        // a data-dependent branch per row would not buy anything.
        auto m_new = ct::max(m_run, ct::reduce_max<1>(s));
        auto corr  = ct::exp(m_run - m_new);
        auto p     = ct::exp(s - m_new);

        l_run = l_run * corr + ct::sum<1>(p);
        acc   = acc * corr + ct::matmul(p, vv);
        m_run = m_new;
    }

    // A row whose every key was masked still holds the sentinel max. The
    // reference would produce nan there; emit zeros instead, matching what the
    // scalar and wmma kernels do, since such rows are zero-filled downstream
    // anyway and a nan would only risk contaminating something else.
    auto inv = ct::select(m_run > NEG_THRESH,
                          ct::full<RowTile>(1.0f) / l_run,
                          ct::zeros<RowTile>());
    o_view.store_masked(acc * inv, m_tile, 0);
}

// Block shape per head_dim, measured on SM 8.6 (see the sweep in the commit
// that added this file). The kernel holds Q, O, K, V and the score tile live
// at once, so the footprint goes as BLOCK_M*HEAD_DIM*2 + BLOCK_N*HEAD_DIM*2 +
// BLOCK_M*BLOCK_N; past a threshold the compiler starts spilling and the cost
// jumps by an order of magnitude rather than degrading smoothly. On this card
// the cliff sits between the entries below and their immediate neighbours --
// at head_dim 64, BLOCK_N 16 runs at 1.5 ms where BLOCK_N 32 runs at 10.9 ms.
// These are worth re-measuring on a different architecture.
template <int HEAD_DIM> struct BlockCfg;
template <> struct BlockCfg<8>  { static constexpr int M = 64; static constexpr int N = 64; };
template <> struct BlockCfg<16> { static constexpr int M = 32; static constexpr int N = 16; };
template <> struct BlockCfg<32> { static constexpr int M = 64; static constexpr int N = 16; };
template <> struct BlockCfg<64> { static constexpr int M = 32; static constexpr int N = 16; };

template <int HEAD_DIM>
void launch_for_head_dim(const float* q, const float* k, const float* v,
                         const bool* mask, const long long* ms,
                         float* out, int B, int H, int S,
                         bool is_causal, float scale, cudaStream_t stream) {
    constexpr int BLOCK_M = BlockCfg<HEAD_DIM>::M;
    constexpr int BLOCK_N = BlockCfg<HEAD_DIM>::N;

    // One "thread" per block: a tile kernel's body runs once per block and the
    // compiler decides how many real threads carry it.
    const dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    const dim3 block(1);

    if (is_causal) {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::Causal>
            <<<grid, block, 0, stream>>>(q, k, v, nullptr, 0, 0, 0, 0,
                                         out, B, H, S, scale);
    } else if (mask != nullptr) {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::Explicit>
            <<<grid, block, 0, stream>>>(q, k, v, mask, ms[0], ms[1], ms[2], ms[3],
                                         out, B, H, S, scale);
    } else {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::None>
            <<<grid, block, 0, stream>>>(q, k, v, nullptr, 0, 0, 0, 0,
                                         out, B, H, S, scale);
    }
}

}  // namespace

namespace tile_attn {

bool available() { return true; }

bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms,
            float* out, int B, int H, int S, int head_dim,
            bool is_causal, float scale, cudaStream_t stream) {
    switch (head_dim) {
        case 8:
            launch_for_head_dim<8>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
            return true;
        case 16:
            launch_for_head_dim<16>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
            return true;
        case 32:
            launch_for_head_dim<32>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
            return true;
        case 64:
            launch_for_head_dim<64>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
            return true;
        default:
            return false;
    }
}

}  // namespace tile_attn

#else  // !TRANSFORMER_HAVE_TILE

// Built without tile support (CUDA < 13.3, or no -enable-tile). The symbols
// still exist so fused_attention.cu links either way; they just decline.
namespace tile_attn {

bool available() { return false; }

bool launch(const float*, const float*, const float*,
            const bool*, const long long*,
            float*, int, int, int, int, bool, float, cudaStream_t) {
    return false;
}

}  // namespace tile_attn

#endif  // TRANSFORMER_HAVE_TILE
