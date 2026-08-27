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
// Scope: float32 tensors in and out, head_dim in {8,16,32,64}. The MathMode
// parameter narrows only the two GEMMs' operands, which is what decides
// whether the tensor cores run: Fp32 stays on the CUDA cores everywhere and is
// exact; Tf32 reaches the MMA units keeping fp32's exponent range and 10 of
// its 23 mantissa bits (~1e-3, the same arithmetic cuBLAS gives the baseline
// under allow_tf32); Bf16 reaches them with 8 mantissa bits and costs ~4
// orders of magnitude. There is no Fp16 mode -- cuTile accumulates a __half
// matmul into __half, and this kernel sums hundreds of products per output.
//
// Verify which units a mode actually got with:
//     cuobjdump -sass build/tile_attention.cuda.o | grep HMMA
// Fp32 kernels contain none; Tf32 emits HMMA.1688.F32.TF32 and Bf16
// HMMA.16816.F32.BF16.
//
// Requires CUDA 13.3+ (for <cuda_tile.h>), -std=c++20 and -enable-tile.
// TRANSFORMER_HAVE_TILE is defined by the build only when all of that is
// present; without it this file still compiles, into a launcher that reports
// "not available" so the caller falls back.

#include "tile_attention.h"

#ifdef TRANSFORMER_HAVE_TILE

#include <cuda_tile.h>
// crt/cuda_tile.h (which <cuda_tile.h> is a one-line wrapper around) contains
// no #includes at all: it forward-declares __half, __nv_bfloat16, __nv_fp8_*
// and __nv_tf32 and leaves completing them to whoever wants to instantiate a
// tile of one. So each narrow operand type needs its defining header pulled in
// here, and that is the *only* thing standing between a mode and the MMA units.
#include <cuda_bf16.h>

// __nv_tf32 is defined in <cuda_tf32.h> (a 4-byte struct, gated only on
// __cplusplus) and has been since CUDA 13.3 -- the same toolkit that first
// ships <cuda_tile.h>, so in practice it is always there. It is guarded anyway
// because it was added separately from the tile header and a future toolkit
// could rename it. Including it defines __CUDA_TF32_TYPES_EXIST__, which is
// what turns the Tf32 mode on below; -DTILE_HAVE_TF32 still forces it.
#if __has_include(<cuda_tf32.h>)
#include <cuda_tf32.h>
#endif
#if !defined(TILE_HAVE_TF32) && defined(__CUDA_TF32_TYPES_EXIST__)
#define TILE_HAVE_TF32 1
#endif

namespace ct = cuda::tiles;

using tile_attn::MathMode;

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

// The element type each GEMM's operands are cast to. Only this changes between
// math modes -- the accumulator stays fp32 in all of them, which cuTile
// guarantees for float, bf16 and tf32 operands (and pointedly does not for
// __half, which accumulates into __half; that is why there is no Fp16 mode).
template <MathMode MODE> struct Operand;
template <> struct Operand<MathMode::Fp32> { using type = float; };
template <> struct Operand<MathMode::Bf16> { using type = __nv_bfloat16; };
#ifdef TILE_HAVE_TF32
template <> struct Operand<MathMode::Tf32> { using type = __nv_tf32; };
#endif

template <MathMode MODE>
constexpr bool mode_compiled =
    MODE == MathMode::Fp32 || MODE == MathMode::Bf16
#ifdef TILE_HAVE_TF32
    || MODE == MathMode::Tf32
#endif
    ;

// Narrows a tile to the mode's operand type. For Fp32 this is the identity, so
// that path emits exactly the code it did before math modes existed.
template <MathMode MODE, size_t R, size_t C, typename In>
__tile__ auto as_operand(In x) {
    using E = typename Operand<MODE>::type;
    if constexpr (ct::same_as<E, float>) {
        return x;
    } else {
        return ct::tile<E, ct::shape<R, C>>(x);
    }
}

template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MaskMode MODE, MathMode MATH>
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
    const int h = static_cast<int>(bid.y);
    const int b = static_cast<int>(bid.z);

    // Under causal masking a block's cost depends on where it sits: block m
    // walks m+1 key tiles. Blocks are dispatched in roughly increasing linear
    // index and grid.x varies fastest, so the natural mapping hands out the
    // cheapest blocks first and leaves the most expensive ones for the final
    // wave, where they alone set the makespan. Reversing it is
    // longest-processing-time-first: the expensive blocks start immediately and
    // the cheap ones fill in around them. The dense kernel has no such spread,
    // so it keeps the identity mapping.
    //
    // Worth knowing before trusting that reasoning: A/B'd against the identity
    // mapping, both compiled and timed interleaved in one process, it is close
    // to a wash. It wins where the spread is widest and loses in the middle:
    //
    //   seq 2048 causal head_dim 32   0.411 -> 0.387   1.06x
    //   seq 2048 causal head_dim 64   0.999 -> 0.960   1.04x
    //   seq 2048 causal head_dim 16   0.223 -> 0.221   1.01x
    //   seq 128  causal head_dim 64   0.083 -> 0.083   1.00x
    //   seq 512  causal head_dim 64   0.301 -> 0.317   0.95x
    //
    // About 1% on the geometric mean, positive on absolute milliseconds because
    // the wins land on the long cases. Kept for that, not for the theory; three
    // lines to drop if the seq 512 regression ever matters more.
    //
    // n_m is the same ceil(S / BLOCK_M) the launcher used for grid.x -- both S
    // and BLOCK_M are already here, so this costs no extra kernel argument.
    const int n_m = (S + BLOCK_M - 1) / BLOCK_M;
    const int m_tile = (MODE == MaskMode::Causal) ? (n_m - 1 - static_cast<int>(bid.x))
                                                  : static_cast<int>(bid.x);

    // [B,H,S,head_dim] is contiguous, so one (b,h) slice is a flat [S,head_dim]
    // matrix and the views below need no stride gymnastics.
    const long long bh_off = static_cast<long long>(b * H + h) * S * HEAD_DIM;

    // ct::extents deduces a *static* extent from a ct::integral_constant and a
    // dynamic one from a plain int, so `ct::extents{S, HEAD_DIM}` -- HEAD_DIM
    // being an ordinary int template parameter -- gave both extents a runtime
    // value and put the row stride in a register. Handing head_dim over as an
    // integral_constant makes it a compile-time power of two instead. Measured
    // neutral on CUDA 13.3 / sm_86 (4462 SASS instructions either way, so the
    // tile compiler was already recovering it), kept because it is the honest
    // description of the tensor and costs nothing.
    constexpr ct::integral_constant<HEAD_DIM> HD{};
    auto q_view = ct::partition_view{ct::tensor_span{q + bh_off, ct::extents{S, HD}},
                                     ct::shape<BLOCK_M, HEAD_DIM>{}};
    // K is read only as K^T. Rather than loading a [BLOCK_N, HEAD_DIM] tile and
    // calling ct::transpose on it -- which is a real shuffle through shared
    // memory, once per key tile -- describe the same bytes as a column-major
    // [HEAD_DIM, S] matrix. layout_left puts stride 1 on dim 0 and HEAD_DIM on
    // dim 1, so element (d, j) is k[j * HEAD_DIM + d]: exactly K^T, for free.
    // This is the tile-level equivalent of the wmma kernel loading k_s into a
    // col_major matrix_b fragment instead of transposing it.
    auto kt_view = ct::partition_view{ct::tensor_span{k + bh_off, ct::extents{HD, S},
                                                      ct::layout_left{}},
                                      ct::shape<HEAD_DIM, BLOCK_N>{}};
    auto v_view = ct::partition_view{ct::tensor_span{v + bh_off, ct::extents{S, HD}},
                                     ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto o_view = ct::partition_view{ct::tensor_span{out + bh_off, ct::extents{S, HD}},
                                     ct::shape<BLOCK_M, HEAD_DIM>{}};

    // Q is read once and reused against every key tile. load_masked zero-fills
    // rows past S; those rows are computed but never stored.
    //
    // The narrowing cast to the MATH mode's operand type is hoisted here rather
    // than left in the key loop: Q does not change across key tiles, and under
    // tf32 that cast is expensive (see as_operand). The wmma kernel converts
    // q_frag to tf32 once before its key loop for the same reason.
    //
    // The scale is deliberately *not* folded in here. Scaling Q changes what
    // gets rounded to tf32 and measurably widened the error (3.1e-4 -> 6.2e-4
    // on the 8x8x128x64 case), for no measurable speed; it stays on the fp32
    // score tile below, where the rounding is already done.
    auto q_op = as_operand<MATH, BLOCK_M, HEAD_DIM>(q_view.load_masked(m_tile, 0));

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
        auto ktt = kt_view.load_masked(0, kt / BLOCK_N);
        auto vv  = v_view.load_masked(kt / BLOCK_N, 0);

        // S = Q @ K^T. One expression per block; no fragments. The scale and
        // the base change are already in q_op, and K^T came out of the view
        // that way, so the only work left here is the cast that puts the
        // operands on the tensor cores.
        // exp(x) is evaluated as exp2(x * log2e), so folding log2e into the
        // scale that is applied here anyway removes one multiply per score
        // element per key tile. It is applied to the fp32 result of the GEMM,
        // not to Q before the narrowing cast: pre-scaling Q changes what gets
        // rounded to tf32 and measurably widens the error.
        auto s = ct::matmul(q_op, as_operand<MATH, HEAD_DIM, BLOCK_N>(ktt))
                 * (scale * 1.4426950408889634f);

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
        auto corr  = ct::exp2(m_run - m_new);
        auto p     = ct::exp2(s - m_new);

        l_run = l_run * corr + ct::sum<1>(p);
        // ct::mma(A, B, C) is a fused multiply-accumulate into C, the tile
        // equivalent of wmma's mma_sync(o, p, v, o). Written as
        // `acc * corr + ct::matmul(...)` the product had to be materialised
        // into its own [BLOCK_M, HEAD_DIM] tile and then added, which is a
        // whole extra pass over the accumulator per key tile.
        acc = ct::mma(as_operand<MATH, BLOCK_M, BLOCK_N>(p),
                      as_operand<MATH, BLOCK_N, HEAD_DIM>(vv),
                      acc * corr);
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

// Block shape per (head_dim, math mode), measured on SM 8.6.
//
// The kernel holds Q, O, K, V and the score tile live at once, so the footprint
// goes as BLOCK_M*HEAD_DIM*2 + BLOCK_N*HEAD_DIM*2 + BLOCK_M*BLOCK_N. Past a
// threshold the compiler spills and the cost jumps an order of magnitude rather
// than degrading smoothly -- at head_dim 64 in Fp32, BLOCK_N 16 runs at 1.5 ms
// where BLOCK_N 32 runs at 10.9 ms.
//
// A narrow mode halves the operand width, which moves that cliff rather than
// just shifting the curve: the head_dim 64 that wants 32x16 in Fp32 runs best
// at 64x64 in Bf16 -- the shape that was *worst* in Fp32. So these are per
// mode, not merely per head_dim.
//
// Measured on SM 8.6. On a part with working TMA the load cost changes and the
// cliff will sit elsewhere; re-measure rather than trusting these.
// Swept block shapes for the tf32 mode, overridable from the build line so
// scripts/tune_tile_tf32.py can search them without editing this file.
//
// Measured on an RTX 3070 (SM 8.6), best-of-5 interleaved rounds over six shapes
// spanning seq_len 128/512/2048, causal and dense:
//
//   head_dim 8    128x64    0.585 ms   (next 64x64 0.587 -- within noise)
//   head_dim 16   128x32    0.751 ms   (next 64x64 0.779, 1.04x)
//   head_dim 32   128x64    1.419 ms   (next 128x32 1.469, 1.04x)
//   head_dim 64   128x32    2.855 ms   (next 64x64 3.151, 1.10x)
//
// BLOCK_M is 128 in all four, which is exactly what FlashAttention-2 does:
// tile_size_fwd_sm8x returns kBlockM=128 unconditionally for every head_dim,
// arch and dtype. Only BLOCK_N moves. Reaching for that table first would have
// been cheaper than searching a 16x16..128x128 grid blind.
//
// Every extent must be a power of two -- cuTile enforces is_pow2 per dimension
// (crt/cuda_tile.h:749). FA2's kBlockN for the headdim<=64 bucket that all four
// of these fall into is 112, which cannot be expressed here at all; 64 and 128
// are its legal neighbours, and 128 loses badly at head_dim 64 (11.6 ms).
//
// Two traps, both of which caught this kernel once:
//
//   Never compare timings across runs. Run-to-run variance here was large
//   enough to invert the ranking outright: a cross-run comparison "showed"
//   128x128 beating 128x32 by 1.58x at head_dim 8, when timed interleaved in
//   one process it is 4th of 5. Rank candidates only within a single run, and
//   re-measure the incumbent alongside any challenger.
//
//   Score short and long sequences together. Summing raw ms weights seq_len
//   2048 about 10x over seq_len 128, so a shape that tanks short sequences can
//   still win the sum. An early pass scored only 512/2048 and regressed
//   seq_len 128 by ~20%.
//
// What block shape cannot fix: at seq_len 128 a 128-row block leaves only
// batch*heads blocks in the grid, and no choice here fills 46 SMs. FlashAttention
// keeps kBlockM at 128 regardless and solves that by splitting the key dimension
// across blocks (Flash-Decoding), reducing the partials afterwards, with a
// heuristic that stops splitting once ~80% of SMs are busy. That is a structural
// change to this kernel, not a tuning parameter, and is not implemented here --
// it is why tile-tf32 still trails wmma at seq_len 128.
#ifndef TF32_M_8
#define TF32_M_8  128
#define TF32_N_8  64
#endif
#ifndef TF32_M_16
#define TF32_M_16 128
#define TF32_N_16 32
#endif
#ifndef TF32_M_32
#define TF32_M_32 128
#define TF32_N_32 64
#endif
#ifndef TF32_M_64
#define TF32_M_64 128
#define TF32_N_64 32
#endif

// ...and again for the causal kernel, which wants a different shape.
//
// Causal masking makes a block's cost depend on where it sits in the sequence:
// block m walks m+1 key tiles, not S/BLOCK_N of them. A 128-row block is
// therefore doing two things at once -- it halves the number of blocks
// available to fill 46 SMs, and it doubles the *spread* between the cheapest
// and the most expensive block, so the tail of the grid is longer. Both push
// the optimum towards a smaller BLOCK_M than the dense kernel wants, and
// measurably so: at head_dim 64, seq_len 2048 causal, 64x64 runs 0.867 ms where
// the dense winner 128x32 runs 0.981; at seq_len 128 causal it is 0.082 against
// 0.110. Tuning one shape across both mask modes gave up that much.
//
// Swept by scripts/tune_tile_tf32.py --causal, which scores only causal cases.
#ifndef TF32_CM_8
#define TF32_CM_8  128
#define TF32_CN_8  64
#endif
#ifndef TF32_CM_16
#define TF32_CM_16 128
#define TF32_CN_16 32
#endif
#ifndef TF32_CM_32
#define TF32_CM_32 128
#define TF32_CN_32 64
#endif
#ifndef TF32_CM_64
#define TF32_CM_64 64
#define TF32_CN_64 64
#endif

// ...and once more for a dense grid too small to fill the device.
//
// BLOCK_M sets how many blocks there are: ceil(S/BLOCK_M) * H * B. At batch 8,
// heads 8, seq_len 128 and BLOCK_M 128 that is 64 blocks on a 46-SM card --
// 1.4 waves, so the second wave runs 18 blocks against 46 slots and a third of
// the device idles for half the kernel. Halving BLOCK_M doubles the block count
// and fills it. Measured at head_dim 64, seq_len 128 dense, one interleaved
// run: 64x32 = 0.096 ms against the long-sequence winner 128x32 = 0.109. The
// same shape loses at seq_len 512 (0.474 vs 0.376), which is why this is a
// launch-time choice rather than a retune.
//
// This is also the whole realistic upside of split-KV (Flash-Decoding), which
// attacks the same idle SMs by giving each block a slice of the key range
// instead of a slice of the query range. Split-KV is the better lever when K/V
// traffic dominates Q -- long sequences with few heads -- because it does not
// replicate the Q tile per block. At the short sequences where this grid is
// actually starved, Q and K/V are the same size and the two degenerate to the
// same trade, so the cheap one wins.
//
// Only head_dim 64 has been measured. The others default to their dense shape,
// which makes have_short_shape below false and costs no extra instantiation.
#ifndef TF32_SM_8
#define TF32_SM_8  TF32_M_8
#define TF32_SN_8  TF32_N_8
#endif
#ifndef TF32_SM_16
#define TF32_SM_16 TF32_M_16
#define TF32_SN_16 TF32_N_16
#endif
#ifndef TF32_SM_32
#define TF32_SM_32 TF32_M_32
#define TF32_SN_32 TF32_N_32
#endif
#ifndef TF32_SM_64
#define TF32_SM_64 64
#define TF32_SN_64 32
#endif

// CAUSAL is a separate axis because the causal kernel's grid is triangular;
// see the TF32_CM_* note above. Only tf32 has been swept both ways -- fp32 and
// bf16 use one shape for both, which is what they were measured at.
template <int HEAD_DIM, MathMode MODE, bool CAUSAL = false> struct BlockCfg;

template <bool C> struct BlockCfg<8,  MathMode::Fp32, C> { static constexpr int M = 64; static constexpr int N = 64; };
template <bool C> struct BlockCfg<16, MathMode::Fp32, C> { static constexpr int M = 32; static constexpr int N = 16; };
template <bool C> struct BlockCfg<32, MathMode::Fp32, C> { static constexpr int M = 64; static constexpr int N = 16; };
template <bool C> struct BlockCfg<64, MathMode::Fp32, C> { static constexpr int M = 32; static constexpr int N = 16; };

template <bool C> struct BlockCfg<8,  MathMode::Bf16, C> { static constexpr int M = 64; static constexpr int N = 64; };
template <bool C> struct BlockCfg<16, MathMode::Bf16, C> { static constexpr int M = 64; static constexpr int N = 64; };
template <bool C> struct BlockCfg<32, MathMode::Bf16, C> { static constexpr int M = 64; static constexpr int N = 64; };
template <bool C> struct BlockCfg<64, MathMode::Bf16, C> { static constexpr int M = 64; static constexpr int N = 64; };

#ifdef TILE_HAVE_TF32
// tf32 is the one mode where the two pressures pull apart. It occupies the same
// 32 bits as fp32, so the spill cliff sits where fp32's does -- but it runs on
// the MMA units, whose 16x8x8 shape the narrow fp32 tiles (BLOCK_N 16) starve:
// at head_dim 16 the inherited 32x16 emitted 4 HMMA where bf16's 64x64 emitted
// 16. Inheriting either mode's shapes is therefore wrong, and these were swept
// on SM 8.6 rather than derived. See the fp32 note above on re-measuring.
template <> struct BlockCfg<8,  MathMode::Tf32, false> { static constexpr int M = TF32_M_8;  static constexpr int N = TF32_N_8;  };
template <> struct BlockCfg<8,  MathMode::Tf32, true>  { static constexpr int M = TF32_CM_8; static constexpr int N = TF32_CN_8; };
template <> struct BlockCfg<16, MathMode::Tf32, false> { static constexpr int M = TF32_M_16;  static constexpr int N = TF32_N_16;  };
template <> struct BlockCfg<16, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_16; static constexpr int N = TF32_CN_16; };
template <> struct BlockCfg<32, MathMode::Tf32, false> { static constexpr int M = TF32_M_32;  static constexpr int N = TF32_N_32;  };
template <> struct BlockCfg<32, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_32; static constexpr int N = TF32_CN_32; };
template <> struct BlockCfg<64, MathMode::Tf32, false> { static constexpr int M = TF32_M_64;  static constexpr int N = TF32_N_64;  };
template <> struct BlockCfg<64, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_64; static constexpr int N = TF32_CN_64; };
#endif

// The dense shape to use when the grid is too small to fill the device. Only
// tf32 has been measured; every other mode reports its dense shape, which turns
// the short-grid path off for it entirely.
template <int HEAD_DIM, MathMode MODE> struct ShortCfg {
    static constexpr int M = BlockCfg<HEAD_DIM, MODE, false>::M;
    static constexpr int N = BlockCfg<HEAD_DIM, MODE, false>::N;
};
#ifdef TILE_HAVE_TF32
template <> struct ShortCfg<8,  MathMode::Tf32> { static constexpr int M = TF32_SM_8;  static constexpr int N = TF32_SN_8;  };
template <> struct ShortCfg<16, MathMode::Tf32> { static constexpr int M = TF32_SM_16; static constexpr int N = TF32_SN_16; };
template <> struct ShortCfg<32, MathMode::Tf32> { static constexpr int M = TF32_SM_32; static constexpr int N = TF32_SN_32; };
template <> struct ShortCfg<64, MathMode::Tf32> { static constexpr int M = TF32_SM_64; static constexpr int N = TF32_SN_64; };
#endif

// SM count of the current device, queried once per device rather than per
// launch: the short-grid test below runs on every call, and a driver round trip
// is a measurable share of a 60 us kernel.
inline int sm_count() {
    constexpr int kMaxDevices = 16;
    static int cached[kMaxDevices] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) {
        dev = 0;
    }
    if (cached[dev] == 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            // Never let a failed query change behaviour: a count of 1 makes the
            // threshold unreachable, so the long-grid shape always wins, which
            // is what this did before the short shape existed.
            n = 1;
        }
        cached[dev] = n;
    }
    return cached[dev];
}

// The two dense mask modes at a fixed block shape. Factored out because the
// shape is chosen at run time now, so this pair of launches is needed twice and
// must not drift apart.
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MathMode MATH>
void launch_dense(const float* q, const float* k, const float* v,
                  const bool* mask, const long long* ms,
                  float* out, int B, int H, int S,
                  float scale, cudaStream_t stream) {
    const dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    const dim3 block(1);
    if (mask != nullptr) {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::Explicit, MATH>
            <<<grid, block, 0, stream>>>(q, k, v, mask, ms[0], ms[1], ms[2], ms[3],
                                         out, B, H, S, scale);
    } else {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::None, MATH>
            <<<grid, block, 0, stream>>>(q, k, v, nullptr, 0, 0, 0, 0,
                                         out, B, H, S, scale);
    }
}

template <int HEAD_DIM, MathMode MATH>
void launch_for_head_dim(const float* q, const float* k, const float* v,
                         const bool* mask, const long long* ms,
                         float* out, int B, int H, int S,
                         bool is_causal, float scale, cudaStream_t stream) {
    if (is_causal) {
        // The causal kernel already runs a smaller BLOCK_M (see TF32_CM_*), so
        // its grid is twice as long to begin with and the short-grid test below
        // would almost never fire. It keeps one shape.
        using Cfg = BlockCfg<HEAD_DIM, MATH, true>;
        const dim3 grid((S + Cfg::M - 1) / Cfg::M, H, B);
        const dim3 block(1);  // one "thread" per block: the body runs once per
                              // block, and the compiler picks the real threads
        tile_attention_kernel<Cfg::M, Cfg::N, HEAD_DIM, MaskMode::Causal, MATH>
            <<<grid, block, 0, stream>>>(q, k, v, nullptr, 0, 0, 0, 0,
                                         out, B, H, S, scale);
        return;
    }

    using Long  = BlockCfg<HEAD_DIM, MATH, false>;
    using Short = ShortCfg<HEAD_DIM, MATH>;
    // When the two agree there is nothing to choose between, and instantiating
    // the second kernel would only cost compile time.
    constexpr bool have_short_shape = (Short::M != Long::M) || (Short::N != Long::N);

    if constexpr (have_short_shape) {
        // Two waves is the threshold: below it the tail wave runs a partial
        // grid, and the SMs it leaves idle cost more than the extra blocks do.
        // 64 blocks (batch 8, heads 8, seq 128) falls under 2*46 and takes the
        // short shape; 128 blocks (seq 512 and up) does not -- which is exactly
        // where the two shapes swapped places when measured.
        const long long blocks =
            static_cast<long long>((S + Long::M - 1) / Long::M) * H * B;
        if (blocks < 2LL * sm_count()) {
            launch_dense<Short::M, Short::N, HEAD_DIM, MATH>(
                q, k, v, mask, ms, out, B, H, S, scale, stream);
            return;
        }
    }
    launch_dense<Long::M, Long::N, HEAD_DIM, MATH>(
        q, k, v, mask, ms, out, B, H, S, scale, stream);
}

}  // namespace

namespace tile_attn {

bool available() { return true; }

bool supports(MathMode mode) {
    switch (mode) {
        case MathMode::Fp32: return mode_compiled<MathMode::Fp32>;
        case MathMode::Bf16: return mode_compiled<MathMode::Bf16>;
        case MathMode::Tf32: return mode_compiled<MathMode::Tf32>;
    }
    return false;
}

namespace {

// MATH is resolved before head_dim so a mode that was not compiled costs no
// instantiations at all -- head_dim is a template parameter, so every
// (head_dim, mode) pair that reaches here becomes its own compiled kernel.
template <MathMode MATH>
bool launch_mode(const float* q, const float* k, const float* v,
                 const bool* mask, const long long* ms,
                 float* out, int B, int H, int S, int head_dim,
                 bool is_causal, float scale, cudaStream_t stream) {
    if constexpr (!mode_compiled<MATH>) {
        return false;
    } else {
        switch (head_dim) {
            case 8:
                launch_for_head_dim<8, MATH>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
                return true;
            case 16:
                launch_for_head_dim<16, MATH>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
                return true;
            case 32:
                launch_for_head_dim<32, MATH>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
                return true;
            case 64:
                launch_for_head_dim<64, MATH>(q, k, v, mask, ms, out, B, H, S, is_causal, scale, stream);
                return true;
            default:
                return false;
        }
    }
}

}  // namespace

bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms,
            float* out, int B, int H, int S, int head_dim,
            bool is_causal, float scale, MathMode mode, cudaStream_t stream) {
    switch (mode) {
        case MathMode::Fp32:
            return launch_mode<MathMode::Fp32>(q, k, v, mask, ms, out, B, H, S,
                                               head_dim, is_causal, scale, stream);
        case MathMode::Bf16:
            return launch_mode<MathMode::Bf16>(q, k, v, mask, ms, out, B, H, S,
                                               head_dim, is_causal, scale, stream);
        case MathMode::Tf32:
            return launch_mode<MathMode::Tf32>(q, k, v, mask, ms, out, B, H, S,
                                               head_dim, is_causal, scale, stream);
    }
    return false;
}

}  // namespace tile_attn

#else  // !TRANSFORMER_HAVE_TILE

// Built without tile support (CUDA < 13.3, or no -enable-tile). The symbols
// still exist so fused_attention.cu links either way; they just decline.
namespace tile_attn {

bool available() { return false; }
bool supports(MathMode) { return false; }

bool launch(const float*, const float*, const float*,
            const bool*, const long long*,
            float*, int, int, int, int, bool, float, MathMode, cudaStream_t) {
    return false;
}

}  // namespace tile_attn

#endif  // TRANSFORMER_HAVE_TILE
