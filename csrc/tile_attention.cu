// Fused attention on the CUDA tile programming model (cuTile).
//
// Same FlashAttention math as the scalar and wmma kernels in
// fused_attention.cu, written per *block* instead of per *thread*: a tile is a
// fixed-size array the whole block owns, ct::matmul multiplies two of them, and
// register/shared allocation, the load schedule, bank-conflict avoidance and
// intra-block synchronisation are the compiler's job. There is no threadIdx in
// this file, and the launch uses one "thread" per block because the block *is*
// the unit of work.
//
// Scope: float32 in and out, head_dim in {8,16,32,64}. MathMode narrows only
// the two GEMMs' operands -- see tile_attention.h.
//
// Requires CUDA 13.3+, -std=c++20 and -enable-tile. The build defines
// TRANSFORMER_HAVE_TILE only when all of that is present; without it this file
// compiles into a launcher that declines so the caller falls back.
//
// Block shapes and thresholds here are measured, not derived. csrc/TUNING.md
// has the tables and the two rules for re-measuring them.

#include "tile_attention.h"

#ifdef TRANSFORMER_HAVE_TILE

#include <cuda_tile.h>
// crt/cuda_tile.h has no #includes of its own: it forward-declares __half,
// __nv_bfloat16 and __nv_tf32 and leaves completing them to whoever wants a
// tile of one. Pulling in the defining header is the *only* thing standing
// between a narrow mode and the MMA units.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// <cuda_tf32.h> defines __nv_tf32 and has shipped since CUDA 13.3 -- the same
// toolkit that first ships <cuda_tile.h> -- but it was added separately, so it
// is guarded. Including it defines __CUDA_TF32_TYPES_EXIST__, which is what
// turns the Tf32 mode on below; -DTILE_HAVE_TF32 forces it.
#if __has_include(<cuda_tf32.h>)
#include <cuda_tf32.h>
#endif
#if !defined(TILE_HAVE_TF32) && defined(__CUDA_TF32_TYPES_EXIST__)
#define TILE_HAVE_TF32 1
#endif

#include <cstddef>
#include <cstdlib>

namespace ct = cuda::tiles;

using tile_attn::MathMode;

namespace {

// Finite stand-in for -inf on masked-out scores: a true -inf makes the rescale
// term (m_run - m_new) nan on a row where every key is masked, and that nan
// spreads through the accumulator. The final reciprocal is where a dead row
// becomes zeros instead.
constexpr float NEG = -1e30f;
constexpr float NEG_THRESH = -1e29f;

// Masking modes. Causal and an explicit mask are never combined -- the torch
// wrapper folds causal into attn_mask when a caller wants both -- so these are
// three separate specialisations rather than two independent flags.
enum class MaskMode { None, Causal, Explicit };

// The element type each GEMM's operands are cast to. Only this changes between
// math modes: cuTile accumulates float, bf16 and tf32 operands into float.
template <MathMode MODE> struct Operand;
template <> struct Operand<MathMode::Fp32> { using type = float; };
template <> struct Operand<MathMode::Bf16> { using type = __nv_bfloat16; };
template <> struct Operand<MathMode::Fp16> { using type = __half; };
#ifdef TILE_HAVE_TF32
template <> struct Operand<MathMode::Tf32> { using type = __nv_tf32; };
#endif

template <MathMode MODE>
constexpr bool mode_compiled =
    MODE == MathMode::Fp32 || MODE == MathMode::Bf16 || MODE == MathMode::Fp16
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

// SPLIT selects the Flash-Decoding variant: the block covers one slice of the
// key range and writes an *unnormalised* partial -- accumulator, running max,
// running sum -- for split_combine_kernel to fold together, instead of a
// finished output row. One template rather than two kernels because the
// online-softmax recurrence is exactly what must not drift between the paths.
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MaskMode MODE, MathMode MATH,
          bool SPLIT>
__tile_global__ void tile_attention_kernel(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           long long qs0, long long qs1,
                                           long long qs2,
                                           const bool* __restrict__ mask,
                                           long long ms0, long long ms1,
                                           long long ms2, long long ms3,
                                           float* __restrict__ out,
                                           float* __restrict__ part_o,
                                           float* __restrict__ part_m,
                                           float* __restrict__ part_l,
                                           int B, int H, int S,
                                           float scale, int splits) {
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

    // Under causal masking block m walks m+1 key tiles, so reversing the block
    // index is longest-processing-time-first: the expensive blocks start
    // immediately instead of landing in the final wave alone. Worth ~1% on the
    // geometric mean, and it is close to a wash case by case -- see TUNING.md
    // before trusting the reasoning. The dense kernel has no such spread and
    // keeps the identity mapping.
    //
    // n_m is the same ceil(S / BLOCK_M) the launcher used for grid.x, rederived
    // from arguments already here rather than passed in.
    const int n_m = (S + BLOCK_M - 1) / BLOCK_M;

    // Under split-KV grid.x carries both axes: a CUDA grid has only three and
    // y/z are spoken for by heads and batch. `splits` consecutive blocks share
    // one query tile and divide its key range -- adjacent rather than strided,
    // so blocks reading the same Q rows are co-resident and Q is fetched once
    // into L2 per group. m stays the quotient, keeping the ordering above.
    int lane  = static_cast<int>(bid.x);
    int split = 0;
    if constexpr (SPLIT) {
        split = lane % splits;
        lane  = lane / splits;
    }
    const int m_tile = (MODE == MaskMode::Causal) ? (n_m - 1 - lane) : lane;

    // q/k/v are addressed through the caller's strides; `out` is always a
    // freshly allocated packed [B,H,S,head_dim], so it keeps the flat formula.
    // The two coincide only when the caller handed us contiguous inputs.
    const long long qkv_off =
        static_cast<long long>(b) * qs0 + static_cast<long long>(h) * qs1;
    const long long out_off = static_cast<long long>(b * H + h) * S * HEAD_DIM;

    // ct::extents deduces a static extent from a ct::integral_constant and a
    // dynamic one from a plain int, so passing HEAD_DIM as an int put the row
    // stride in a register. Measured neutral (4462 SASS instructions either
    // way -- the tile compiler was already recovering it), kept because it is
    // the honest description of the tensor and costs nothing.
    constexpr ct::integral_constant<HEAD_DIM> HD{};
    // The row pitch is qs2 rather than HEAD_DIM. layout_right_padded is exactly
    // "row-major, last axis contiguous, rows spaced by an arbitrary stride",
    // which is the layout MySelfAttention's fused QKV projection produces -- so
    // the kernel reads those views directly instead of being handed clones.
    // A contiguous caller passes qs2 == HEAD_DIM and gets the old mapping back.
    const int pitch = static_cast<int>(qs2);
    auto q_view = ct::partition_view{
        ct::tensor_span{q + qkv_off,
                        ct::layout_right_padded_mapping{ct::extents{S, HD}, pitch}},
        ct::shape<BLOCK_M, HEAD_DIM>{}};
    // K is read only as K^T. Rather than load a [BLOCK_N, HEAD_DIM] tile and
    // ct::transpose it -- a real shuffle through shared memory, once per key
    // tile -- describe the same bytes as a column-major [HEAD_DIM, S] matrix:
    // layout_left makes element (d, j) be k[j * HEAD_DIM + d], exactly K^T for
    // free. The tile-level equivalent of wmma's col_major matrix_b fragment.
    // layout_left_padded pads the *first* extent, so element (d, j) lands at
    // k[j * pitch + d] -- still K^T for free, now with the strided pitch.
    auto kt_view = ct::partition_view{
        ct::tensor_span{k + qkv_off,
                        ct::layout_left_padded_mapping{ct::extents{HD, S}, pitch}},
        ct::shape<HEAD_DIM, BLOCK_N>{}};
    auto v_view = ct::partition_view{
        ct::tensor_span{v + qkv_off,
                        ct::layout_right_padded_mapping{ct::extents{S, HD}, pitch}},
        ct::shape<BLOCK_N, HEAD_DIM>{}};
    // The output view is built in the epilogue instead of here: under SPLIT this
    // block writes partials rather than `out`, and `out` is null on that path.

    // Q is read once and reused against every key tile, so the narrowing cast
    // is hoisted out of the key loop -- under tf32 it is expensive. load_masked
    // zero-fills rows past S; those rows are computed but never stored.
    //
    // The scale is deliberately not folded in here: pre-scaling Q changes what
    // gets rounded to tf32 and widened the error 3.1e-4 -> 6.2e-4 on the
    // 8x8x128x64 case for no measurable speed. It goes on the fp32 score tile
    // below, where the rounding is already done.
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

    // Each split takes a contiguous run of whole key tiles out of *this
    // block's* range, not out of [0, S). That is why split-KV works under
    // causal masking: slicing the dense range would leave the later splits of
    // an early m with nothing to do while a late m still carried everything.
    //
    // A split can come up empty when a block has fewer key tiles than there are
    // splits. It falls through the loop and stores the initial (NEG, 0, 0),
    // which the combine pass weights to exactly zero -- cheaper than a
    // launch-time special case, and impossible to get wrong.
    int kt_begin = 0;
    int kt_end   = key_limit;
    if constexpr (SPLIT) {
        const int n_kt = (key_limit + BLOCK_N - 1) / BLOCK_N;
        const int per  = (n_kt + splits - 1) / splits;
        kt_begin = split * per * BLOCK_N;
        kt_end   = kt_begin + per * BLOCK_N;
        if (kt_begin > key_limit) kt_begin = key_limit;
        if (kt_end   > key_limit) kt_end   = key_limit;
    }

    for (int kt = kt_begin; kt < kt_end; kt += BLOCK_N) {
        auto ktt = kt_view.load_masked(0, kt / BLOCK_N);
        auto vv  = v_view.load_masked(kt / BLOCK_N, 0);

        // S = Q @ K^T. One expression per block, no fragments -- K^T came out
        // of the view that way, so the only work here is the cast that puts
        // the operands on the tensor cores. log2(e) is folded into the scale
        // because the softmax below is exp2, which saves one multiply per
        // score element per key tile.
        //
        // ct::matmul would return a __half tile under MATH == Fp16 --
        // matmul_element_result<__half> is __half, and summing HEAD_DIM
        // products into 10 bits is not good enough. ct::mma against a zeroed
        // fp32 tile is the same multiply with the accumulator named, which is
        // what low_precision_mma_v permits. The other modes keep matmul so
        // their codegen is untouched.
        auto k_op = as_operand<MATH, HEAD_DIM, BLOCK_N>(ktt);
        auto s = [&] {
            if constexpr (MATH == MathMode::Fp16) {
                return ct::mma(q_op, k_op, ct::zeros<STile>());
            } else {
                return ct::matmul(q_op, k_op);
            }
        }() * (scale * 1.4426950408889634f);

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
        // new max appears, this rescales every iteration: the whole tile is one
        // expression, so a data-dependent branch per row would buy nothing.
        auto m_new = ct::max(m_run, ct::reduce_max<1>(s));
        auto corr  = ct::exp2(m_run - m_new);
        auto p     = ct::exp2(s - m_new);

        l_run = l_run * corr + ct::sum<1>(p);
        // ct::mma(A, B, C) accumulates into C, the tile equivalent of wmma's
        // mma_sync(o, p, v, o). Written as `acc * corr + ct::matmul(...)` the
        // product materialises into its own tile first -- a whole extra pass
        // over the accumulator per key tile.
        acc = ct::mma(as_operand<MATH, BLOCK_M, BLOCK_N>(p),
                      as_operand<MATH, BLOCK_N, HEAD_DIM>(vv),
                      acc * corr);
        m_run = m_new;
    }

    // A row whose every key was masked still holds the sentinel max. The
    // reference would produce nan there; emit zeros instead, as the scalar and
    // wmma kernels do -- such rows are zero-filled downstream anyway.
    if constexpr (SPLIT) {
        // The three running quantities go over untouched: this block saw one
        // slice of the row, so l_run is a partial denominator and m_run a
        // partial max, and the reciprocal can only be taken once every split
        // has been folded in.
        //
        // Layout is [B,H,SPLITS,S,HEAD_DIM] and [B,H,SPLITS,S], so head_dim
        // stays the fastest axis (the combine pass reads each row contiguously)
        // and one split's writes stay contiguous here.
        const long long p_row =
            (static_cast<long long>(b * H + h) * splits + split) * S;
        constexpr ct::integral_constant<1> ONE{};
        auto po_view = ct::partition_view{
            ct::tensor_span{part_o + p_row * HEAD_DIM, ct::extents{S, HD}},
            ct::shape<BLOCK_M, HEAD_DIM>{}};
        auto pm_view = ct::partition_view{
            ct::tensor_span{part_m + p_row, ct::extents{S, ONE}},
            ct::shape<BLOCK_M, 1>{}};
        auto pl_view = ct::partition_view{
            ct::tensor_span{part_l + p_row, ct::extents{S, ONE}},
            ct::shape<BLOCK_M, 1>{}};
        po_view.store_masked(acc, m_tile, 0);
        pm_view.store_masked(m_run, m_tile, 0);
        pl_view.store_masked(l_run, m_tile, 0);
    } else {
        auto o_view = ct::partition_view{
            ct::tensor_span{out + out_off, ct::extents{S, HD}},
            ct::shape<BLOCK_M, HEAD_DIM>{}};
        auto inv = ct::select(m_run > NEG_THRESH,
                              ct::full<RowTile>(1.0f) / l_run,
                              ct::zeros<RowTile>());
        o_view.store_masked(acc * inv, m_tile, 0);
    }
}

// Second pass of split-KV: fold the per-split partials into the finished row.
//
// Ordinary CUDA rather than tile code, deliberately -- this is one reduction of
// at most kMaxSplits terms per output element with no matmul in it, and the tile
// dialect has nothing to offer a pure streaming pass. -enable-tile adds the tile
// dialect to the translation unit, it does not remove the other one.
//
// The math is the inner loop's rescale one level up: each split reports the max
// over its own slice, so every partial is re-based onto the max over all slices
// before its numerator and denominator are added. Everything stays base-2,
// since the kernel folded log2(e) into the scale.
//
// One thread per (row, head_dim element), blockDim.y rows per block, so a warp
// stays contiguous in head_dim and the [B,H,S,HEAD_DIM] writes coalesce.
template <int HEAD_DIM>
__global__ void split_combine_kernel(const float* __restrict__ part_o,
                                     const float* __restrict__ part_m,
                                     const float* __restrict__ part_l,
                                     float* __restrict__ out,
                                     int S, int splits, long long n_rows) {
    const int d = static_cast<int>(threadIdx.x);
    const long long row =
        static_cast<long long>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    // n_rows is B*H*S and the partials are [B,H,SPLITS,S], so the split axis
    // sits between (b,h) and s. Recover both halves; stepping j costs one S.
    const long long bh   = row / S;
    const long long s    = row - bh * S;
    const long long base = bh * splits * S + s;

    float m_g = NEG;
    for (int j = 0; j < splits; ++j) {
        const float mj = part_m[base + static_cast<long long>(j) * S];
        m_g = mj > m_g ? mj : m_g;
    }
    // Every split of this row was empty or entirely masked out. Zeros, for the
    // same reason as the single-pass kernel's dead rows.
    if (m_g <= NEG_THRESH) {
        out[row * HEAD_DIM + d] = 0.0f;
        return;
    }

    float l_g = 0.0f;
    float o_g = 0.0f;
    for (int j = 0; j < splits; ++j) {
        const long long off = base + static_cast<long long>(j) * S;
        const float w = exp2f(part_m[off] - m_g);   // <= 1 by construction
        l_g += w * part_l[off];
        o_g += w * part_o[off * HEAD_DIM + d];
    }
    // Cannot divide by zero once the gate above has passed: the split that set
    // m_g has weight exactly 1, and its own running sum is at least 1 because
    // the element attaining its max contributes exp2(0).
    out[row * HEAD_DIM + d] = o_g / l_g;
}

// ---------------------------------------------------------------------------
// Block shapes
// ---------------------------------------------------------------------------
//
// Swept per (head_dim, math mode, mask mode) on SM 8.6, not derived. The kernel
// holds Q, O, K, V and the score tile live at once, and past a footprint
// threshold the compiler spills and the cost jumps an order of magnitude rather
// than degrading smoothly -- which is also why a narrow mode wants a different
// shape rather than a scaled one. Every extent must be a power of two; cuTile
// enforces is_pow2 per dimension (crt/cuda_tile.h:749).
//
// csrc/TUNING.md has the sweeps, the FlashAttention-2 shapes they agree with,
// and the two ways this measurement goes wrong. On a part with working TMA the
// cliff sits elsewhere; re-measure rather than trusting these.

// tf32, dense. Overridable from the build line so scripts/tune_block_shapes.py can
// search shapes without editing this file.
#ifndef TF32_M_8
#define TF32_M_8  128
#define TF32_N_8  16
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

// tf32, causal. Block m walks m+1 key tiles, so a 128-row block both halves
// the block count and doubles the spread between cheapest and most expensive
// block; the optimum sits at a smaller BLOCK_M than dense wants. Swept by
// scripts/tune_block_shapes.py --backend tile-tf32 --causal.
#ifndef TF32_CM_8
#define TF32_CM_8  64
#define TF32_CN_8  64
#endif
#ifndef TF32_CM_16
#define TF32_CM_16 64
#define TF32_CN_16 64
#endif
#ifndef TF32_CM_32
#define TF32_CM_32 128
#define TF32_CN_32 64
#endif
#ifndef TF32_CM_64
#define TF32_CM_64 64
#define TF32_CN_64 64
#endif

// tf32, dense, on a grid too small to fill the device. BLOCK_M sets the block
// count (ceil(S/BLOCK_M) * H * B), so halving it is the cheap way to fill 46 SMs
// at seq_len 128 -- and it loses at seq_len 512, which is why this is a
// launch-time choice rather than a retune. Split-KV attacks the same idle SMs
// and is the alternative, not a stack; see TUNING.md for which wins where.
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

// fp32 and bf16, dense. Overridable on the same terms as TF32_M_*: these were
// measured, but by an earlier pass that scored one shape across both mask
// modes. The causal macros below default to the dense value, so leaving them
// alone reproduces exactly what was measured.
#ifndef FP32_M_8
#define FP32_M_8  64
#define FP32_N_8  32
#endif
#ifndef FP32_M_16
#define FP32_M_16 32
#define FP32_N_16 16
#endif
#ifndef FP32_M_32
#define FP32_M_32 64
#define FP32_N_32 16
#endif
#ifndef FP32_M_64
#define FP32_M_64 32
#define FP32_N_64 16
#endif

#ifndef BF16_M_8
#define BF16_M_8  64
#define BF16_N_8  64
#endif
#ifndef BF16_M_16
#define BF16_M_16 64
#define BF16_N_16 64
#endif
#ifndef BF16_M_32
#define BF16_M_32 64
#define BF16_N_32 64
#endif
#ifndef BF16_M_64
#define BF16_M_64 64
#define BF16_N_64 64
#endif

// fp32 and bf16, causal. Defaulting to the dense shape keeps today's behaviour;
// the causal grid is triangular (see the TF32_CM_* note) and tf32 lost 12-25%
// at head_dim 64 to sharing one shape across both modes, so these exist to let
// the same question be asked of the other two math modes.
#ifndef FP32_CM_8
#define FP32_CM_8  FP32_M_8
#define FP32_CN_8  FP32_N_8
#endif
#ifndef FP32_CM_16
#define FP32_CM_16 FP32_M_16
#define FP32_CN_16 FP32_N_16
#endif
#ifndef FP32_CM_32
// Measured, not inherited: causal wants half the dense BLOCK_M here.
// 32x16 = 1.054 ms against the dense shape 64x16 at 1.237 (1.17x), one
// interleaved run. Dense at head_dim 32 still prefers 64x16 and is unchanged.
#define FP32_CM_32 32
#define FP32_CN_32 16
#endif
#ifndef FP32_CM_64
#define FP32_CM_64 FP32_M_64
#define FP32_CN_64 FP32_N_64
#endif

#ifndef BF16_CM_8
#define BF16_CM_8  BF16_M_8
#define BF16_CN_8  BF16_N_8
#endif
#ifndef BF16_CM_16
#define BF16_CM_16 BF16_M_16
#define BF16_CN_16 BF16_N_16
#endif
#ifndef BF16_CM_32
#define BF16_CM_32 BF16_M_32
#define BF16_CN_32 BF16_N_32
#endif
#ifndef BF16_CM_64
#define BF16_CM_64 BF16_M_64
#define BF16_CN_64 BF16_N_64
#endif

// CAUSAL is a separate axis because the causal grid is triangular; see the
// TF32_CM_* note. Splitting fp32 and bf16 on it costs no extra instantiation --
// MaskMode is already a parameter of the kernel template, so dense and causal
// were always separate kernels; only the shape they read was shared.
template <int HEAD_DIM, MathMode MODE, bool CAUSAL = false> struct BlockCfg;

template <> struct BlockCfg<8,  MathMode::Fp32, false> { static constexpr int M = FP32_M_8;   static constexpr int N = FP32_N_8;   };
template <> struct BlockCfg<8,  MathMode::Fp32, true>  { static constexpr int M = FP32_CM_8;  static constexpr int N = FP32_CN_8;  };
template <> struct BlockCfg<16, MathMode::Fp32, false> { static constexpr int M = FP32_M_16;  static constexpr int N = FP32_N_16;  };
template <> struct BlockCfg<16, MathMode::Fp32, true>  { static constexpr int M = FP32_CM_16; static constexpr int N = FP32_CN_16; };
template <> struct BlockCfg<32, MathMode::Fp32, false> { static constexpr int M = FP32_M_32;  static constexpr int N = FP32_N_32;  };
template <> struct BlockCfg<32, MathMode::Fp32, true>  { static constexpr int M = FP32_CM_32; static constexpr int N = FP32_CN_32; };
template <> struct BlockCfg<64, MathMode::Fp32, false> { static constexpr int M = FP32_M_64;  static constexpr int N = FP32_N_64;  };
template <> struct BlockCfg<64, MathMode::Fp32, true>  { static constexpr int M = FP32_CM_64; static constexpr int N = FP32_CN_64; };

// fp16 starts from bf16's shapes: same 16-bit operand, same tile geometry, so
// the sweep that tuned those is the best available prior. Overridable the same
// way if scripts/tune_block_shapes.py is ever pointed at this mode.
#ifndef FP16_M_8
#define FP16_M_8  BF16_M_8
#define FP16_N_8  BF16_N_8
#define FP16_M_16 BF16_M_16
#define FP16_N_16 BF16_N_16
#define FP16_M_32 BF16_M_32
#define FP16_N_32 BF16_N_32
#define FP16_M_64 BF16_M_64
#define FP16_N_64 BF16_N_64
#endif
#ifndef FP16_CM_8
#define FP16_CM_8  BF16_CM_8
#define FP16_CN_8  BF16_CN_8
#define FP16_CM_16 BF16_CM_16
#define FP16_CN_16 BF16_CN_16
#define FP16_CM_32 BF16_CM_32
#define FP16_CN_32 BF16_CN_32
#define FP16_CM_64 BF16_CM_64
#define FP16_CN_64 BF16_CN_64
#endif

template <> struct BlockCfg<8,  MathMode::Fp16, false> { static constexpr int M = FP16_M_8;   static constexpr int N = FP16_N_8;   };
template <> struct BlockCfg<8,  MathMode::Fp16, true>  { static constexpr int M = FP16_CM_8;  static constexpr int N = FP16_CN_8;  };
template <> struct BlockCfg<16, MathMode::Fp16, false> { static constexpr int M = FP16_M_16;  static constexpr int N = FP16_N_16;  };
template <> struct BlockCfg<16, MathMode::Fp16, true>  { static constexpr int M = FP16_CM_16; static constexpr int N = FP16_CN_16; };
template <> struct BlockCfg<32, MathMode::Fp16, false> { static constexpr int M = FP16_M_32;  static constexpr int N = FP16_N_32;  };
template <> struct BlockCfg<32, MathMode::Fp16, true>  { static constexpr int M = FP16_CM_32; static constexpr int N = FP16_CN_32; };
template <> struct BlockCfg<64, MathMode::Fp16, false> { static constexpr int M = FP16_M_64;  static constexpr int N = FP16_N_64;  };
template <> struct BlockCfg<64, MathMode::Fp16, true>  { static constexpr int M = FP16_CM_64; static constexpr int N = FP16_CN_64; };

template <> struct BlockCfg<8,  MathMode::Bf16, false> { static constexpr int M = BF16_M_8;   static constexpr int N = BF16_N_8;   };
template <> struct BlockCfg<8,  MathMode::Bf16, true>  { static constexpr int M = BF16_CM_8;  static constexpr int N = BF16_CN_8;  };
template <> struct BlockCfg<16, MathMode::Bf16, false> { static constexpr int M = BF16_M_16;  static constexpr int N = BF16_N_16;  };
template <> struct BlockCfg<16, MathMode::Bf16, true>  { static constexpr int M = BF16_CM_16; static constexpr int N = BF16_CN_16; };
template <> struct BlockCfg<32, MathMode::Bf16, false> { static constexpr int M = BF16_M_32;  static constexpr int N = BF16_N_32;  };
template <> struct BlockCfg<32, MathMode::Bf16, true>  { static constexpr int M = BF16_CM_32; static constexpr int N = BF16_CN_32; };
template <> struct BlockCfg<64, MathMode::Bf16, false> { static constexpr int M = BF16_M_64;  static constexpr int N = BF16_N_64;  };
template <> struct BlockCfg<64, MathMode::Bf16, true>  { static constexpr int M = BF16_CM_64; static constexpr int N = BF16_CN_64; };

#ifdef TILE_HAVE_TF32
// tf32 is the one mode where the two pressures pull apart: it occupies fp32's
// 32 bits, so the spill cliff sits where fp32's does, but it runs on the MMA
// units, whose 16x8x8 shape the narrow fp32 tiles starve. Inheriting either
// neighbour's shapes is wrong; these were swept.
template <> struct BlockCfg<8,  MathMode::Tf32, false> { static constexpr int M = TF32_M_8;  static constexpr int N = TF32_N_8;  };
template <> struct BlockCfg<8,  MathMode::Tf32, true>  { static constexpr int M = TF32_CM_8; static constexpr int N = TF32_CN_8; };
template <> struct BlockCfg<16, MathMode::Tf32, false> { static constexpr int M = TF32_M_16;  static constexpr int N = TF32_N_16;  };
template <> struct BlockCfg<16, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_16; static constexpr int N = TF32_CN_16; };
template <> struct BlockCfg<32, MathMode::Tf32, false> { static constexpr int M = TF32_M_32;  static constexpr int N = TF32_N_32;  };
template <> struct BlockCfg<32, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_32; static constexpr int N = TF32_CN_32; };
template <> struct BlockCfg<64, MathMode::Tf32, false> { static constexpr int M = TF32_M_64;  static constexpr int N = TF32_N_64;  };
template <> struct BlockCfg<64, MathMode::Tf32, true>  { static constexpr int M = TF32_CM_64; static constexpr int N = TF32_CN_64; };
#endif

// The dense shape for a grid too small to fill the device. Only tf32 has been
// measured; every other mode reports its dense shape, which turns the
// short-grid path off for it entirely.
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

// SM count of the current device, cached: the short-grid test runs on every
// call, and a driver round trip is a measurable share of a 60 us kernel.
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
            // A count of 1 makes every threshold unreachable, so a failed
            // query falls back to the plain long-grid path.
            n = 1;
        }
        cached[dev] = n;
    }
    return cached[dev];
}

// ---------------------------------------------------------------------------
// Split-KV (Flash-Decoding)
// ---------------------------------------------------------------------------
//
// Block shape only cuts up the *query* dimension, which runs out at short
// sequences: shared-memory use pins this kernel at one block per SM, so a grid
// of ceil(S/BLOCK_M)*H*B blocks can leave a third of the device idle with no
// BLOCK_M that fixes it. Split-KV cuts the *key* dimension instead -- each
// block takes a slice of the keys and writes an unnormalised partial, and a
// second pass folds them together. That multiplies the grid by `splits` without
// touching BLOCK_M, and the extra blocks read disjoint K/V.
//
// It costs a full pass over [B,H,SPLITS,S,HEAD_DIM] plus the scratch, so
// choose_splits() is conservative about when to do it at all. TUNING.md has the
// A/B table behind every constant here.

// Ceiling on how far the key range is cut up. Past a handful of splits the
// combine pass -- splits*B*H*S*(head_dim+2) floats -- costs more than the idle
// SMs it buys back, and the scratch stops being incidental.
constexpr int kMaxSplits = 8;

// Key tiles a split must walk before splitting is worth doing at all: the guard
// against the failure mode split-KV shares with shrinking BLOCK_M, that both
// buy blocks by replicating work. Not one number, because causal and dense are
// buying different things.
template <MathMode MATH>
constexpr int min_tiles_per_split(bool causal) {
    // Causal buys load balance, not occupancy: the makespan is set by the
    // largest block however full the grid is, and splitting evens that out. It
    // pays at two splits over a short range, hence the low bar.
    if (causal) return 4;

    // Dense and explicit have no imbalance to fix, so a split buys only idle
    // SMs against a full extra pass -- a loss at four tiles per split, a win at
    // eight. bf16's main kernel is the fastest of the three, so the same fixed
    // pass is a much larger fraction of it; 16 puts dense split-KV out of reach
    // for bf16 entirely, which is the intended effect.
    return (MATH == MathMode::Bf16 || MATH == MathMode::Fp16) ? 16 : 8;
}

// Scratch cap. A backstop against a pathological shape rather than a real
// constraint: batch 8, heads 8, seq 512, head_dim 64 at 4 splits is 34 MB.
constexpr size_t kMaxWorkspace = static_cast<size_t>(96) << 20;

// Runtime off switch, so split-KV can be A/B'd inside a single process; see
// set_split_kv() in the header. The environment variable supplies only the
// initial value. Deliberately unsynchronised: it is a benchmarking knob flipped
// between timed runs from one thread, not while launches are in flight.
bool& split_flag() {
    static bool on = [] {
        const char* e = std::getenv("TILE_SPLIT_KV");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

inline bool split_enabled() { return split_flag(); }

// Bytes of scratch `splits` ways needs. part_o is [B,H,SPLITS,S,HEAD_DIM] and
// part_m / part_l are one float per row each: hence head_dim + 2.
size_t split_bytes(int splits, int B, int H, int S, int head_dim) {
    if (splits < 2) return 0;
    const size_t rows = static_cast<size_t>(splits) *
                        static_cast<size_t>(B) * static_cast<size_t>(H) *
                        static_cast<size_t>(S);
    return rows * (static_cast<size_t>(head_dim) + 2) * sizeof(float);
}

// How many ways to cut the key range, for one block shape. 1 means "do not".
template <int BLOCK_M, int BLOCK_N, MathMode MATH>
int choose_splits(int B, int H, int S, int head_dim, bool causal) {
    if (!split_enabled()) return 1;

    const long long blocks =
        static_cast<long long>((S + BLOCK_M - 1) / BLOCK_M) * H * B;
    const long long target = 2LL * sm_count();
    // A grid that already fills the device twice over has nothing to gain from
    // more blocks, and a combine pass to lose by taking them.
    if (blocks >= target) return 1;

    // Key tiles the average block walks. Under causal that is half the dense
    // count; taking the dense count would over-split and leave tail splits
    // empty.
    const int keys = causal ? ((S + 1) / 2) : S;
    const int n_kt = (keys + BLOCK_N - 1) / BLOCK_N;

    const int by_grid = static_cast<int>((target + blocks - 1) / blocks);
    const int by_work = n_kt / min_tiles_per_split<MATH>(causal);
    int s = by_grid < by_work ? by_grid : by_work;
    if (s > kMaxSplits) s = kMaxSplits;
    while (s >= 2 && split_bytes(s, B, H, S, head_dim) > kMaxWorkspace) --s;
    return s < 2 ? 1 : s;
}

// The split count for a (head_dim, math mode, causal), resolved through the
// same BlockCfg the launch will use. workspace_bytes() and launch() both come
// through here, so the size allocated and the size written cannot disagree.
template <int HEAD_DIM, MathMode MATH>
int splits_for(int B, int H, int S, bool is_causal) {
    if constexpr (!mode_compiled<MATH>) {
        return 1;
    } else if (is_causal) {
        using Cfg = BlockCfg<HEAD_DIM, MATH, true>;
        return choose_splits<Cfg::M, Cfg::N, MATH>(B, H, S, HEAD_DIM, true);
    } else {
        using Cfg = BlockCfg<HEAD_DIM, MATH, false>;
        return choose_splits<Cfg::M, Cfg::N, MATH>(B, H, S, HEAD_DIM, false);
    }
}

template <MathMode MATH>
int splits_for_head_dim(int B, int H, int S, int head_dim, bool is_causal) {
    switch (head_dim) {
        case 8:  return splits_for<8,  MATH>(B, H, S, is_causal);
        case 16: return splits_for<16, MATH>(B, H, S, is_causal);
        case 32: return splits_for<32, MATH>(B, H, S, is_causal);
        case 64: return splits_for<64, MATH>(B, H, S, is_causal);
        default: return 1;
    }
}

// One (block shape, mask mode) launch -- single-pass, or the split pair. The
// only place that talks to the kernels.
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MaskMode MODE, MathMode MATH>
void launch_one(const float* q, const float* k, const float* v,
                const bool* mask, const long long* ms, const long long* qs,
                float* out, void* ws, int splits,
                int B, int H, int S, float scale, cudaStream_t stream) {
    const int n_m = (S + BLOCK_M - 1) / BLOCK_M;
    const dim3 block(1);  // one "thread" per block: the body runs once per
                          // block, and the compiler picks the real threads
    // ms is only meaningful alongside a mask, and the caller passes null for it
    // when there is none.
    const long long s0 = (mask != nullptr) ? ms[0] : 0;
    const long long s1 = (mask != nullptr) ? ms[1] : 0;
    const long long s2 = (mask != nullptr) ? ms[2] : 0;
    const long long s3 = (mask != nullptr) ? ms[3] : 0;

    if (splits < 2) {
        tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MODE, MATH, false>
            <<<dim3(n_m, H, B), block, 0, stream>>>(
                q, k, v, qs[0], qs[1], qs[2], mask, s0, s1, s2, s3,
                out, nullptr, nullptr, nullptr, B, H, S, scale, 1);
        return;
    }

    // Carve the caller's buffer into the three partial arrays. split_bytes()
    // sizes this same layout and the kernel indexes it; all three derive `rows`
    // the same way, so there is one definition of it to get wrong.
    const size_t rows = static_cast<size_t>(splits) *
                        static_cast<size_t>(B) * static_cast<size_t>(H) *
                        static_cast<size_t>(S);
    float* part_o = static_cast<float*>(ws);
    float* part_m = part_o + rows * HEAD_DIM;
    float* part_l = part_m + rows;

    tile_attention_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, MODE, MATH, true>
        <<<dim3(n_m * splits, H, B), block, 0, stream>>>(
            q, k, v, qs[0], qs[1], qs[2], mask, s0, s1, s2, s3,
            nullptr, part_o, part_m, part_l, B, H, S, scale, splits);

    // 256 threads, laid out head_dim-major so a warp never straddles two
    // output rows and the writes stay coalesced.
    constexpr int kRows = (256 / HEAD_DIM) > 0 ? (256 / HEAD_DIM) : 1;
    const long long n_rows = static_cast<long long>(B) * H * S;
    const long long n_blk  = (n_rows + kRows - 1) / kRows;
    split_combine_kernel<HEAD_DIM>
        <<<dim3(static_cast<unsigned>(n_blk)), dim3(HEAD_DIM, kRows), 0, stream>>>(
            part_o, part_m, part_l, out, S, splits, n_rows);
}

// The two dense mask modes at a fixed block shape. Factored out because the
// shape is a run-time choice, so this pair is needed twice.
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, MathMode MATH>
void launch_dense(const float* q, const float* k, const float* v,
                  const bool* mask, const long long* ms, const long long* qs,
                  float* out, void* ws, int splits,
                  int B, int H, int S, float scale, cudaStream_t stream) {
    if (mask != nullptr) {
        launch_one<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::Explicit, MATH>(
            q, k, v, mask, ms, qs, out, ws, splits, B, H, S, scale, stream);
    } else {
        launch_one<BLOCK_M, BLOCK_N, HEAD_DIM, MaskMode::None, MATH>(
            q, k, v, nullptr, ms, qs, out, ws, splits, B, H, S, scale, stream);
    }
}

template <int HEAD_DIM, MathMode MATH>
void launch_for_head_dim(const float* q, const float* k, const float* v,
                         const bool* mask, const long long* ms, const long long* qs,
                         float* out, void* ws, int splits,
                         int B, int H, int S,
                         bool is_causal, float scale, cudaStream_t stream) {
    if (is_causal) {
        // The causal kernel already runs a smaller BLOCK_M, so its grid is
        // twice as long and the short-grid test below would rarely fire. One
        // shape -- but it does take the split path, which slices each block's
        // own causal range and so stays balanced across splits.
        using Cfg = BlockCfg<HEAD_DIM, MATH, true>;
        launch_one<Cfg::M, Cfg::N, HEAD_DIM, MaskMode::Causal, MATH>(
            q, k, v, nullptr, ms, qs, out, ws, splits, B, H, S, scale, stream);
        return;
    }

    using Long  = BlockCfg<HEAD_DIM, MATH, false>;
    using Short = ShortCfg<HEAD_DIM, MATH>;
    // When the two agree there is nothing to choose between, and instantiating
    // the second kernel would only cost compile time.
    constexpr bool have_short_shape = (Short::M != Long::M) || (Short::N != Long::N);

    // Split-KV and ShortCfg attack the same starved grid, so they are
    // alternatives rather than a stack -- and splits was computed against
    // Long's shape, so taking it means keeping Long. ShortCfg is the answer
    // where there is too little key work to split.
    if (splits >= 2) {
        launch_dense<Long::M, Long::N, HEAD_DIM, MATH>(
            q, k, v, mask, ms, qs, out, ws, splits, B, H, S, scale, stream);
        return;
    }

    if constexpr (have_short_shape) {
        // Two waves is the threshold: below it the tail wave runs a partial
        // grid, and the idle SMs cost more than the extra blocks do. That is
        // also where the two shapes swapped places when measured.
        const long long blocks =
            static_cast<long long>((S + Long::M - 1) / Long::M) * H * B;
        if (blocks < 2LL * sm_count()) {
            launch_dense<Short::M, Short::N, HEAD_DIM, MATH>(
                q, k, v, mask, ms, qs, out, nullptr, 1, B, H, S, scale, stream);
            return;
        }
    }
    launch_dense<Long::M, Long::N, HEAD_DIM, MATH>(
        q, k, v, mask, ms, qs, out, nullptr, 1, B, H, S, scale, stream);
}

}  // namespace

namespace tile_attn {

bool available() { return true; }

bool supports(MathMode mode) {
    switch (mode) {
        case MathMode::Fp32: return mode_compiled<MathMode::Fp32>;
        case MathMode::Bf16: return mode_compiled<MathMode::Bf16>;
        case MathMode::Tf32: return mode_compiled<MathMode::Tf32>;
        case MathMode::Fp16: return mode_compiled<MathMode::Fp16>;
    }
    return false;
}

namespace {

// MATH is resolved before head_dim so an uncompiled mode costs no
// instantiations: every (head_dim, mode) pair reaching here is its own kernel.
template <MathMode MATH>
bool launch_mode(const float* q, const float* k, const float* v,
                 const bool* mask, const long long* ms, const long long* qs,
                 float* out, void* ws, size_t ws_bytes,
                 int B, int H, int S, int head_dim,
                 bool is_causal, float scale, cudaStream_t stream) {
    if constexpr (!mode_compiled<MATH>) {
        return false;
    } else {
        // No scratch, or too little, degrades to the single-pass kernel rather
        // than failing: the split path is a performance choice, so a caller
        // that skipped workspace_bytes() made a slower call, not a wrong one.
        const int want = splits_for_head_dim<MATH>(B, H, S, head_dim, is_causal);
        const int splits =
            (ws != nullptr && ws_bytes >= split_bytes(want, B, H, S, head_dim))
                ? want
                : 1;
        switch (head_dim) {
            case 8:
                launch_for_head_dim<8, MATH>(q, k, v, mask, ms, qs, out, ws, splits, B, H, S, is_causal, scale, stream);
                return true;
            case 16:
                launch_for_head_dim<16, MATH>(q, k, v, mask, ms, qs, out, ws, splits, B, H, S, is_causal, scale, stream);
                return true;
            case 32:
                launch_for_head_dim<32, MATH>(q, k, v, mask, ms, qs, out, ws, splits, B, H, S, is_causal, scale, stream);
                return true;
            case 64:
                launch_for_head_dim<64, MATH>(q, k, v, mask, ms, qs, out, ws, splits, B, H, S, is_causal, scale, stream);
                return true;
            default:
                return false;
        }
    }
}

}  // namespace

void set_split_kv(bool enabled) { split_flag() = enabled; }

bool split_kv_enabled() { return split_flag(); }

size_t workspace_bytes(int B, int H, int S, int head_dim,
                       bool is_causal, MathMode mode) {
    int splits = 1;
    switch (mode) {
        case MathMode::Fp32:
            splits = splits_for_head_dim<MathMode::Fp32>(B, H, S, head_dim, is_causal);
            break;
        case MathMode::Bf16:
            splits = splits_for_head_dim<MathMode::Bf16>(B, H, S, head_dim, is_causal);
            break;
        case MathMode::Fp16:
            splits = splits_for_head_dim<MathMode::Fp16>(B, H, S, head_dim, is_causal);
            break;
        case MathMode::Tf32:
            splits = splits_for_head_dim<MathMode::Tf32>(B, H, S, head_dim, is_causal);
            break;
    }
    return split_bytes(splits, B, H, S, head_dim);
}

bool launch(const float* q, const float* k, const float* v,
            const bool* mask, const long long* ms, const long long* qs,
            float* out, void* ws, size_t ws_bytes,
            int B, int H, int S, int head_dim,
            bool is_causal, float scale, MathMode mode, cudaStream_t stream) {
    switch (mode) {
        case MathMode::Fp32:
            return launch_mode<MathMode::Fp32>(q, k, v, mask, ms, qs, out, ws, ws_bytes,
                                               B, H, S, head_dim, is_causal, scale, stream);
        case MathMode::Bf16:
            return launch_mode<MathMode::Bf16>(q, k, v, mask, ms, qs, out, ws, ws_bytes,
                                               B, H, S, head_dim, is_causal, scale, stream);
        case MathMode::Fp16:
            return launch_mode<MathMode::Fp16>(q, k, v, mask, ms, qs, out, ws, ws_bytes,
                                               B, H, S, head_dim, is_causal, scale,
                                               stream);
        case MathMode::Tf32:
            return launch_mode<MathMode::Tf32>(q, k, v, mask, ms, qs, out, ws, ws_bytes,
                                               B, H, S, head_dim, is_causal, scale, stream);
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

void set_split_kv(bool) {}
bool split_kv_enabled() { return false; }

size_t workspace_bytes(int, int, int, int, bool, MathMode) { return 0; }

bool launch(const float*, const float*, const float*,
            const bool*, const long long*,
            float*, void*, size_t, int, int, int, int, bool, float,
            MathMode, cudaStream_t) {
    return false;
}

}  // namespace tile_attn

#endif  // TRANSFORMER_HAVE_TILE
