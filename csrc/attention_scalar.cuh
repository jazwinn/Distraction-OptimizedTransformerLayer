// Scalar kernel: one thread owns one query row -- or, past head_dim 64, a
// 64-dim slice of one.
//
// A thread keeps q and the output accumulator for its row in registers, which
// is the whole reason the key loop is cheap: no reload per key, and the
// accumulate is a register FMA. That is HEAD_DIM floats twice over, so at
// head_dim 128 it wants 256 registers per thread against a hardware ceiling of
// 255 -- not a tuning problem, a wall. Which is why this kernel stopped at 64
// and why REPORT.md's "scalar at head_dim 128" measurements were always ATen.
//
// So past 64 the row is split across HEAD_DIM/64 threads instead -- two at
// head_dim 128, four at 256. Each owns 64 dims of q and the matching 64 of acc,
// computes a partial dot product against the key, and a log2(TPR)-step
// __shfl_xor_sync butterfly over the group makes the score whole in every
// member at once. Everything after that -- running max, rescale, accumulate,
// normalise -- every partner runs identically on its own slice. log2(TPR)
// shuffles per key, no other communication, and the same total shared-memory
// traffic as before: the threads read disjoint slices of the key row.
//
// That also means the per-thread register cost stops growing at head_dim 64:
// 128 floats of q_reg + acc whether the head is 64, 128 or 256 wide. What grows
// instead is the block -- 64, 128, 256 threads -- and the key tile has to
// shrink to pay for it, which is what BLOCK_N below does.
//
// The shuffle mask is TPR lanes wide, and that is safe because a group shares a
// row index: the i < S guard, the causal break and the mask continue all depend
// on the row and never on which slice, so a group is always converged. Group
// members are adjacent lanes by construction (row = tid / TPR) and never
// straddle a warp boundary, since TPR is a power of two that divides 32.

#pragma once

#include "kernel_common.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdlib>
#include <type_traits>

namespace {

// Forces the generic catch-all below to serve head_dims the six specializations
// already cover, so the two can be diffed on identical shapes inside one
// process. Same knob contract as the wmma kernel's flags: a benchmarking switch
// read once, never flipped while launches are in flight. Off by default, so a
// normal run never touches it.
bool scalar_force_generic() {
    static const bool on = [] {
        const char* e = std::getenv("SCALAR_FORCE_GENERIC");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

template <typename scalar_t, int HEAD_DIM>
struct ScalarCfg {
    static constexpr int BLOCK_M = 64;              // query rows per block

    // Dims one thread owns, capped so q_reg + acc clear the register ceiling.
    // At or below 64 a thread takes the whole row and TPR collapses to 1, which
    // compiles the shuffle away entirely -- the existing head_dims are
    // untouched by any of this.
    static constexpr int DIMS = (HEAD_DIM > 64) ? 64 : HEAD_DIM;
    static constexpr int TPR = HEAD_DIM / DIMS;     // threads per row
    static constexpr int NTHREADS = BLOCK_M * TPR;

    // Keys per shared tile. Picked so k_s + v_s land at 32 KB for every
    // head_dim from 64 up, comfortably inside the 48 KB that keeps two blocks
    // resident per SM. The tile is BLOCK_N * HEAD_DIM elements twice over, so
    // each doubling of head_dim halves this.
    static constexpr int BLOCK_N =
        (HEAD_DIM >= 256) ? 16
                          : ((HEAD_DIM >= 128) ? 32 : ((HEAD_DIM >= 64) ? 64 : 128));
    static constexpr size_t SMEM =
        2 * static_cast<size_t>(BLOCK_N) * HEAD_DIM * sizeof(scalar_t);

    // The largest dynamic allocation a block gets without opting into the
    // bigger carveout, which would cost the second resident block.
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    static constexpr bool SUPPORTED = (SMEM <= SMEM_LIMIT);

    static_assert(HEAD_DIM % DIMS == 0, "head_dim must split evenly over a row");
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0,
                  "the score butterfly assumes a power-of-two lane group");
    static_assert(NTHREADS % 32 == 0, "block must be whole warps");
    static_assert(32 % TPR == 0, "a row's threads must not straddle two warps");
};

template <typename scalar_t, int HEAD_DIM>
__global__ void fused_attention_kernel(const scalar_t* __restrict__ q,
                                       const scalar_t* __restrict__ k,
                                       const scalar_t* __restrict__ v,
                                       int64_t qs0, int64_t qs1, int64_t qs2,
                                       const bool* __restrict__ mask,
                                       int64_t ms0, int64_t ms1,
                                       int64_t ms2, int64_t ms3,
                                       scalar_t* __restrict__ out,
                                       int B, int H, int S,
                                       bool is_causal, float scale,
                                       bool out_bshd) {
    using Cfg = ScalarCfg<scalar_t, HEAD_DIM>;
    constexpr int BLOCK_M = Cfg::BLOCK_M;
    constexpr int BLOCK_N = Cfg::BLOCK_N;
    constexpr int DIMS = Cfg::DIMS;
    constexpr int TPR = Cfg::TPR;

    extern __shared__ __align__(16) char smem_raw[];
    scalar_t* k_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* v_s = k_s + BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int row = tid / TPR;          // query row inside the block
    const int d0 = (tid % TPR) * DIMS;  // first head_dim this thread owns
    const int m_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int i = m_tile * BLOCK_M + row;
    const bool active = (i < S);

    // The TPR lanes sharing a row, and nothing else. Safe as a fully-populated
    // mask because a group never diverges -- see the note above the kernel. At
    // TPR == 1 there is no shuffle to mask.
    const unsigned group_mask =
        (TPR == 1) ? 0u
                   : (((1u << TPR) - 1u)
                      << (threadIdx.x & 31u & ~static_cast<unsigned>(TPR - 1)));

    const int64_t bh_off =
        static_cast<int64_t>(b) * qs0 + static_cast<int64_t>(h) * qs1;

    // q is read once and reused against every key, so it earns registers.
    float q_reg[DIMS];
    if (active) {
        const scalar_t* q_row = q + bh_off + static_cast<int64_t>(i) * qs2 + d0;
        #pragma unroll
        for (int d = 0; d < DIMS; ++d) {
            q_reg[d] = static_cast<float>(q_row[d]);
        }
    }

    // Running softmax state, accumulated in float even for half inputs -- the
    // reference does its softmax in fp32 and we have to match that.
    float acc[DIMS];
    #pragma unroll
    for (int d = 0; d < DIMS; ++d) {
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

        // head_dim is stride-1 in every layout the caller can pass, so one key
        // row is still a flat span and the loads stay coalesced; only the
        // spacing between rows is a runtime value. Shared memory stays packed,
        // hence the flat destination index.
        //
        // Rows are the outer loop so the 64-bit row offset is computed once per
        // row rather than once per element. Flattening the two into a single
        // strided loop costs a divide, a modulo and a 64-bit multiply per
        // element, and measured slower.
        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * qs2;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * qs2;
        for (int r = 0; r < n_keys; ++r) {
            const int64_t g = static_cast<int64_t>(r) * qs2;
            for (int c = tid; c < HEAD_DIM; c += Cfg::NTHREADS) {
                k_s[r * HEAD_DIM + c] = k_base[g + c];
                v_s[r * HEAD_DIM + c] = v_base[g + c];
            }
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
                for (int d = 0; d < DIMS; ++d) {
                    s += q_reg[d] * static_cast<float>(k_s[j * HEAD_DIM + d0 + d]);
                }
                // A slice of the dot product each; a butterfly over the
                // group leaves every member holding the whole of it.
                if constexpr (TPR > 1) {
                    #pragma unroll
                    for (int off = 1; off < TPR; off <<= 1) {
                        s += __shfl_xor_sync(group_mask, s, off);
                    }
                }
                s *= scale;

                // Online softmax: rescale only when a new max appears, which
                // after the first few keys is rare.
                if (s > m_run) {
                    const float corr = __expf(m_run - s);
                    #pragma unroll
                    for (int d = 0; d < DIMS; ++d) {
                        acc[d] *= corr;
                    }
                    l_run *= corr;
                    m_run = s;
                }

                const float p = __expf(s - m_run);
                l_run += p;
                #pragma unroll
                for (int d = 0; d < DIMS; ++d) {
                    acc[d] += p * static_cast<float>(v_s[j * HEAD_DIM + d0 + d]);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        // Both partners carry the same l_run -- they summed the same scores --
        // so each normalises its own half without another exchange.
        scalar_t* out_row =
            out + out_base(out_bshd, b, h, i, H, S, HEAD_DIM) + d0;
        // l_run == 0 means every key was masked. The reference would produce
        // NaN there; emit 0 instead, since such rows are zero-filled downstream
        // anyway and NaN would only risk contaminating something else.
        const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        #pragma unroll
        for (int d = 0; d < DIMS; ++d) {
            out_row[d] = static_cast<scalar_t>(acc[d] * inv);
        }
    }
}

// Returns false when this (dtype, head_dim) pair asks for more shared memory
// than a block gets. Declining is the point: it turns "no kernel for this case"
// into the same coverage gap every other impl reports, rather than a launch
// failure surfaced several frames away. double at head_dim 32 and up is what
// actually reaches it -- 64 KB of tiles against a 48 KB budget -- and it used
// to launch and fail.
template <typename scalar_t, int HEAD_DIM>
bool launch_kernel(const torch::Tensor& q, const torch::Tensor& k,
                   const torch::Tensor& v, const bool* mask_ptr,
                   const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                   int B, int H, int S, bool is_causal, double scale) {
    using Cfg = ScalarCfg<scalar_t, HEAD_DIM>;

    // if constexpr, not a plain if: an unsupported pair then never instantiates
    // the kernel at all, so nothing is compiled for a launch that cannot happen.
    if constexpr (!Cfg::SUPPORTED) {
        return false;
    } else {
        const dim3 block(Cfg::NTHREADS);
        const dim3 grid((S + Cfg::BLOCK_M - 1) / Cfg::BLOCK_M, H, B);

        // A 3-D output is [B, S, H*head_dim]; a 4-D one is [B, H, S, head_dim].
        // Reading the layout off the tensor keeps it out of every dispatch
        // signature between here and fused_attention_forward.
        const bool out_bshd = (out.dim() == 3);

        fused_attention_kernel<scalar_t, HEAD_DIM>
            <<<grid, block, Cfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                qs[0], qs[1], qs[2],
                mask_ptr, ms[0], ms[1], ms[2], ms[3],
                out.data_ptr<scalar_t>(), B, H, S, is_causal,
                static_cast<float>(scale), out_bshd);
        return true;
    }
}

// ---------------------------------------------------------------------------
// Generic scalar kernel -- the catch-all.
//
// The six specializations above are the fast path and nothing here touches
// them. This covers everything they and the wmma kernel decline -- any head_dim
// at all, in any dtype -- so that "no kernel for this shape" stops being a
// RuntimeError. Same algorithm as the kernel above, with the three things that
// were compile-time constants made runtime arguments:
//
//   head_dim   the real width, used for every global load and store
//   TPR        threads per query row
//   BLOCK_N    keys per shared tile, sized from the shared-memory budget
//
// What cannot become a runtime value is DIMS, the per-thread slice. q_reg and
// acc are register arrays, and a register array needs a compile-time length and
// compile-time indices -- a runtime loop bound puts both in local memory and
// the key loop stops being cheap, which was the whole point of the design. So
// DIMS stays pinned at 64, the largest slice the register file takes and
// exactly what the specializations use past head_dim 64, and the row is split
// over TPR = ceil(head_dim / 64) threads rounded up to a power of two, so the
// existing score butterfly still works unchanged.
//
// The rounding is what being general costs: the block computes PDIM = TPR * 64
// dims of which only head_dim are real. head_dim 96 runs as 128 and throws away
// a quarter of the dot product; head_dim 65 runs as 128 and throws away half.
// That is the same bucket tax the wmma kernel would pay, one notch coarser.
//
// Columns past head_dim are zeroed in q_reg AND in the shared tiles. Zeroing q
// alone would keep the score right -- 0 * anything drops out of the sum --
// except that shared memory is not initialized, and 0 * NaN is NaN.
//
// The ceiling is TPR 32: the butterfly is a __shfl_xor_sync over one warp, so a
// row's threads have to fit in a warp. That puts the widest head_dim this
// kernel covers at 64 * 32 = 2048, eight times the widest any specialization
// takes. Past it the reduction needs shared memory and a barrier inside the key
// loop, which is a different kernel rather than a bigger constant -- so it
// declines, and the caller reports a coverage gap the way it always has.

// Dims one thread owns. Not a tuning knob: see the note above.
constexpr int kGenericDims = 64;

template <typename scalar_t>
__global__ __launch_bounds__(256)
void fused_attention_generic_kernel(const scalar_t* __restrict__ q,
                                    const scalar_t* __restrict__ k,
                                    const scalar_t* __restrict__ v,
                                    int64_t qs0, int64_t qs1, int64_t qs2,
                                    const bool* __restrict__ mask,
                                    int64_t ms0, int64_t ms1,
                                    int64_t ms2, int64_t ms3,
                                    scalar_t* __restrict__ out,
                                    int B, int H, int S, int head_dim,
                                    int block_m, int block_n, int tpr,
                                    bool is_causal, float scale,
                                    bool out_bshd) {
    constexpr int DIMS = kGenericDims;
    const int pdim = tpr * DIMS;          // padded width the block actually walks

    extern __shared__ __align__(16) char smem_raw[];
    scalar_t* k_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* v_s = k_s + static_cast<size_t>(block_n) * pdim;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int row = tid / tpr;            // query row inside the block
    const int d0 = (tid % tpr) * DIMS;    // first head_dim this thread owns
    const int m_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int i = m_tile * block_m + row;
    const bool active = (i < S);

    // As above, the TPR lanes sharing a row and nothing else. The tpr == 32 case
    // is spelled separately because 1u << 32 is undefined, not zero.
    const unsigned group_mask =
        (tpr == 1)
            ? 0u
            : ((tpr == 32)
                   ? 0xffffffffu
                   : (((1u << tpr) - 1u)
                      << (threadIdx.x & 31u & ~static_cast<unsigned>(tpr - 1))));

    const int64_t bh_off =
        static_cast<int64_t>(b) * qs0 + static_cast<int64_t>(h) * qs1;

    const scalar_t zero_v = static_cast<scalar_t>(0.0f);

    // q is read once and reused against every key, so it earns registers. The
    // bound test is per element rather than per thread: one thread's slice can
    // straddle head_dim -- at head_dim 96 the second slice holds 32 real dims
    // and 32 of padding -- so a whole-slice predicate would be wrong.
    float q_reg[DIMS];
    if (active) {
        const scalar_t* q_row = q + bh_off + static_cast<int64_t>(i) * qs2;
        #pragma unroll
        for (int d = 0; d < DIMS; ++d) {
            const int c = d0 + d;
            q_reg[d] = (c < head_dim) ? static_cast<float>(q_row[c]) : 0.0f;
        }
    }

    float acc[DIMS];
    #pragma unroll
    for (int d = 0; d < DIMS; ++d) {
        acc[d] = 0.0f;
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int64_t mask_bh =
        (mask != nullptr) ? (static_cast<int64_t>(b) * ms0 + static_cast<int64_t>(h) * ms1) : 0;

    const int key_limit = is_causal ? min(S, m_tile * block_m + block_m) : S;

    for (int kt = 0; kt < key_limit; kt += block_n) {
        const int n_keys = min(block_n, S - kt);

        const scalar_t* k_base = k + bh_off + static_cast<int64_t>(kt) * qs2;
        const scalar_t* v_base = v + bh_off + static_cast<int64_t>(kt) * qs2;
        for (int r = 0; r < n_keys; ++r) {
            const int64_t g = static_cast<int64_t>(r) * qs2;
            for (int c = tid; c < pdim; c += nthreads) {
                const bool inb = (c < head_dim);
                k_s[r * pdim + c] = inb ? k_base[g + c] : zero_v;
                v_s[r * pdim + c] = inb ? v_base[g + c] : zero_v;
            }
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
                for (int d = 0; d < DIMS; ++d) {
                    s += q_reg[d] * static_cast<float>(k_s[j * pdim + d0 + d]);
                }
                // tpr is a kernel argument, so this branch is uniform across the
                // whole block, never mind the warp.
                if (tpr > 1) {
                    for (int off = 1; off < tpr; off <<= 1) {
                        s += __shfl_xor_sync(group_mask, s, off);
                    }
                }
                s *= scale;

                if (s > m_run) {
                    const float corr = __expf(m_run - s);
                    #pragma unroll
                    for (int d = 0; d < DIMS; ++d) {
                        acc[d] *= corr;
                    }
                    l_run *= corr;
                    m_run = s;
                }

                const float p = __expf(s - m_run);
                l_run += p;
                #pragma unroll
                for (int d = 0; d < DIMS; ++d) {
                    acc[d] += p * static_cast<float>(v_s[j * pdim + d0 + d]);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        scalar_t* out_row = out + out_base(out_bshd, b, h, i, H, S, head_dim);
        const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        // Padding columns exist only to fill a thread's slice; dropped here, so
        // nothing past head_dim is ever written.
        #pragma unroll
        for (int d = 0; d < DIMS; ++d) {
            const int c = d0 + d;
            if (c < head_dim) {
                out_row[c] = static_cast<scalar_t>(acc[d] * inv);
            }
        }
    }
}

// Returns false only for a head_dim this kernel genuinely cannot reach: past
// 2048, where a row's threads would outgrow a warp, or a padded key tile so
// large that not even one key fits the shared-memory budget. Everything else
// launches, which is what makes this the catch-all.
template <typename scalar_t>
bool launch_generic_kernel(const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& v, const bool* mask_ptr,
                           const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                           int B, int H, int S, int head_dim,
                           bool is_causal, double scale) {
    constexpr int DIMS = kGenericDims;
    if (head_dim <= 0) {
        return false;
    }

    // Threads per row: enough 64-dim slices to cover head_dim, rounded up to a
    // power of two because the butterfly halves the group each step.
    int tpr = 1;
    while (tpr * DIMS < head_dim) {
        tpr <<= 1;
    }
    if (tpr > 32) {
        return false;  // a row's group would straddle warps -- see the note above
    }
    const int pdim = tpr * DIMS;

    // A wide head buys its extra slices with query rows, and the budget it
    // spends them against is the register file, not the 1024-thread block
    // limit. q_reg + acc is 128 floats per thread whatever head_dim is, so a
    // block of 1024 would want ~131k registers against an SM's 65536 and the
    // launch fails outright with "too many resources requested" -- measured,
    // from head_dim 320 up, before this cap existed.
    //
    // 256 is what the specializations already prove launchable: head_dim 256
    // runs TPR 4 x BLOCK_M 64. Capping there reproduces their thread counts
    // exactly at TPR 1, 2 and 4, and buys the wider heads by shrinking
    // BLOCK_M instead -- 32 rows at TPR 8, down to 8 at TPR 32.
    static constexpr int kMaxThreads = 256;
    int block_m = kMaxThreads / tpr;
    if (block_m > 64) {
        block_m = 64;
    }
    const int nthreads = block_m * tpr;

    // Keys per tile, from the budget rather than a table: the padded row is a
    // runtime width, so what survives is the 48 KB that keeps two blocks
    // resident, not a per-head_dim BLOCK_N.
    const size_t per_key = 2 * static_cast<size_t>(pdim) * sizeof(scalar_t);
    size_t block_n = (48 * 1024) / per_key;
    if (block_n == 0) {
        return false;  // one key does not fit; nothing here can help
    }
    if (block_n > 128) {
        block_n = 128;  // the widest tile any specialization uses
    }
    if (block_n > static_cast<size_t>(S)) {
        block_n = static_cast<size_t>(S);  // never stage past the sequence
    }
    const size_t smem = block_n * per_key;

    const dim3 block(nthreads);
    const dim3 grid((S + block_m - 1) / block_m, H, B);
    const bool out_bshd = (out.dim() == 3);

    fused_attention_generic_kernel<scalar_t>
        <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            qs[0], qs[1], qs[2],
            mask_ptr, ms[0], ms[1], ms[2], ms[3],
            out.data_ptr<scalar_t>(), B, H, S, head_dim,
            block_m, static_cast<int>(block_n), tpr, is_causal,
            static_cast<float>(scale), out_bshd);
    return true;
}

// Returns false when nothing covers this case -- no specialization for the
// head_dim, or one whose tiles do not fit in shared memory for this dtype.
template <typename scalar_t>
bool dispatch_head_dim(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const bool* mask_ptr,
                       const int64_t* ms, const int64_t* qs, torch::Tensor& out,
                       int B, int H, int S, int head_dim,
                       bool is_causal, double scale) {
    // Declining here drops the case to launch_generic_kernel, which covers
    // every head_dim these six do. Only the A/B knob takes that path for a
    // head_dim that has a specialization.
    if (scalar_force_generic()) {
        return false;
    }
    switch (head_dim) {
        case 8:
            return launch_kernel<scalar_t, 8>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 16:
            return launch_kernel<scalar_t, 16>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 32:
            return launch_kernel<scalar_t, 32>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 64:
            return launch_kernel<scalar_t, 64>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 128:
            return launch_kernel<scalar_t, 128>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        case 256:
            return launch_kernel<scalar_t, 256>(q, k, v, mask_ptr, ms, qs, out, B, H, S, is_causal, scale);
        default:
            return false;
    }
}

}  // namespace
