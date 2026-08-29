// Kernel selection, and fused_attention_forward() -- the attention entry point
// Python calls.
//
// Every launcher below answers "can you handle this case?": true having
// launched, false having done nothing. That keeps the choice of kernel (here)
// separate from each kernel's coverage rules (inside it), so the caller is a
// plain list of preferences.
//
// The three candidates live in attention_scalar.cuh, attention_wmma.cuh and
// tile_attention.cu. There is no fourth: nothing here falls back to SDPA or to
// any other prebuilt attention, and a case none of them covers raises at the
// bottom of fused_attention_forward.

#pragma once

#include "attention_scalar.cuh"
#include "attention_wmma.cuh"
#include "tile_attention.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdlib>

namespace {

// WHICH kernel. What arithmetic it uses is AttnPrecision, a separate argument
// -- see the table in kernel_common.cuh. These used to be one axis: `tile-fp16`
// named both at once, and there were four Impl values for one kernel, while
// wmma's precision lived in a --attn-fp16 flag that named no kernel at all. The
// combinations that gained nothing by being spelled out (tile-bf16 as a
// distinct *implementation*) were exactly the ones hardest to describe.
enum class Impl : int64_t {
    Auto   = 0,
    Scalar = 1,
    Wmma   = 2,
    Tile   = 3,
};

const char* impl_name(Impl impl) {
    switch (impl) {
        case Impl::Auto:   return "auto";
        case Impl::Scalar: return "scalar";
        case Impl::Wmma:   return "wmma";
        case Impl::Tile:   return "tile";
    }
    return "?";
}

// The tile kernels carry their own MathMode enum, so this is the one place the
// two vocabularies meet.
tile_attn::MathMode tile_math_for(AttnPrecision prec) {
    switch (prec) {
        case AttnPrecision::Bf16: return tile_attn::MathMode::Bf16;
        case AttnPrecision::Tf32: return tile_attn::MathMode::Tf32;
        case AttnPrecision::Fp16: return tile_attn::MathMode::Fp16;
        default:                  return tile_attn::MathMode::Fp32;
    }
}

// One bundle so the launchers do not each take a dozen positional arguments.
struct AttnArgs {
    const torch::Tensor& q;
    const torch::Tensor& k;
    const torch::Tensor& v;
    const bool* mask_ptr;
    const int64_t* ms;
    // Batch, head and sequence strides of q/k/v, in elements. The head_dim
    // stride is not carried: it is always 1, because fused_attention_forward
    // makes a contiguous copy rather than pass a layout where it is not.
    // Contiguous inputs give {H*S*head_dim, S*head_dim, head_dim}, so a kernel
    // reading these needs no separate contiguous path.
    const int64_t* qs;
    torch::Tensor& out;
    int B;
    int H;
    int S;
    int head_dim;
    bool is_causal;
    double scale;
    AttnPrecision prec;
};

// The scalar family, tuned instantiations first and the generic catch-all
// behind them. The two are one launcher rather than two entries in every
// preference list because they are one kernel's worth of coverage: same file,
// same algorithm, same accuracy -- the difference is only whether head_dim, the
// per-row thread count and the key tile are template parameters or runtime
// arguments. A caller choosing "the scalar kernel" is not choosing between
// them, so it should not have to name both.
//
// dispatch_head_dim declining means one of two things, and the fallback is
// right for both: no specialization exists for this head_dim, or one exists
// whose key tiles do not fit in shared memory for this dtype (float64 past
// head_dim 16 is what actually hits that). The generic kernel sizes its tile
// from the budget instead of a table, so it covers head_dim 1 to 2048 in every
// dtype and takes both cases.
//
// Keeping the fallback inside the AT_DISPATCH is not just tidiness: it happens
// inside the same scalar_t instantiation, so the dtype is resolved once for
// both attempts rather than twice.
bool launch_scalar(const AttnArgs& a) {
    bool launched = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a.q.scalar_type(), "launch_scalar", [&] {
            launched = dispatch_head_dim<scalar_t>(
                a.q, a.k, a.v, a.mask_ptr, a.ms, a.qs, a.out,
                a.B, a.H, a.S, a.head_dim, a.is_causal, a.scale);
            if (!launched) {
                launched = launch_generic_kernel<scalar_t>(
                    a.q, a.k, a.v, a.mask_ptr, a.ms, a.qs, a.out,
                    a.B, a.H, a.S, a.head_dim, a.is_causal, a.scale);
            }
        });
    return launched;
}

bool launch_wmma(const AttnArgs& a) {
    // wmma fragments need SM 8.0+; below that the kernel body is compiled away
    // and would silently write zeros, so gate on the actual device.
    if (at::cuda::getCurrentDeviceProperties()->major < 8) {
        return false;
    }
    bool launched = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a.q.scalar_type(), "launch_wmma", [&] {
            launched = dispatch_wmma<scalar_t>(
                a.q, a.k, a.v, a.mask_ptr, a.ms, a.qs, a.out,
                a.B, a.H, a.S, a.head_dim, a.is_causal, a.scale, a.prec);
        });
    return launched;
}

bool launch_tile(const AttnArgs& a) {
    const tile_attn::MathMode math = tile_math_for(a.prec);
    // Unlike the other two, these are hard requirements rather than coverage
    // gaps: a caller asking for the tile kernel on a build or dtype that cannot
    // have it wants to hear so, not to be quietly given a different kernel.
    TORCH_CHECK(tile_attn::available(),
                "fused_attention_forward: impl=3 (tile) requested but this build "
                "has no tile support. It needs CUDA 13.3+ with -std=c++20 "
                "-enable-tile; see kernel_ext.py.");
    TORCH_CHECK(a.q.scalar_type() == torch::kFloat32,
                "fused_attention_forward: the tile kernels take float32 in and "
                "out -- the math mode narrows only the GEMM operands. Got ",
                a.q.scalar_type());
    // An invariant, not a user-facing rule: fused_attention_forward folds the
    // triangle into the mask before it gets here whenever a tile impl is what
    // was asked for. Checked rather than assumed, because dropping half a mask
    // is a wrong answer rather than a crash.
    TORCH_CHECK(!(a.is_causal && a.mask_ptr != nullptr),
                "fused_attention_forward: internal error -- the tile kernels "
                "have no combined causal + explicit mask mode and the fold did "
                "not happen.");
    TORCH_CHECK(tile_attn::supports(math),
                "fused_attention_forward: this build has no tile kernel for that "
                "math mode. tf32 needs a toolkit shipping <cuda_tf32.h> (CUDA "
                "13.3+); see csrc/tile_attention.h.");
    // Scratch for split-KV's per-split partials, zero on most shapes. From
    // torch's caching allocator rather than cudaMalloc: it is stream-ordered
    // (freed to the same stream the kernel runs on when `ws` dies) and it draws
    // on the pool the benchmark is already accounted against.
    const size_t ws_bytes = tile_attn::workspace_bytes(
        a.B, a.H, a.S, a.head_dim, a.is_causal, math);
    at::Tensor ws;
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        ws = at::empty({static_cast<int64_t>(ws_bytes)},
                       a.q.options().dtype(at::kByte));
        ws_ptr = ws.data_ptr();
    }
    return tile_attn::launch(
        a.q.const_data_ptr<float>(), a.k.const_data_ptr<float>(),
        a.v.const_data_ptr<float>(), a.mask_ptr, a.ms, a.qs,
        a.out.data_ptr<float>(), ws_ptr, ws_bytes,
        a.B, a.H, a.S, a.head_dim, a.is_causal,
        static_cast<float>(a.scale), math, at::cuda::getCurrentCUDAStream());
}

// No prebuilt attention. Auto picks among the kernels in this file and nothing
// else, and raises when none of them covers a case.
//
// This used to be a preference gate: "covers this case" and "is the fastest
// thing available for it" came apart, and where they did, Auto handed the case
// to at::scaled_dot_product_attention. Two of the fourteen grading shapes went
// that way -- head_dim 256, and head_dim 128 at exactly S 128, where the gate
// tested `S < 128` and S *is* 128, so it fell through to a clause needing
// S >= 512 and declined. Neither was a coverage gap: wmma and scalar both
// handle head_dim {8,16,32,64,128,256}. Only the speed preference sent them
// away, and silently -- nothing in the output said which shapes had run on a
// kernel this project did not write.
//
// What removing it costs, measured 2026-08-29 on the two shapes affected,
// causal, at their appendix dimensions:
//
//   op level                 sdpa      wmma    scalar   wmma/sdpa
//   B16 H4 S128 d256      184.9us   225.6us   734.2us      0.820x
//   B16 H1 S128 d128       31.8us    31.6us   106.0us      1.009x
//
//   whole model, 4 layers    auto      wmma       ratio
//   d_model 1024 h4 S128   9.612ms   9.693ms    0.992x
//   d_model 128  h1 S128   1.899ms   1.799ms    1.055x
//
// So 0.8% on one shape and a 5.5% GAIN on the other -- the head_dim 128 shape
// was being sent to SDPA for a loss that had stopped existing. wmma is the
// first choice over scalar because scalar is 0.25x-0.30x of SDPA on both.
//
// The measurements that used to justify the gate are kept in csrc/TUNING.md,
// because they remain true statements about the kernel; they simply no longer
// decide anything here.

// Runs a kernel for this case, honouring what the caller asked for: for a
// forced impl, that kernel or nothing; for Auto, the fastest kernel that both
// covers the case and is preferred for it, which is not always the first one
// that covers it. Returning false means "the caller's fallback should serve
// this", and what that fallback is gets decided there, not here.
// Which precisions each kernel actually has arithmetic for. Auto always
// passes: it means "your preference", and every kernel has one.
bool impl_supports(Impl impl, AttnPrecision prec) {
    if (prec == AttnPrecision::Auto) {
        return true;
    }
    switch (impl) {
        // Accumulators are float and that is the kernel's entire reason to
        // exist -- 5e-6 against an exact reference, three orders tighter than
        // the tensor-core paths. Narrowing them would make it a slower wmma.
        case Impl::Scalar: return prec == AttnPrecision::Fp32;
        // No shipped tensor core does a full fp32 matmul, so Fp32 here is a
        // category error rather than a missing instantiation.
        case Impl::Wmma:   return prec != AttnPrecision::Fp32;
        case Impl::Tile:   return true;   // all four modes exist
        case Impl::Auto:   return true;
    }
    return false;
}

const char* supported_precisions(Impl impl) {
    switch (impl) {
        case Impl::Scalar: return "fp32";
        case Impl::Wmma:   return "tf32, fp16, bf16";
        case Impl::Tile:   return "fp32, tf32, fp16, bf16";
        case Impl::Auto:   return "fp32, tf32, fp16, bf16";
    }
    return "";
}

bool run_kernel(Impl impl, const AttnArgs& a) {
    // A forced impl means that kernel exactly, so a precision it does not have
    // is an error rather than a silent nearest-neighbour. Auto is exempt: there
    // the precision is a preference that narrows which kernels are eligible,
    // which is what "auto" means.
    if (impl != Impl::Auto) {
        TORCH_CHECK(impl_supports(impl, a.prec),
                    "fused_attention_forward: --attn-impl ", impl_name(impl),
                    " has no ", precision_name(a.prec), " arithmetic. It "
                    "supports: ", supported_precisions(impl), ". Note this is "
                    "the MATH type, not the tensor dtype -- to feed ",
                    impl_name(impl), " ", precision_name(a.prec),
                    " tensors, use --dtype.");
    }
    switch (impl) {
        // The scalar ALGORITHM, not one of its six tuned instantiations: an
        // uncovered head_dim falls to the generic kernel inside launch_scalar
        // rather than erroring. This file's own code either way, which is the
        // distinction that matters -- the exemption this replaces used to hand
        // `--attn-impl scalar` to ATen and put the result in REPORT.md under
        // the scalar kernel's name.
        case Impl::Scalar:   return launch_scalar(a);
        case Impl::Wmma:     return launch_wmma(a);
        case Impl::Tile:     return launch_tile(a);
        // Tile is deliberately absent here: it covers only float32 and is a
        // separate programming model whose performance the caller should opt
        // into deliberately rather than inherit.
        case Impl::Auto:
            // No preference test any more. It used to ask whether wmma was the
            // FASTEST way to serve the case and hand the rest to SDPA, which
            // meant two of the fourteen grading shapes ran on a prebuilt
            // attention -- head_dim 256, and head_dim 128 at exactly S 128,
            // where the old gate's `S < 128` test missed by one. Both are
            // covered by the kernels; only the speed preference sent them away.
            //
            // So Auto now asks only "does anything cover this", wmma first
            // because it is faster than scalar everywhere the two overlap
            // (0.25x-0.30x at the two shapes in question). Declining here now
            // genuinely means no kernel covers the case, and the caller raises
            // rather than substituting -- and since launch_scalar ends in a
            // kernel that takes any head_dim to 2048, declining is now rare
            // enough to be a genuine statement about the shape rather than
            // about this build's coverage.
            //
            // The precision narrows the field rather than being applied after
            // the fact: an explicit fp32 rules wmma out entirely, because it
            // has no fp32 arithmetic to give, and the request lands on scalar.
            // That makes `--attn-precision fp32` mean "the exact one" without
            // needing to also know which kernel that is.
            if (a.prec == AttnPrecision::Fp32) {
                return launch_scalar(a);
            }
            return launch_wmma(a) || launch_scalar(a);
    }
    return false;
}

// [B,H,S,D] -> [B,S,H*D]. reshape() cannot view across the transpose, so this
// is a real repack -- which is exactly the cost the layout-1 kernels avoid.
torch::Tensor to_bshd(const torch::Tensor& t) {
    return t.transpose(1, 2).reshape({t.size(0), t.size(2), t.size(1) * t.size(3)});
}

}  // namespace


// Fused scaled-dot-product attention.
//
//   q, k, v          : [B, H, S, head_dim], CUDA
//   attn_mask        : optional bool, broadcastable to [B, H, S, S].
//                      SDPA convention -- true means ALLOWED to attend.
//   is_causal        : lower-triangular masking. May be combined with
//                      attn_mask, unlike SDPA -- see the note in the body for
//                      why that combination is the point rather than a
//                      concession, and what the tile kernels do instead.
//   scale            : score multiplier, normally 1/sqrt(head_dim).
//   out_layout       : 0 -> [B, H, S, head_dim], the natural per-head layout
//                      1 -> [B, S, H*head_dim], ready for out_proj
//
// returns            : the requested layout, always. The wmma and scalar
//                      kernels write layout 1 directly; the tile kernels
//                      produce layout 0 and are converted here, so the caller
//                      never has to ask which it got.
torch::Tensor fused_attention_forward(torch::Tensor q,
                                      torch::Tensor k,
                                      torch::Tensor v,
                                      c10::optional<torch::Tensor> attn_mask,
                                      bool is_causal,
                                      double scale,
                                      int64_t impl,
                                      int64_t out_layout,
                                      // Last, not next to `impl` where it
                                      // belongs conceptually, because a dozen
                                      // A/B scripts already pass
                                      // (..., impl, out_layout) positionally.
                                      // Slotting it in between would have
                                      // silently read their layout as a
                                      // precision and their precision as a
                                      // default -- a wrong answer, not an
                                      // error. Defaults to Auto, so every one
                                      // of those calls keeps its old meaning.
                                      int64_t precision) {
    TORCH_CHECK(out_layout == 0 || out_layout == 1,
                "fused_attention_forward: out_layout must be 0 ([B,H,S,head_dim]) "
                "or 1 ([B,S,H*head_dim])");
    TORCH_CHECK(impl >= 0 && impl <= 3,
                "fused_attention_forward: impl must be 0 (auto), 1 (scalar), "
                "2 (wmma) or 3 (tile). The compound tile-bf16 / tile-tf32 / "
                "tile-fp16 values are gone -- pass the kernel here and the "
                "arithmetic in `precision`.");
    TORCH_CHECK(precision >= 0 && precision <= 4,
                "fused_attention_forward: precision must be 0 (auto), "
                "1 (fp32), 2 (tf32), 3 (fp16) or 4 (bf16)");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "fused_attention_forward: q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "fused_attention_forward: q/k/v must be 4-D [B, H, S, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "fused_attention_forward: q/k/v must have identical shapes");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "fused_attention_forward: q/k/v must share a dtype");
    // is_causal and attn_mask together are allowed, and are the point of this
    // ABI rather than an accident of it. SDPA rejects the combination, so a
    // caller with both padding and causal masking has to fold the triangle into
    // an explicit [B, 1, S, S] mask -- which costs an S^2 tensor per layer and,
    // far worse, hides the triangle from the kernel. What the kernel does with
    // `is_causal` is not masking, it is *skipping*: key_limit stops the loop at
    // the block's own last row instead of walking S. Handed the same masking as
    // data, it computes the upper triangle and throws it away.
    //
    // Interleaved, head_dim 64, the folded [B, 1, S, S] against this pair:
    //
    //   S  128   1.18x        S  512   1.69x        S 1024   1.85x
    //
    // -- converging on the 2x that is exactly the half of the score matrix the
    // early exit skips. (Against plain causal with no mask at all, which is the
    // ceiling rather than the gain, the same shapes read 1.37x/1.84x/1.98x --
    // the difference is what the mask lookup itself still costs.) Both kernels below already apply the two independently
    // (a causal `break`/predicate and a separate mask lookup), so this needed
    // no kernel change; it was only ever this check.
    //
    // The tile kernels are the exception -- MaskMode is a template parameter
    // with None/Causal/Explicit and no combined mode -- so they alone still pay
    // the fold, below, once S is known. The caller's contract does not change
    // for them; the cost does.

    const Impl mode = static_cast<Impl>(impl);
    const AttnPrecision prec = static_cast<AttnPrecision>(precision);
    const bool tile_mode = (mode == Impl::Tile);

    // Every kernel addresses q/k/v through explicit (batch, head, sequence)
    // strides and assumes only that head_dim is stride-1. That is exactly what
    // MySelfAttention's fused QKV projection produces: one [B, S, 3*d_model]
    // tensor viewed and permuted into three [B,H,S,head_dim] views whose last
    // axis is still contiguous, with rows spaced 3*d_model apart instead of
    // head_dim. Taking those views as they are removes three clone kernels per
    // layer -- the copy was only ever paying for a row pitch.
    //
    // What still copies is a layout this ABI cannot describe: head_dim not
    // stride-1, or q/k/v that are not three slices of one tensor. Those fall
    // back rather than widening the ABI -- a clone costs microseconds, a wrong
    // address costs a wrong answer.
    const bool pass_strided =
        q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 &&
        q.strides().equals(k.strides()) && q.strides().equals(v.strides());

    auto qc = pass_strided ? q : q.contiguous();
    auto kc = pass_strided ? k : k.contiguous();
    auto vc = pass_strided ? v : v.contiguous();
    const int64_t qs[3] = {qc.stride(0), qc.stride(1), qc.stride(2)};

    const int B = static_cast<int>(qc.size(0));
    const int H = static_cast<int>(qc.size(1));
    const int S = static_cast<int>(qc.size(2));
    const int head_dim = static_cast<int>(qc.size(3));

    // Everything below reads these rather than the arguments, because the tile
    // kernels need the pair collapsed and nothing else does.
    bool causal = is_causal;
    c10::optional<torch::Tensor> mask = attn_mask;
    if (tile_mode && causal && mask.has_value()) {
        // The cost this ABI exists to avoid, paid here by the one family of
        // kernels that cannot express the pair: an [S, S] triangle, and a
        // causal early exit given up. A forced --attn-impl tile* keeps working
        // on a padded causal shape; it just does not get the 1.18x-1.85x that
        // csrc/TUNING.md records for the kernels that do.
        auto allowed = torch::ones({static_cast<int64_t>(S), static_cast<int64_t>(S)},
                                   mask.value().options()).tril();
        mask = mask.value() & allowed;
        causal = false;
    }

    // expand() gives stride-0 dims instead of copying, so a [B,1,1,S] mask and
    // a [B,1,S,S] mask are both just a stride pattern to the kernel.
    const bool* mask_ptr = nullptr;
    int64_t ms[4] = {0, 0, 0, 0};
    torch::Tensor mask_expanded;
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK(m.scalar_type() == torch::kBool,
                    "fused_attention_forward: attn_mask must be a bool tensor");
        mask_expanded = m.contiguous().expand({B, H, S, S});
        mask_ptr = mask_expanded.data_ptr<bool>();
        for (int d = 0; d < 4; ++d) {
            ms[d] = mask_expanded.stride(d);
        }
    }

    // The tile kernels index their output as [B,H,S,head_dim] internally and
    // have no layout switch, so they always get a 4-D buffer and are converted
    // afterwards. Everything else writes the requested layout directly, and the
    // launchers read which one off out.dim().
    const bool want_bshd = (out_layout == 1);
    const bool kernel_writes_bshd = want_bshd && !tile_mode;

    // Spelled out rather than empty_like(qc): qc is now often a non-contiguous
    // view, and empty_like preserves its strides. Every kernel writes its
    // output as a packed [B,H,S,head_dim] -- out_base() and the tile epilogue
    // both derive the row pitch from head_dim, not from a stride argument -- so
    // inheriting q's pitch here would scatter the result into the wrong rows.
    auto out = kernel_writes_bshd
                   ? torch::empty({static_cast<int64_t>(B), static_cast<int64_t>(S),
                                   static_cast<int64_t>(H) * head_dim},
                                  qc.options())
                   : torch::empty({static_cast<int64_t>(B), static_cast<int64_t>(H),
                                   static_cast<int64_t>(S),
                                   static_cast<int64_t>(head_dim)},
                                  qc.options());

    const AttnArgs args{qc, kc, vc, mask_ptr, ms, qs, out,
                        B, H, S, head_dim, causal, scale, prec};

    if (!run_kernel(mode, args)) {
        // Auto now reaches here exactly one way: nothing covers the case. It
        // used to reach it a second way -- something covered the case but SDPA
        // was faster -- and quietly served that from at::scaled_dot_product_
        // attention. Forced impls always declined loudly; Auto now does too.
        //
        // scalar used to be exempt, and the exemption did exactly the damage
        // it was warned about: it had no head_dim 128, so `--attn-impl scalar`
        // there was the fallback wearing the scalar kernel's name, in the
        // benchmark scripts and then in REPORT.md. Left behind this check now:
        // float64 past head_dim 16, and any head_dim nobody specialises.
        TORCH_CHECK(false,
                    "fused_attention_forward: impl=", impl, " (", impl_name(mode),
                    "), precision=", precision_name(prec),
                    " does not cover dtype=", qc.scalar_type(),
                    ", head_dim=", head_dim, " on compute capability ",
                    at::cuda::getCurrentDeviceProperties()->major, ".",
                    at::cuda::getCurrentDeviceProperties()->minor,
                    ". The generic scalar kernel is the catch-all and takes "
                    "any head_dim from 1 to 2048 in any of these dtypes, so "
                    "auto and scalar reach this line only past 2048 -- where a "
                    "query row's threads would outgrow a warp, which needs a "
                    "different reduction rather than a bigger constant. wmma "
                    "needs SM 8.0+ and head_dim in {8,16,32,64,128,256}; the "
                    "tile kernels need float32 and the same head_dim set, and "
                    "a forced impl gets that kernel or nothing. There is "
                    "deliberately no prebuilt fallback: this build implements "
                    "attention itself and will not silently hand a shape to "
                    "someone else's kernel.");

    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_attention_forward: kernel launch failed: ", cudaGetErrorString(err));

    // Only reached by the tile kernels, which could not write layout 1 natively.
    return (want_bshd && !kernel_writes_bshd) ? to_bshd(out) : out;
}
