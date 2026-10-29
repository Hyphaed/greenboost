// SPDX-License-Identifier: Apache-2.0
// GreenBoost Stage-A2: sm_120a block-scaled NVFP4 GEMM as a torch extension.
//
// Adapted from CUTLASS example 79b (blackwell_geforce_nvfp4_nvfp4_gemm) — the
// GeForce Blackwell (sm_120a) block-scaled NVFP4×NVFP4 tensor-op GEMM — with a
// plain bf16 output (LinearCombination epilogue) instead of the example's
// block-scaled fp4 output, since gb_quant consumes a bf16 activation/result.
//
// Exposes one entry: gemm_nvfp4(A_e2m1, SFA, B_e2m1, SFB, M, N, K) -> bf16 [M,N]
//   A_e2m1 : uint8 [M, K/2]   two float_e2m1 packed per byte (row-major)
//   SFA    : uint8 [ ... ]     e4m3 block scale factors, interleaved layout
//   B_e2m1 : uint8 [N, K/2]   weight, column-major w.r.t. the GEMM (see below)
//   SFB    : uint8 [ ... ]     e4m3 block scale factors for B
//
// IMPORTANT (why this ships inert): the packed NVFP4 layout gb_quant's
// _init_nvfp4() produces has NOT yet been reconciled against CUTLASS's
// interleaved scale-factor layout (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF*)
// nor benched for numerics on real weights. gb_cutlass.available() therefore
// requires GB_CUTLASS_ENABLE=1 on top of a successful build, so a compiled-
// but-unvalidated extension can never silently route a production GEMM. Flip
// that env only after tests/bench/bench_cutlass_nvfp4.py passes on the box.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace {

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // packed fp4 + block SF
using LayoutATag = cutlass::layout::RowMajor;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

// bf16 output via the default LinearCombination epilogue (no block-scaled D).
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace

// D[M,N] (bf16) = A[M,K] (nvfp4) @ B[N,K]^T (nvfp4).  Pointers are raw device
// bytes reinterpreted as the CUTLASS element types.
torch::Tensor gemm_nvfp4(torch::Tensor A_e2m1, torch::Tensor SFA,
                         torch::Tensor B_e2m1, torch::Tensor SFB,
                         int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A_e2m1.is_cuda() && B_e2m1.is_cuda(), "operands must be CUDA");
    auto opts = torch::TensorOptions()
                    .dtype(torch::kBFloat16)
                    .device(A_e2m1.device());
    torch::Tensor D = torch::empty({M, N}, opts);

    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape((int)M, (int)K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape((int)N, (int)K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape((int)M, (int)N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape((int)M, (int)N, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)M, (int)N, (int)K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)M, (int)N, (int)K, 1));

    using ElemA  = typename ElementA::DataType;
    using ElemSF = typename ElementA::ScaleFactorType;
    using ElemB  = typename ElementB::DataType;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {(int)M, (int)N, (int)K, 1},
        {reinterpret_cast<const ElemA*>(A_e2m1.data_ptr()), stride_A,
         reinterpret_cast<const ElemB*>(B_e2m1.data_ptr()), stride_B,
         reinterpret_cast<const ElemSF*>(SFA.data_ptr()), layout_SFA,
         reinterpret_cast<const ElemSF*>(SFB.data_ptr()), layout_SFB},
        {{1.0f, 0.0f},
         nullptr, stride_C,
         reinterpret_cast<ElementD*>(D.data_ptr()), stride_D}};

    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto ws_opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_e2m1.device());
    torch::Tensor workspace = torch::empty({(int64_t)ws}, ws_opts);

    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass can_implement failed");
    st = gemm.initialize(args, workspace.data_ptr());
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass initialize failed");
    st = gemm.run(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass run failed");
    return D;
}

// ---------------------------------------------------------------------------
// Scale-factor interleaving (the "weight-packing reconciliation", 2026-08-22).
//
// gb_cutlass.pack_nvfp4() produces a FLAT scale array: one uint8 (ue4m3) per
// 16-element block, row-major [MN, ceil(K/16)]. The GEMM instead indexes its SF
// operands through Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}, whose atom
// is
//     Shape <Shape<_32,_4>, Shape<SFVecSize,_4>>
//     Stride<Stride<_16,_4>, Stride<        _0,_1>>
// i.e. a K-major swizzled atom addressed in ELEMENT granularity on the K mode,
// with a stride-0 sub-mode of extent SFVecSize -- that zero stride is what makes
// all 16 elements of a block resolve to the same scale byte.
//
// Deriving that mapping by hand in Python was explicitly declined upstream in
// this file's own header, because a wrong guess yields silently WRONG NUMERICS
// rather than a compile error. So it is not re-derived here: the scatter below
// evaluates THE SAME layout object the GEMM will use, so the two cannot
// disagree by construction. Any future change to the CUTLASS atom follows
// automatically.
template <class Layout>
__global__ void gb_scatter_sf_kernel(const uint8_t* __restrict__ flat,
                                     uint8_t* __restrict__ out,
                                     Layout layout, int MN, int Kb, int sfvec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN * Kb) return;
    int mn = idx / Kb;
    int kb = idx % Kb;
    // k in ELEMENT granularity; the atom's stride-0 mode collapses the block.
    out[layout(cute::make_coord(mn, kb * sfvec, 0))] = flat[idx];
}

template <bool IsA>
static torch::Tensor gb_pack_sf(torch::Tensor flat, int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(flat.is_cuda(), "flat SF must be CUDA");
    TORCH_CHECK(flat.scalar_type() == torch::kUInt8, "flat SF must be uint8");
    TORCH_CHECK(flat.is_contiguous(), "flat SF must be contiguous");

    constexpr int sfvec = Sm1xxBlkScaledConfig::SFVecSize;
    const int64_t MN = IsA ? M : N;
    const int64_t Kb = (K + sfvec - 1) / sfvec;
    TORCH_CHECK(flat.numel() == MN * Kb,
                "flat SF has ", flat.numel(), " elements, expected ", MN * Kb);

    auto shape = cute::make_shape((int)M, (int)N, (int)K, 1);
    auto layout = IsA ? Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape)
                      : Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape);

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(flat.device());
    // cosize = codomain size, i.e. exactly the buffer the GEMM will address.
    torch::Tensor out = torch::zeros({(int64_t)cute::cosize(layout)}, opts);

    const int total = (int)(MN * Kb);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    gb_scatter_sf_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        flat.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), layout,
        (int)MN, (int)Kb, sfvec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor pack_sfa(torch::Tensor flat, int64_t M, int64_t N, int64_t K) {
    return gb_pack_sf<true>(flat, M, N, K);
}
torch::Tensor pack_sfb(torch::Tensor flat, int64_t M, int64_t N, int64_t K) {
    return gb_pack_sf<false>(flat, M, N, K);
}

// ---------------------------------------------------------------------------
// Fused activation quantizer (2026-08-22).
//
// Why this exists: with the scale-factor layout reconciled, the NVFP4 GEMM
// measured 1.6x-6.3x faster than bf16 and yet 2.5x-5x SLOWER end to end,
// because gb_cutlass.pack_nvfp4() quantizes the activation in PyTorch with an
// argmin over the 8-entry E2M1 table -- roughly 8x the memory traffic of a
// plain read, costing 0.13-3.74 ms against a 0.017-0.164 ms GEMM. Weights are
// packed once offline and never mattered; activations are packed every forward
// and dominated everything.
//
// This does the whole thing in ONE pass over the activation: block absmax,
// UE4M3 scale encode, E2M1 round, nibble pack, and the scale-factor scatter --
// one thread per 16-element block, each reading its 16 elements exactly once.
// The encodings are CUTLASS's own numeric conversions, not reimplementations,
// for the same reason pack_sfa uses the kernel's own layout object.
// __nv_bfloat16 has no implicit float conversion in device code.
template <class T>
__device__ __forceinline__ float gb_to_float(T v) { return static_cast<float>(v); }
template <>
__device__ __forceinline__ float gb_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <class T, class Layout>
__global__ void gb_quant_act_kernel(const T* __restrict__ src,
                                    uint8_t* __restrict__ packed,
                                    uint8_t* __restrict__ sf_out,
                                    Layout layout, int M, int K, int Kb, int sfvec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * Kb) return;
    const int m = idx / Kb, kb = idx % Kb;
    const T* p = src + (size_t)m * K + (size_t)kb * sfvec;

    float amax = 0.f;
    for (int i = 0; i < sfvec; ++i) amax = fmaxf(amax, fabsf(gb_to_float(p[i])));

    // 6.0 is the largest E2M1 magnitude, so amax/6 maps the block onto the grid.
    cutlass::float_ue4m3_t sfe(amax * (1.0f / 6.0f));
    sf_out[layout(cute::make_coord(m, kb * sfvec, 0))] = sfe.storage;

    // Divide by the ROUNDED scale, not the ideal one: dequant only ever has the
    // stored byte, so encode and decode must agree on which scale was used.
    const float actual = static_cast<float>(sfe);
    const float inv = (actual > 0.f) ? (1.0f / actual) : 0.f;

    uint8_t* out = packed + (size_t)m * (K / 2) + (size_t)kb * (sfvec / 2);
    for (int i = 0; i < sfvec; i += 2) {
        cutlass::float_e2m1_t lo(gb_to_float(p[i])     * inv);
        cutlass::float_e2m1_t hi(gb_to_float(p[i + 1]) * inv);
        out[i / 2] = (uint8_t)((lo.storage & 0xF) | ((hi.storage & 0xF) << 4));
    }
}

// Returns {packed [M, K/2] uint8, SFA in CUTLASS interleaved layout}.
std::vector<torch::Tensor> quantize_act_nvfp4(torch::Tensor A, int64_t M,
                                              int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda(), "activation must be CUDA");
    TORCH_CHECK(A.is_contiguous(), "activation must be contiguous");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == M && A.size(1) == K,
                "activation must be [M, K]");
    constexpr int sfvec = Sm1xxBlkScaledConfig::SFVecSize;
    TORCH_CHECK(K % sfvec == 0, "K must be a multiple of ", sfvec);

    const int64_t Kb = K / sfvec;
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)M, (int)N, (int)K, 1));

    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    torch::Tensor packed = torch::empty({M, K / 2}, u8);
    torch::Tensor sf = torch::zeros({(int64_t)cute::cosize(layout)}, u8);

    const int total = (int)(M * Kb);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (A.scalar_type() == torch::kBFloat16) {
        gb_quant_act_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            packed.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), layout,
            (int)M, (int)K, (int)Kb, sfvec);
    } else if (A.scalar_type() == torch::kFloat) {
        gb_quant_act_kernel<<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            sf.data_ptr<uint8_t>(), layout, (int)M, (int)K, (int)Kb, sfvec);
    } else {
        TORCH_CHECK(false, "activation must be bfloat16 or float32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, sf};
}

int64_t sf_vec_size() { return (int64_t)Sm1xxBlkScaledConfig::SFVecSize; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_nvfp4", &gemm_nvfp4, "sm_120a block-scaled NVFP4xNVFP4 -> bf16 GEMM");
    m.def("pack_sfa", &pack_sfa, "flat [M,ceil(K/16)] ue4m3 -> CUTLASS interleaved SFA");
    m.def("pack_sfb", &pack_sfb, "flat [N,ceil(K/16)] ue4m3 -> CUTLASS interleaved SFB");
    m.def("sf_vec_size", &sf_vec_size, "NVFP4 scale-factor vector size");
    m.def("quantize_act_nvfp4", &quantize_act_nvfp4,
          "fused amax + E2M1 encode + SF scatter -> {packed, SFA}");
}
