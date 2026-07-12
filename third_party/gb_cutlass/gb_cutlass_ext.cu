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
#include <cuda_runtime.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_nvfp4", &gemm_nvfp4, "sm_120a block-scaled NVFP4xNVFP4 -> bf16 GEMM");
}
