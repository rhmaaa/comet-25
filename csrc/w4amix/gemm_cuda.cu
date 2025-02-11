
#include "gemm_cuda.h"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include "./comet_opt_v1/kernel/grouped_gemm.h"
#include "./comet_opt_v1/kernel/default_grouped_gemm.h"
#include "./comet_opt_v1/device/grouped_gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/matrix_coord.h"


// #define KERNEL_LAUNCH_CODE                                                                                                                                               \
//   auto kernel_func =                                                                                         \
//       dense_kernel0<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;                                 \
//   cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
//                        kSmemByteSize);                                                                       \
//   kernel_func<<<num_blocks, threads_per_block, kSmemByteSize>>>(                                             \
//       in_feats, kernel, zeros, scales_i8, wscales, ascales, out_feats, num_in_feats, num_out_channels,       \
//       num_in_channels);


namespace COMET::matmul{


// std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(torch::Tensor b, torch::Tensor batch_sizes) {
//     const size_t num_experts = batch_sizes.size(0);
//     const size_t k = b.size(1), n = b.size(2);
//     std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
//     for (int i = 0; i < num_experts; ++i) {
//         problem_sizes[i] = cutlass::gemm::GemmCoord(batch_sizes.data_ptr<int64_t>()[i], n, k);
//     }
//     return problem_sizes;
// }
inline std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(size_t m,size_t n, torch::Tensor group_k) {
    const size_t num_experts = 4;
    // const size_t k = b.size(1), n = b.size(2);
    std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        problem_sizes[i] = cutlass::gemm::GemmCoord(m,n,group_k.data_ptr<int64_t>()[i]);//[1][2]    [1,2,3]
     }
    return problem_sizes;
}


template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
    size_t bytes = x.size() * sizeof(T);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
    torch::Tensor out = torch::empty(bytes, options);

    cudaMemcpyAsync(out.data_ptr(),
                    x.data(), bytes,
                    cudaMemcpyHostToDevice);
    return out;
}

template <typename Gemm>
typename Gemm::Arguments MakeArguments(torch::Tensor a,
                torch::Tensor b,
                torch::Tensor c,
                torch::Tensor batch_sizes) {
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = c.size(-1);
    auto problem_sizes_host = MakeProblemSizes(m, n, batch_sizes);
    cutlass::gemm::GemmCoord problem_size(m, n, k);                
    // Calculate the number of threadblocks to use and validate the result.
    int64_t num_experts = 4;

    // NOTE: This is borrowed from FasterTransformer.
    int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);

    if (!threadblock_count) {
        TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
    }

    // Create the host arrays of leading dimension data and pointer data.
    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
    std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
    std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
    int64_t elements_a = 0, elements_b = 0, elements_c = 0;

    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;
    std::vector<ElementA *> ptr_a_host(num_experts);
    std::vector<ElementB *> ptr_b_host(num_experts); 
    std::vector<ElementC *> ptr_c_host(num_experts);
    std::vector<int> type_host(num_experts);
    using OutputParams = typename Gemm::EpilogueOutputOp::Params;
    std::vector<OutputParams> output_params_host(num_experts);

    for (int i = 0; i < num_experts; ++i) {

        if(i < num_experts-1) {
            type_host[i] = 0;
        }else {
            type_host[i] = 1;
        }
        auto problem = problem_sizes_host[i];
        lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

        offsets_a[i] = elements_a;
        offsets_b[i] = elements_b;
        offsets_c[i] = elements_c;

        ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
        ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
        ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

        elements_a += problem.m() * problem.k();
        elements_b += problem.k() * problem.n();
        elements_c += problem.m() * problem.n();
        OutputParams output_op(1, 0);
        output_params_host[i] = output_op;
    }

    // Copy the problem sizes, pointers and leading dimension data to the device.
    torch::Tensor lda = CopyToDevice(lda_host, a.device());
    torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
    torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
    torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
    torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
    torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
    torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());
    torch::Tensor type = CopyToDevice(type_host, a.device());
    torch::Tensor output_params = CopyToDevice(output_params_host, a.device());

    using TensorRefOut = cutlass::TensorRef<ElementC, LayoutC>;
    // using  OutputTensorRef =  cutlass::gemm::device::TensorRef<ElementC, LayoutC>;
    TensorRefOut output_ref(static_cast<ElementC *> (c.data_ptr()), problem_size.n());
    cutlass::gemm::GemmCoord pz(m,n,k);
    typename Gemm::Arguments arguments(
                    pz,
                    (int*)type.data_ptr(),
                    output_ref,
                    (cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
                    (int)num_experts,
                    (int)threadblock_count,
                    (OutputParams*)output_params.data_ptr(),
                    (ElementA**)ptr_a.data_ptr(),
                    (ElementB**)ptr_b.data_ptr(),
                    (ElementC**)ptr_c.data_ptr(),
                    (ElementC**)ptr_c.data_ptr(),
                    /*lda=*/(int64_t*)lda.data_ptr(),
                    /*ldb=*/(int64_t*)ldb.data_ptr(),
                    /*ldc=*/(int64_t*)ldc.data_ptr(),
                    /*ldd=*/(int64_t*)ldc.data_ptr(),
                    (cutlass::gemm::GemmCoord*)problem_sizes_host.data());


                    
    return arguments;
}

void w4amix_gemm_forward_new(torch::Tensor _in_feats,
                        torch::Tensor _kernel,
                        torch::Tensor _group,
                        torch::Tensor _wscales,
                        torch::Tensor _ascales,
                        torch::Tensor _out_feats)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<int8_t *>(_in_feats.data_ptr<int8_t>());
    auto kernel = reinterpret_cast<int8_t *>(_kernel.data_ptr<int8_t>());
    auto wscales = reinterpret_cast<half2 *>(_wscales.data_ptr());
    auto ascales = reinterpret_cast<half *>(_ascales.data_ptr());
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

    // constexpr int G = 128;

    using ElementInput = int8_t;
    using ElementA1 = cutlass::int4b_t;
    using ElementA0 = int8_t;
    using WeightType = int8_t;
    using ElementB = cutlass::int4b_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGroupedGemm<
        ElementA0,
        cutlass::layout::RowMajor,
        ElementA1,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        16,
        16,
        ElementB,
        cutlass::layout::ColumnMajor,
        cutlass::ComplexTransform::kNone,
        16,
        ElementAccumulator,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,  // int8 的目前都暂时设为这个吧
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::gemm::GemmShape<128, 128, 128>, // int4 tensor core
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::epilogue::thread::LinearCombination<ElementAccumulator,
                                                    128 / cutlass::sizeof_bits<ElementAccumulator>::value,
                                                    ElementAccumulator,
                                                    ElementAccumulator>,
        // cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, // stream k 初始化不一样
        ThreadblockSwizzle,
        2,
        cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>::GemmKernel;

    using Gemm = cutlass::gemm::device::GroupedGemm<GemmKernel>;

    using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
    using ElementCompute   = typename EpilogueOutputOp::ElementCompute;
    using OutputParams     = typename EpilogueOutputOp::Params;
    using GemmCoord = cutlass::gemm::GemmCoord;
    Gemm gemm;
    // GemmCoord problem
    
    torch::Tensor batch_sizes = torch::tensor({num_in_channels/4,num_in_channels/4,num_in_channels/4,num_in_channels/4});

    std::vector<int> batch_vec = {num_in_channels/4,num_in_channels/4,num_in_channels/4,num_in_channels/4};
    Gemm::Arguments args = MakeArguments<Gemm>(_in_feats, _kernel, _out_feats, batch_sizes );
    
    // int threadblock_count = Gemm::sufficient(problem_sizes.data(), problem_count());

    int64_t workspace_size = gemm.get_workspace_size(args);
    printf("workspace_size is %ld \n",workspace_size);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(_in_feats.device());
    torch::Tensor workspace = torch::empty(workspace_size, options);
    // torch::Tensor workspace(workspace_size);
    // gemm.initialize(args, workspace.data_ptr());
    if(gemm.initialize(args, workspace.data_ptr()) != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
    }
    if(gemm.run() != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
      }
    // cutlass::Status status = gemm.run();
    printf("run our kernel done\n");
}


}