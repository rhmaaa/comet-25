#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "../arch/mma.h"
#include "mixed_gemm_B_layout.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits {
};

template<typename arch>
struct MixedGemmArchTraits<float, float, arch> {
    static constexpr int Stages = 2;
    using OperatorClass         = cutlass::arch::OpClassSimt;
    using AccType               = float;
    using LayoutB               = cutlass::layout::RowMajor;

    static constexpr int ElementsPerAccessA = 1;
    static constexpr int ElementsPerAccessB = 1;
    static constexpr int ElementsPerAccessC = 1;
    static constexpr int ThreadblockK       = 8;
    using InstructionShape                  = cutlass::gemm::GemmShape<1, 1, 1>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

// ========================= Volta Traits ===========================
// Volta will always dequantize after the global memory load.
// This will instantiate any HMMA tensorcore kernels for Volta.
// Note that volta does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm70>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = cutlass::gemm::GemmShape<8, 8, 4>;

    using Operator = typename LayoutDetails::Operator;
};

// ======================= Turing Traits ==============================
// Note that turing does not have native bfloat support so weights and activations will be casted to fp16
// and compute will happen in fp16 then will be converted for bf16 output.
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm75>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 8>;

    using Operator = typename LayoutDetails::Operator;
};

// ======================= Ampere Traits ==============================
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm80,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm80>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType       = float;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 16>; // 这里int8也是可以的

    using Operator = typename LayoutDetails::Operator; // 这个operator ??? 
};


template<>
struct MixedGemmArchTraits<
    int8_t,
    cutlass::int4b_t,
    cutlass::arch::Sm80> {
private:
    using LayoutDetails = LayoutDetailsB<cutlass::int4b_t, cutlass::arch::Sm80>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType       = int32_t;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<int8_t>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    // 这里输出的应该是int32类型的数据
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<int32_t>::value;
    // static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<int8_t>::value;
    
    // <16, 8, 32>,  ref: cutlass 
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 32>; // 这里int8也是可以的

    using Operator = typename LayoutDetails::Operator; // 这个operator ??? 
};



}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass