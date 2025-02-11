/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

// #include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

#include "grouped_gemm.h"

#include "../layout/ft_gemm_configs.h"
#include "default_fpA_intB_traits.h"
#include "../threadblock/default_mma.h"
#include "../threadblock/default_dq_mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA0_,
    typename ElementA1_,
    typename LayoutA1_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    int kAlignmentA1,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape1,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_ = GroupScheduleMode::kDeviceOnly,
    /// Operation performed by GEMM
    typename Operator = typename device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA1_, ElementB_, ElementC_,
        ElementAccumulator>::Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    ///
    typename Enable = void
    >
struct DefaultGroupedGemm;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    typename ElementA,
    typename LayoutA,
    typename ElementA1,
    typename LayoutA1,
    int kAlignmentA,
    int kAlignmentA1,
    typename ElementB,
    typename LayoutB,
    int kAlignmentB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename OperatorClass,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename ThreadblockShape1,
    typename WarpShape1,
    typename InstructionShape1,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    GroupScheduleMode GroupScheduleMode_,
    typename Operator,
    SharedMemoryClearOption SharedMemoryClear,
    typename PermuteDLayout
>
struct DefaultGroupedGemm<
  ElementA,
  LayoutA,
  ElementA1,
  LayoutA1,
  ComplexTransform::kNone,   // transform A
  kAlignmentA,
  kAlignmentA1,
  ElementB,
  LayoutB,
  ComplexTransform::kNone,   // transform B
  kAlignmentB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  ThreadblockShape1,
  WarpShape1,
  InstructionShape1,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  GroupScheduleMode_,
  Operator,
  SharedMemoryClear,
  PermuteDLayout,
  typename platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type 
> {

  // static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;
  static bool const kInternalTranspose = false;
  using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementA, ElementB, ArchTag>;
  using DefaultGemmKernel1 =typename cutlass::gemm::kernel::DefaultGemm<
        ElementA,
        LayoutA,
        MixedGemmArchTraits::ElementsPerAccessA,
        ElementB,
        typename MixedGemmArchTraits::LayoutB,
        MixedGemmArchTraits::ElementsPerAccessB,
        ElementC,
        // ElementA,
        LayoutC,
        ElementAccumulator,
        typename MixedGemmArchTraits::OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOutputOp,
        // typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        true,
        typename MixedGemmArchTraits::Operator,
        SharedMemoryClear,
        false,
        false,
        false,
        PermuteDLayout>::GemmKernel;

  
  using DefaultGemmKernel0 = typename kernel::DefaultGemm<
    ElementA1,
    LayoutA1,
    kAlignmentA1,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape1,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear,
    false, /*GatherA*/
    false, /*GatherB*/
    false, /*ScatterD*/
    PermuteDLayout
  >::GemmKernel;

    /// Define the kernel in terms of the default kernel
  using GemmKernel = kernel::GroupedGemm<
    typename DefaultGemmKernel0::Mma,
    typename DefaultGemmKernel0::Mma,
    typename DefaultGemmKernel0::Epilogue,
    ThreadblockSwizzle,
    GroupScheduleMode_,
    kInternalTranspose
  >;

};

} // namespace kernel 
} // namespace gemm
} // namespace cutlass