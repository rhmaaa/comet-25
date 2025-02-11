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
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,              ///< shape of CTA        (concept: MatrixShape)
  typename OutputOp_ ,          ///< output operator     (concept: epilogue::thread operator)
  typename ReductionOp_,        ///< reduction operator  (concept: ReductionOperator)
  int PartitionsPerStage = 4    ///< number of partitions to issue 
>
class GroupedReduction {
public:

  using Shape = Shape_;
  using ReductionOp = ReductionOp_;
  // using OutputOp = OutputOp_; //! 这里不需要进行 outputop 的操作，因此 不需要传统的outputop
  static int const kElementsPerAccess = OutputOp_::kCount;
  static int const kPartitionsPerStage = PartitionsPerStage;

  using ElementSource = typename ReductionOp::Element;
  using ElementOutput = typename ReductionOp::ElementAccumulator;

  using SourceTensorRef = TensorRef<ElementSource, layout::RowMajor>;
  using OutputTensorRef = TensorRef<ElementOutput, layout::RowMajor>;
  using StrideIndex = typename SourceTensorRef::Layout::Stride::Index;

  using LayoutSource = typename SourceTensorRef::Layout;

  using FragmentSource = AlignedArray<ElementSource, kElementsPerAccess>; //? 根据对齐内容的计算，判定需要对其所采用的fragement workspace 的大小
  using FragmentAccumulator = Array<ElementOutput, kElementsPerAccess>;
  using FragmentOutput = AlignedArray<ElementOutput, kElementsPerAccess>;

  //
  // Types
  //

  /// Params structure
  struct Params {

    MatrixCoord problem_size;
    int partitions;
    size_t partition_stride;
    OutputTensorRef destination;
    // SourceTensorRef source;
    ElementSource ** source;
    typename ReductionOp::Params reduction;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      MatrixCoord problem_size_,
      int partitions_,
      size_t partition_stride_,
      OutputTensorRef destination_,
      ElementSource ** source_,
      typename ReductionOp::Params reduction_ = typename ReductionOp::Params()
    ):
      problem_size(problem_size_),
      partitions(partitions_),
      partition_stride(sizeof(FragmentSource) * partition_stride_ / kElementsPerAccess), //? 这里为什么要进行这样的计算内容？ 
      destination(destination_),
      source(source_),
      reduction(reduction_) { }
  };

  struct SharedStorage { };


public:

  /// Computes the grid size given a chosen threadblock shape
  CUTLASS_HOST_DEVICE
  static dim3 grid_shape(
    cutlass::MatrixCoord problem_size) {

    return dim3(
      (problem_size.row() + Shape::kRow - 1) / Shape::kRow,   //row 
      (problem_size.column() + Shape::kColumn - 1) / Shape::kColumn); // column
  }

  /// Determines the threadblock shape
  CUTLASS_HOST_DEVICE
  static dim3 block_shape() {
    return dim3(Shape::kColumn / kElementsPerAccess, Shape::kRow);// cloumn row
    // return dim3(Shape::kColumn / kElementsPerAccess, Shape::kRow);
  }

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &storage) {
    // printf("reduce kernel problem size is %d \n",params.problem_size.row());
    // printf("reduce kernel problem size is %d \n",params.problem_size.column());
    // Determine CTA position
    MatrixCoord thread_offset(
      MatrixCoord::Index(int(blockIdx.x) * Shape::kRow + threadIdx.y),
      MatrixCoord::Index(int(blockIdx.y) * Shape::kColumn + threadIdx.x * kElementsPerAccess)
      // MatrixCoord::Index(int(blockIdx.y) * Shape::kColumn + threadIdx.x * kElementsPerAccess),
      // MatrixCoord::Index(int(blockIdx.x) * Shape::kRow + threadIdx.y)
    );

    // One guard conditional
    if (!(thread_offset.row() < params.problem_size.row() && 
          thread_offset.column() < params.problem_size.column())) {

      return;
    }
    // printf("thread_offset.row() %ld \n", thread_offset.row());
    // printf("thread_offset stride %ld \n", thread_offset.stride());

    ReductionOp reduction_op(params.reduction);

    FragmentAccumulator output_frag; //? 这里声明了一个 accumulator, 如何确定其有多大呢？ 即这个 FragmentAccumulator 的 memory size

    output_frag.clear();  
    
    //
    // Load the first slice 
    //? 这里是如何判断当前的 slice 在哪个位置的，地址参数是如何进行传递的
    //

    SourceTensorRef source_ref;

    char const *source_ptr; 

    FragmentSource source_frag[kPartitionsPerStage];

    //
    // Load and accumulate with a simple batched loading sequence.
    //
    //? 为什么这里的加载和 reduction op 需要分开计算，但是看起来这两部分可以合并起来进行处理，因为展开实际上可以并行处理，将数据的加载和计算分别进行处理
    // printf("params.partitions is : %d \n", params.partitions);
    // printf("kPartitionsPerStage is : %d \n", kPartitionsPerStage);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < params.partitions; k += kPartitionsPerStage) { // 看起来是用 k 来控制每次 partition 的次数
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) {
      // for (int i = 0; i < 3; ++i) {
        if (k + i < params.partitions) { //? 判定对应的内容不会越界？ stage 表示每次 reduction 的部分？ 
          source_ref.reset(params.source[k + i]); // 修改了对应的配置内容
          // printf("source_ref.offset(thread_offset) : %ld \n", source_ref.offset(thread_offset));
          // source_ptr = reinterpret_cast<char const *> (source_ref.data() + source_ref.offset(thread_offset));
          source_ptr = reinterpret_cast<char const *> (source_ref.data() + thread_offset.column() + thread_offset.row()*params.problem_size.row());
          // source_ptr = reinterpret_cast<char const *> (source_ref.data() );

          source_frag[i] = *reinterpret_cast<FragmentSource const *>(source_ptr); // reinterpret_cast 是一个类型转换操作符，能够在不同类型之间进行强制的转换
          // source_ptr += params.partition_stride; // 每次地址都需要增加这么长的位置？ 这覆盖了整个的 output 
        }
      }   

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) { //? 为什么需要设置 partitionsPerStage ？ 这个有什么用啊
        if (k + i < params.partitions) {
          output_frag = reduction_op(output_frag, source_frag[i]); // partitions 表示了 split-k 划分的次数，进行相应的划分内容 （会进行 reduction_op) 我们应该希望在 reduction-op 中，执行相应的 dquantization 
        }
      }
    }
    //
    // Store 
    //

    FragmentOutput *dest_ptr = reinterpret_cast<FragmentOutput *>(
      params.destination.data() + params.destination.offset(thread_offset)); // compute the destination address and store corresponding results 

    *dest_ptr = reinterpret_cast<FragmentOutput const &>(output_frag);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass
