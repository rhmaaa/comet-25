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
  \brief Functor performing grouped linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation. (每个操作实际计算的数量 128/element-width )
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_
>
class GroupedLinearCombination {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;
  int capacity = 1; 

  /// Host-constructable parameters structure
  // many initialization method to implement corresponding params 
  struct Params 
  {
    ElementCompute *alpha;                  ///< scales accumulators
    ElementCompute *beta;                   ///< scales source tensor
    ElementCompute **alpha_ptr;              ///< pointer to accumulator vector - if not null, loads it from memory
    ElementCompute **beta_ptr;               ///< pointer to source vector - if not null, loads it from memory

    CUTLASS_HOST_DEVICE
    Params():
      alpha(&ElementCompute(1)),
      beta(&ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute *alpha,
      ElementCompute *beta
    ):
      alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) { }

    // 暂时取消掉这样一个初始化的策略？ 
    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute *alpha
    ):
      alpha(alpha), beta(nullptr), alpha_ptr(nullptr), beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute ** alpha_ptr,
      ElementCompute ** beta_ptr
    ):
      alpha(nullptr), beta(nullptr), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute ** alpha_ptr
    ):
      alpha(nullptr), beta(nullptr), alpha_ptr(alpha_ptr), beta_ptr(nullptr) { }
  };

private:

  //
  // Data members
  //

  ElementCompute *alpha_;
  ElementCompute *beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  GroupedLinearCombination(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha); // here accept alpha from pointers or real value 
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) { // 这里的 k_partition 的设置可以不用管
      beta_ = ElementCompute(1);
    }
  }

  /// Computes intermediate: X = beta * source
  CUTLASS_HOST_DEVICE
  FragmentCompute compute_intermediate(
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    FragmentCompute converted_source = source_converter(source);

    if (Scale == ScaleType::NoBetaScaling) {
      return converted_source;
    }
    else {
      multiplies<FragmentCompute> mul_source;
      return mul_source(beta_, converted_source);
    }
  }

  /// Computes linear scaling with intermediate: D = alpha * accumulator + X
  CUTLASS_HOST_DEVICE
  FragmentOutput with_intermediate(
      FragmentAccumulator const& accumulator,
      FragmentCompute const& intermediate) const {

    // Convert accumulator to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    } else {
      // Perform binary operations
      multiply_add<FragmentCompute> mul_add_accumulator;
      FragmentCompute computed_output = mul_add_accumulator(alpha_, converted_accumulator, intermediate);

      return destination_converter(computed_output);
    }
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling) // 这里计算是否有 beta 这部分的计算内容
      intermediate = converted_source;
    else
      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform

    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X (here comes the computing about alpah )

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;
    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    return destination_converter(intermediate);
  }

  //
  // Specializations for scalar (for use with cute::collective::DefaultEpilogue)
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    [[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type

    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;
    multiply_add<ElementCompute> madd;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = source_converter(source);
    }
    else {
      intermediate = multiply(beta_, source);                            // X =  beta * C + uniform
    }

    intermediate = madd(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;
    ElementCompute converted_accumulator = accumulator_converter(accumulator);

    // Convert to destination numeric type
    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;

    intermediate = multiply(alpha_, accumulator);    // D = alpha * Accum
    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
