#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "../arch/mma.h"

#include "dq_mma_multistage.h"
#include "../warp/default_mma_tensor_op.h"
#include "../warp/mma_tensorop_compute_B_with_f16.h"
#include "../layout/tile_interleaved_layout.h"

#include "default_dq_mma.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template<
    /// Type for elementA
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
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
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA,
             LayoutA,
             kAlignmentA,
             ElementB,
             LayoutB,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {
    // ! 看来还要实现一下int8 int4的
    // 直接兼容 应该还会存在更加底层mma的bug
    // 这里为什么只能是16呢
    // static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value || platform::is_same<ElementA, int8_t>::value,
    //               "Element A must be fp16 or bf16");

    // static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
    //               "Mma multistage must dequantize after ldsm");

    // static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value || platform::is_same<ElementB, cutlass::int4b_t>::value,
    //               "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    // 这里也是A和B 
    // 
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB>;

    // ThreadMap for scale iterator
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    using SmemIteratorScale = IteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
};

template<
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
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
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear,
    ///
    int RowsPerTile,//  这个参数也对取数据产生影响
    ///
    int ColumnsInterleaved>
struct DqMma<ElementA,
             LayoutA,
             kAlignmentA,
             ElementB,
             layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

    // static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value || platform::is_same<ElementA, int8_t>::value ,
    //               "Element A must be fp16 or bf16");

    // static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
    //               "Mma multistage must dequantize after ldsm");

    // static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value || platform::is_same<ElementB, cutlass::int4b_t>::value,
    //               "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        layout::ColumnMajor,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    // A的数据是 rowMajor的没有啥特别的
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

private:
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");
    /*
     origial ThreadMap ->
    */
   // original 使用的应该是 int8 int8 -> colunmMajor的 ThreadMap
    using OriginalThreadMap       = typename MmaCore::IteratorThreadMapB;
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    // 经过permute的数据的形状
    using GmemIteratorShape =
        MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    // 这里应该就是经过在原始的数据排布上进行map变换
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>,
        OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
                                 OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<GmemIteratorShape, ElementB, layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

    // ThreadMap for scale iterator
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    using SmemIteratorScale = IteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
};


template<
    /// Type for element A
    // typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    // typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
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
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear,
    ///
    int RowsPerTile,//  这个参数也对取数据产生影响
    ///
    int ColumnsInterleaved>
struct DqMma<int8_t,
             LayoutA,
             kAlignmentA,
             cutlass::int4b_t,
             layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

    // static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value || platform::is_same<ElementA, int8_t>::value ,
    //               "Element A must be fp16 or bf16");

    // static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
    //               "Mma multistage must dequantize after ldsm");

    // static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value || platform::is_same<ElementB, cutlass::int4b_t>::value,
    //               "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<int8_t>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<int4b_t>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        int8_t,
                                                                        LayoutA,
                                                                        int4b_t,
                                                                        layout::ColumnMajor,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<int8_t, kAlignmentA>;
    // A的数据是 rowMajor的没有啥特别的
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        int8_t,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

private:
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");
    /*
     origial ThreadMap ->
    */
   // original 使用的应该是 int8 int8 -> colunmMajor的 ThreadMap
    using OriginalThreadMap       = typename MmaCore::IteratorThreadMapB;
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    // 经过permute的数据的形状
    using GmemIteratorShape =
        MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    // 这里应该就是经过在原始的数据排布上进行map变换
    // 
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>,
        OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
                                 OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<cutlass::int4b_t>::value>;

public:
    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<cutlass::int4b_t, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<GmemIteratorShape, cutlass::int4b_t, layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

    // ThreadMap for scale iterator
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    using SmemIteratorScale = IteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<int8_t,
                                                                    cutlass::int4b_t,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
};


// template<
//     /// Type for elementA
//     // typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Type for element B
//     // typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for the input scale
//     typename ElementScale,
//     /// Layout for the scale operand
//     typename LayoutScale,
//     /// Access granularity of Scales in unit of elements
//     int kAlignmentScale,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Operator class tag
//     typename OperatorClass,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Stages in GEMM
//     int kStages,
//     ///
//     typename Operator,
//     ///
//     SharedMemoryClearOption SharedMemoryClear>
// struct DqMma<int8_t,
//              LayoutA,
//              kAlignmentA,
//              cutlass::int4b_t,
//              LayoutB,
//              kAlignmentB,
//              ElementScale,
//              LayoutScale,
//              kAlignmentScale,
//              ElementAccumulator,
//              layout::RowMajor,
//              OperatorClass,
//              ArchTag,
//              ThreadblockShape,
//              WarpShape,
//              InstructionShape,
//              kStages,
//              Operator,
//              SharedMemoryClear> {
//     // ! 看来还要实现一下int8 int4的
//     // static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value || platform::is_same<ElementA, int8_t>::value,
//     //               "Element A must be fp16 or bf16 or  int8");

//     // // ?? 这是？ 
//     // static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
//     //               "Mma multistage must dequantize after ldsm");

//     // static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
//     //               "Element B must be uint8 or uint4");

//     // static_assert(platform::is_same<ElementB, cutlass::int4b_t>::value,
//     //               "Element B must be int4 ");

//     static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<int8_t>::value * kAlignmentA) == 128) ?
//                                                                     cutlass::arch::CacheOperation::Global :
//                                                                     cutlass::arch::CacheOperation::Always;

//     static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<cutlass::int4b_t>::value * kAlignmentB) == 128) ?
//                                                                     cutlass::arch::CacheOperation::Global :
//                                                                     cutlass::arch::CacheOperation::Always;

//     // Define the MmaCore components
//     // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
//     using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
//                                                                         WarpShape,
//                                                                         InstructionShape,
//                                                                         // ElementA,
//                                                                         int8_t,
//                                                                         LayoutA,
//                                                                         // ElementB,
//                                                                         cutlass::int4b_t,
//                                                                         LayoutB,
//                                                                         ElementAccumulator,
//                                                                         layout::RowMajor,
//                                                                         OperatorClass,
//                                                                         std::max(kStages, 3),
//                                                                         Operator,
//                                                                         false,
//                                                                         CacheOpA,
//                                                                         CacheOpB>;

//     // Define iterators over tiles from the A operand
//     using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
//     using AccessTypeA = cutlass::Array<int8_t, kAlignmentA>;
//     using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
//         cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
//         // ElementA,
//         int8_t,
//         LayoutA,
//         1,
//         ThreadMapA,
//         AccessTypeA>;

//     // Define iterators over tiles from the B operand
//     using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
//     using AccessTypeB = cutlass::Array<cutlass::int4b_t, kAlignmentB>;
//     using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
//         cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
//         // ElementB,
//         cutlass::int4b_t,
//         LayoutB,
//         0,
//         ThreadMapB,
//         AccessTypeB>;

//     // ThreadMap for scale iterator
//     // scale iterator
//     static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
//     using IteratorScaleThreadMap =
//         transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
//                                                   MmaCore::Shape::kN / kAlignmentScale,
//                                                   kAlignmentScale>;

//     // Define iterators over tiles from the scale operand
//     using IteratorScale =
//         cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
//                                                                 ElementScale,
//                                                                 LayoutScale,
//                                                                 0,
//                                                                 IteratorScaleThreadMap,
//                                                                 kAlignmentScale>;

//     // scale 是从smeme上取出来的
//     using SmemIteratorScale = IteratorScale;

//     // 数据转换 <ds>
//     using Converter = FastInterleavedAndBiasedNumericArrayConverter<int8_t,
//                                                                     cutlass::int4b_t,
//                                                                     MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

//     // Define the threadblock-scoped pipelined matrix multiply
//     using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
//                                                                        IteratorA,
//                                                                        typename MmaCore::SmemIteratorA,
//                                                                        MmaCore::kCacheOpA,
//                                                                        IteratorB,
//                                                                        typename MmaCore::SmemIteratorB,
//                                                                        MmaCore::kCacheOpB,
//                                                                        IteratorScale,
//                                                                        SmemIteratorScale,
//                                                                        ElementAccumulator,
//                                                                        layout::RowMajor,
//                                                                        typename MmaCore::MmaPolicy,
//                                                                        kStages,
//                                                                        Converter,
//                                                                        SharedMemoryClear>;
// };


}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass