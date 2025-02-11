#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/gemm/threadblock/index_remat.h"

#if !defined(__CUDACC_RTC__)
#include <iostream>
#include "cutlass/core_io.h"
#include "cutlass/trace.h"
#endif


namespace cutlass {
namespace gemm {
namespace threadblock {

template 
    typename ThreadblockShape,
    typename ThreadblockShape1 = ThreadblockShape
>
struct MultiProblemThreadblockSwizzleStreamK {
    // 问题信息结构
    struct ProblemInfo {
        GemmCoord problem_size;
        int problem_type;
        int64_t total_tiles;
        int64_t start_tile_offset;  // 该问题在全局 tile 中的起始偏移
    };

    // 调度参数
    struct ScheduleParams {
        int problem_count;
        ProblemInfo* problems;
        int avail_sms;
        int device_sms;
        int sm_occupancy;
    };

    // 调度参数
    ScheduleParams schedule_params;

    // Stream-K 调度关键参数
    struct StreamKParams {
        int total_tiles = 0;
        int sk_tiles = 0;
        int dp_tiles = 0;
        
        int sk_waves = 0;
        int sk_blocks = 0;
        int sk_blocks_per_region = 0;
        int sk_regions = 1;
        
        int sk_iters_per_region = 0;
        int sk_iters_per_normal_block = 0;
        int sk_big_blocks_per_region = 0;

        // DP 相关参数
        int dp_blocks = 0;
        int dp_first_wave_tiles = 1;
    } sk_params;

    // 构造函数
    MultiProblemThreadblockSwizzleStreamK(
        GemmCoord* problem_sizes,
        int problem_count,
        int* problem_types,
        int sm_occupancy,
        int device_sms,
        int avail_sms,
        int epilogue_fragments
    ) {
        // 准备问题信息
        std::vector<ProblemInfo> problems(problem_count);
        int64_t current_tile_offset = 0;

        for (int i = 0; i < problem_count; ++i) {
            problems[i].problem_size = problem_sizes[i];
            problems[i].problem_type = problem_types[i];
            
            // 计算每个问题的 tiles
            auto grid = compute_grid_shape(problem_sizes[i], problem_types[i]);
            problems[i].total_tiles = grid.m() * grid.n();
            problems[i].start_tile_offset = current_tile_offset;
            
            sk_params.total_tiles += problems[i].total_tiles;
            current_tile_offset += problems[i].total_tiles;
        }

        schedule_params = {
            problem_count,
            problems.data(),
            avail_sms,
            device_sms,
            sm_occupancy
        };

        // 动态分配 SK 和 DP 块
        compute_block_distribution(epilogue_fragments);
    }

    // 计算网格形状（根据问题类型）
    GemmCoord compute_grid_shape(const GemmCoord& problem, int problem_type) {
        if (problem_type == 0) {
            return GemmCoord(
                (problem.m() - 1 + ThreadblockShape1::kM) / ThreadblockShape1::kM,
                (problem.n() - 1 + ThreadblockShape1::kN) / ThreadblockShape1::kN,
                1
            );
        } else {
            return GemmCoord(
                (problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM,
                (problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN,
                1
            );
        }
    }

    // 计算块分配
    void compute_block_distribution(int epilogue_fragments) {
        int savings_iters = 0;

        // 使用启发式方法确定 SK 和 DP 块
        get_sk_blocks(
            sk_params.sk_blocks,     
            savings_iters, 
            sk_params.total_tiles,
            1,  // iters_per_tile
            schedule_params.avail_sms,
            schedule_params.sm_occupancy,
            true
        );

        // 计算 SK 和 DP tiles
        sk_params.sk_tiles = sk_params.sk_blocks > 0 ? 
            (sk_params.total_tiles - sk_params.sk_blocks) : 
            sk_params.total_tiles;
        sk_params.dp_tiles = sk_params.total_tiles - sk_params.sk_tiles;

        // 计算 SK 相关参数
        sk_params.sk_waves = (sk_params.sk_blocks + schedule_params.avail_sms - 1) 
                              / schedule_params.avail_sms;
        sk_params.sk_regions = 1;  // 默认单一区域
        sk_params.sk_blocks_per_region = sk_params.sk_blocks / sk_params.sk_regions;

        // 计算迭代相关参数
        int sk_iters = sk_params.sk_tiles;
        sk_params.sk_iters_per_normal_block = sk_iters / sk_params.sk_blocks;
        int extra_sk_iters = sk_iters - (sk_params.sk_iters_per_normal_block * sk_params.sk_blocks);
        sk_params.sk_big_blocks_per_region = extra_sk_iters;
        sk_params.sk_iters_per_region = sk_iters;

        // 计算 DP 块
        sk_params.dp_blocks = sk_params.dp_tiles;
        sk_params.dp_first_wave_tiles = (sk_params.dp_tiles + schedule_params.avail_sms - 1) 
                                         / schedule_params.avail_sms;
    }

    // 获取特定迭代的块索引
    CUTLASS_HOST_DEVICE
    int get_sk_block_idx(int iter) const {
        int region_idx = iter / sk_params.sk_iters_per_region;
        int iter_in_region = iter % sk_params.sk_iters_per_region;

        // 计算块索引
        int big_block_iters = sk_params.sk_big_blocks_per_region * 
                               (sk_params.sk_iters_per_normal_block + 1);
        int normal_block_iters = iter_in_region - big_block_iters;

        int big_block_idx = iter_in_region / (sk_params.sk_iters_per_normal_block + 1);
        int normal_block_idx = (iter_in_region - big_block_iters) / sk_params.sk_iters_per_normal_block;

        int block_idx_in_region = (big_block_idx < sk_params.sk_big_blocks_per_region) ? 
            big_block_idx : 
            (sk_params.sk_big_blocks_per_region + normal_block_idx);

        return region_idx * sk_params.sk_blocks_per_region + block_idx_in_region;
    }

    // 获取迭代范围
    CUTLASS_HOST_DEVICE
    void get_iter_extents(
        int sk_block_idx, 
        int& block_iter_begin, 
        int& block_iter_end
    ) const {
        int region_idx, block_idx_in_region;
        div_mod(sk_block_idx, region_idx, block_idx_in_region, sk_params.sk_blocks_per_region);

        block_iter_begin = region_idx * sk_params.sk_iters_per_region + 
                           block_idx_in_region * sk_params.sk_iters_per_normal_block;
        
        // 调整大块和普通块的迭代范围
        if (block_idx_in_region < sk_params.sk_big_blocks_per_region) {
            block_iter_begin += block_idx_in_region;
            block_iter_end = block_iter_begin + sk_params.sk_iters_per_normal_block + 1;
        } else {
            block_iter_begin += sk_params.sk_big_blocks_per_region;
            block_iter_end = block_iter_begin + sk_params.sk_iters_per_normal_block;
        }
    }

    // 获取块索引
    CUTLASS_HOST_DEVICE
    int get_block_idx() const {
        // 根据 Stream-K 调度策略获取块索引的复杂逻辑
        int sk_padding_start_block_idx = sk_params.sk_regions * sk_params.sk_blocks_per_region;
        int dp_start_block_idx = sk_params.sk_waves * schedule_params.avail_sms;
        int reduce_start_block_idx = dp_start_block_idx + sk_params.dp_blocks;

        // 根据块索引返回对应的块
        return 0;  // 简化实现，实际应根据具体调度策略计算
    }

    // 获取特定问题的网格形状
    CUTLASS_HOST_DEVICE
    GemmCoord get_problem_grid_shape(int problem_idx) const {
        if (problem_idx < 0 || problem_idx >= schedule_params.problem_count)
            return GemmCoord(0, 0, 0);
        
        const auto& problem = schedule_params.problems[problem_idx];
        return compute_grid_shape(problem.problem_size, problem.problem_type);
    }

    // 获取特定问题的 tile 偏移
    CUTLASS_HOST_DEVICE
    int64_t get_problem_tile_offset(int problem_idx) const {
        if (problem_idx < 0 || problem_idx >= schedule_params.problem_count)
            return 0;
        
        return schedule_params.problems[problem_idx].start_tile_offset;
    }

private:
    // 块选择的静态方法
    static void get_sk_blocks(
        int& sk_blocks,     
        int& savings_iters, 
        int sk_tiles,
        int iters_per_tile,
        int avail_sms,
        int max_sk_occupancy,
        bool allow_partial_wave
    ) {
        // 参考 CUTLASS 原有实现的启发式块选择算法
        // 这里需要实现复杂的块选择逻辑
        sk_blocks = 0;
        savings_iters = 0;
    }

    // 辅助除法函数
    CUTLASS_HOST_DEVICE
    static void div_mod(int dividend, int& quotient, int& remainder, int divisor) {
        quotient = dividend / divisor;
        remainder = dividend % divisor;
    }
};