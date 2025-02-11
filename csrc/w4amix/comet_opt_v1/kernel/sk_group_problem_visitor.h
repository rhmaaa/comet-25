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
    \brief Base scheduler for grouped problems
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template 
    typename ThreadblockShape,
    typename ThreadblockShape1 = ThreadblockShape
>
struct GroupedStreamKProblemVisitor {
    // 问题信息结构
    struct ProblemInfo {
        GemmCoord problem_size;
        int problem_type;
        int64_t total_tiles;
        int64_t start_tile_offset;
    };

    // Stream-K 调度参数
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

        int dp_blocks = 0;
        int dp_first_wave_tiles = 1;
    };

    // 参数结构
    struct Params {
        GemmCoord* problem_sizes;
        int problem_count;
        int* problem_types;
        int avail_sms;
        int sm_occupancy;
    };

    // 成员变量
    Params params;
    StreamKParams sk_params;
    std::vector<ProblemInfo> problems;

    // 当前处理状态
    int current_global_tile_idx;
    int current_problem_idx;
    int current_local_tile_idx;
    int block_iter_begin;
    int block_iter_end;
    int current_sk_block_idx;

    // 构造函数
    GroupedStreamKProblemVisitor(
        Params const& params_,
        int block_idx
    ) : 
        params(params_),
        current_global_tile_idx(block_idx),
        current_problem_idx(-1),
        current_local_tile_idx(0),
        block_iter_begin(0),
        block_iter_end(0),
        current_sk_block_idx(-1)
    {
        // 初始化问题信息和 Stream-K 调度
        initialize_problems();
        compute_block_distribution();
    }

    // 初始化问题信息
    void initialize_problems() {
        int64_t current_tile_offset = 0;

        for (int i = 0; i < params.problem_count; ++i) {
            ProblemInfo problem_info;
            problem_info.problem_size = params.problem_sizes[i];
            problem_info.problem_type = params.problem_types[i];
            
            // 计算每个问题的 tiles
            auto grid = compute_grid_shape(problem_info.problem_size, problem_info.problem_type);
            problem_info.total_tiles = grid.m() * grid.n();
            problem_info.start_tile_offset = current_tile_offset;
            
            sk_params.total_tiles += problem_info.total_tiles;
            current_tile_offset += problem_info.total_tiles;

            problems.push_back(problem_info);
        }
    }

    // 计算网格形状
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

    // 计算块分配（Stream-K 调度）
    void compute_block_distribution() {
        int savings_iters = 0;

        // 使用启发式方法确定 SK 和 DP 块
        get_sk_blocks(
            sk_params.sk_blocks,     
            savings_iters, 
            sk_params.total_tiles,
            1,  // iters_per_tile
            params.avail_sms,
            params.sm_occupancy,
            true
        );

        // 计算 SK 和 DP tiles
        sk_params.sk_tiles = sk_params.sk_blocks > 0 ? 
            (sk_params.total_tiles - sk_params.sk_blocks) : 
            sk_params.total_tiles;
        sk_params.dp_tiles = sk_params.total_tiles - sk_params.sk_tiles;

        // 计算 SK 相关参数
        compute_sk_parameters();
    }

    // 计算 SK 参数的详细实现
    void compute_sk_parameters() {
        sk_params.sk_waves = (sk_params.sk_blocks + params.avail_sms - 1) / params.avail_sms;
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
        sk_params.dp_first_wave_tiles = (sk_params.dp_tiles + params.avail_sms - 1) / params.avail_sms;
    }

    // 解析当前问题和 tile
    bool resolve_current_problem() {
        int accumulated_tiles = 0;
        for (const auto& problem : problems) {
            if (current_global_tile_idx < accumulated_tiles + problem.total_tiles) {
                current_problem_idx = &problem - problems.data();
                current_local_tile_idx = current_global_tile_idx - accumulated_tiles;
                return true;
            }
            accumulated_tiles += problem.total_tiles;
        }
        return false;
    }

    // Stream-K 块索引计算
    int compute_sk_block_idx() {
        int iter = current_local_tile_idx * sk_params.sk_iters_per_region;
        
        int region_idx = iter / sk_params.sk_iters_per_region;
        int iter_in_region = iter % sk_params.sk_iters_per_region;

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

    // 计算迭代范围
    void compute_iter_extents() {
        int sk_block_idx = compute_sk_block_idx();
        
        int region_idx, block_idx_in_region;
        div_mod(sk_block_idx, region_idx, block_idx_in_region, sk_params.sk_blocks_per_region);

        block_iter_begin = (region_idx * sk_params.sk_iters_per_region) + 
                           (block_idx_in_region * sk_params.sk_iters_per_normal_block);
        
        int block_iters = sk_params.sk_iters_per_normal_block;
        if (block_idx_in_region < sk_params.sk_big_blocks_per_region) {
            block_iter_begin += block_idx_in_region;
            block_iters++;
        } else {
            block_iter_begin += sk_params.sk_big_blocks_per_region;
        }
        block_iter_end = block_iter_begin + block_iters;
    }

    // 下一个 tile
    CUTLASS_DEVICE
    bool next_tile() {
        // 检查是否超出全局 tile 范围
        if (current_global_tile_idx >= sk_params.total_tiles) {
            return false;
        }

        // 解析当前问题
        if (!resolve_current_problem()) {
            return false;
        }

        // 计算 Stream-K 相关参数
        current_sk_block_idx = compute_sk_block_idx();
        compute_iter_extents();

        return true;
    }

    // 推进到下一个块
    CUTLASS_DEVICE
    void advance(int grid_size) {
        current_global_tile_idx += grid_size;
    }

    // 获取当前问题大小
    CUTLASS_DEVICE
    GemmCoord problem_size() const {
        return problems[current_problem_idx].problem_size;
    }

    // 获取当前问题索引
    CUTLASS_DEVICE
    int problem_index() const {
        return current_problem_idx;
    }

    // 获取网格形状
    CUTLASS_DEVICE
    GemmCoord grid_shape() const {
        const auto& problem = problems[current_problem_idx];
        return compute_grid_shape(problem.problem_size, problem.problem_type);
    }

    // 获取线程块索引
    CUTLASS_DEVICE
    int threadblock_idx() const {
        return current_local_tile_idx;
    }

private:
    // 辅助除法函数
    CUTLASS_HOST_DEVICE
    static void div_mod(int dividend, int& quotient, int& remainder, int divisor) {
        quotient = dividend / divisor;
        remainder = dividend % divisor;
    }

    // 块选择的静态方法（保留原始启发式算法）
    static void get_sk_blocks(
        int& sk_blocks,     
        int& savings_iters, 
        int sk_tiles,
        int iters_per_tile,
        int avail_sms,
        int max_sk_occupancy,
        bool allow_partial_wave
    ) {
        savings_iters = INT_MIN;
        sk_blocks = 0;

        if (sk_tiles == 0) {
        return;
        }

        int sk_iters = sk_tiles * iters_per_tile;

        int dp_equiv_waves = (sk_tiles + avail_sms - 1) / avail_sms;
        int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

        int min_sk_blocks = (allow_partial_wave) ? fast_min(avail_sms, sk_tiles + 1) : avail_sms;
        int max_sk_blocks = fast_min(avail_sms * max_sk_occupancy, sk_iters / kMinItersPerSkBlock);

        for (int trial_sk_blocks = min_sk_blocks; trial_sk_blocks <= max_sk_blocks; ++trial_sk_blocks)
        {
            int sk_waves = (trial_sk_blocks + avail_sms - 1) / avail_sms;
            int max_sk_iters_per_block = (sk_iters + trial_sk_blocks - 1) / trial_sk_blocks;
            int sk_iter_equiv = max_sk_iters_per_block * sk_waves;

            int num_peers = ((trial_sk_blocks + sk_tiles - 1) / sk_tiles) + 1;        // add one for alignment skew

            float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

            if (trial_sk_blocks % sk_tiles == 0)
            {
                // aligned
                num_peers = (trial_sk_blocks / sk_tiles);

                iter_cost = 0.0f;
            }

            float peer_cost = 2.0f * float(num_peers);

            float base_cost = 2.0f * float(sk_waves);

            int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

            int trial_savings_iters = dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

            if (trial_savings_iters >= savings_iters) {
                savings_iters = trial_savings_iters;
                sk_blocks = trial_sk_blocks;
            }
        }
    }
};

}
}
}