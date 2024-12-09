/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#ifndef TILE_SIZE_K
#define TILE_SIZE_K 16
#endif

#ifndef TILE_SIZE_M
#define TILE_SIZE_M 16
#endif

__kernel void x32_spmm(
    const int M,
    const int N,
    __global const int *const restrict input,
    __global const int *const restrict bias,
    __global const int *const restrict weight,
    __global const int *const restrict IDX,
    __global const unsigned int *const restrict NNZ,
    __global int *restrict output,
    int min,
    int max) {

    __local int sdata[(TILE_SIZE_M + 1) * (TILE_SIZE_K + 1)];

    int idx_m = get_global_id(0);
    int local_idx_k = get_global_id(1);
    int idx_n = get_global_id(2);
    int global_idx_k = get_local_id(0) * (TILE_SIZE_K + 1) + local_idx_k;

    __global const int *input_addr = input + idx_m;
    __global int *output_addr = output + idx_m;

    if (idx_m < M) {
        const int row_start = NNZ[idx_n];
        const int row_end = NNZ[idx_n + 1];
        int sum = 0.00;

        for (int j = row_start + local_idx_k; j < row_end; j += TILE_SIZE_K) {
            sum += weight[j] * input_addr[IDX[j] * M];
        }

        //parllel reduction in shared memory
        sdata[global_idx_k] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
#if TILE_SIZE_K > 32
        sdata[global_idx_k] = sum += sdata[global_idx_k + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if TILE_SIZE_K > 16
        sdata[global_idx_k] = sum += sdata[global_idx_k + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if TILE_SIZE_K > 8
        sdata[global_idx_k] = sum += sdata[global_idx_k + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if TILE_SIZE_K > 4
        sdata[global_idx_k] = sum += sdata[global_idx_k + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if TILE_SIZE_K > 2
        sdata[global_idx_k] = sum += sdata[global_idx_k + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if TILE_SIZE_K > 1
        sum += sdata[global_idx_k + 1];
#endif

        if (local_idx_k == 0) {
            sum += bias[idx_n];
            if (sum > max)
                sum = max;
            if (sum < min)
                sum = min;
            output_addr[idx_n * M] = sum;
        }
    }
}