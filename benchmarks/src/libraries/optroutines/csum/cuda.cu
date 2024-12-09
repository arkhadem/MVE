#include "csum.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void csum_cuda_kernel(int32_t *ptr, int count, int16_t *sum) {
    __shared__ int64_t smem[1024];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsz = blockDim.x;

    ptr += bid * BLOCK_16K;
    sum += bid;
    int32_t *my_ptr = ptr + tid;
    int64_t *my_smem = smem + tid;

    int64_t sum_tmp = 0;
    int curr_sample_id = tid;
    while (curr_sample_id < BLOCK_16K) {
        sum_tmp += *my_ptr;
        my_ptr += bsz;
        curr_sample_id += bsz;
    }
    *my_smem = sum_tmp;
    __syncthreads();

    int stride = bsz >> 1;
    while (stride > 0) {
        if (tid < stride) {
            int64_t *next_smem = my_smem + stride;
            sum_tmp += *next_smem;
            *my_smem = sum_tmp;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (tid == 0) {
        /* Fold 64-bit sum_tmp to 32 bits */
        sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
        sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);

        /* Fold 32-bit sum_tmp to 16 bits */
        sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
        sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);

        *sum = sum_tmp;
    }
}

#define CUDA_SAFE_CALL(call)                                              \
    do {                                                                  \
        cudaError err = call;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NUM_THREADS 1024

void csum_cuda(config_t *config,
               input_t *input,
               output_t *output) {
    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t *csum_input = (csum_input_t *)input;
    csum_output_t *csum_output = (csum_output_t *)output;

    int count = csum_config->count;
    int32_t *ptr = csum_input->ptr;
    int16_t *sum = csum_output->sum;

    int *d_ptr;
    int16_t *d_sum;
    dim3 blocks(count);

    CUDA_SAFE_CALL(cudaMalloc(&d_ptr, count * BLOCK_16K * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_sum, count * sizeof(int16_t)));

    CUDA_SAFE_CALL(cudaMemcpy(d_ptr, ptr, count * BLOCK_16K * sizeof(int32_t), cudaMemcpyHostToDevice));

    csum_cuda_kernel<<<blocks, NUM_THREADS>>>(d_ptr, count, d_sum);

    CUDA_SAFE_CALL(cudaMemcpy(sum, d_sum, count * sizeof(int16_t), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_ptr));
    CUDA_SAFE_CALL(cudaFree(d_sum));
}