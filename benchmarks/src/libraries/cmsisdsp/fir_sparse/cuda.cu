#include "fir_sparse.hpp"
#include "stdio.h"
#include <cassert>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
__global__ void fir_sparse_gpu(int sample_count, int coeff_count, int *src, int *coeff, int *delay, int *dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sample_idx = idx;
    int acc = 0;
    int *src_addr;
    int *coeff_addr = coeff;
    int *delay_addr = delay;
    int src_temp;
    int coeff_temp;
    int delay_temp;
    src += sample_idx;

    for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
        delay_temp = *delay_addr;
        src_addr = src + delay_temp;
        src_temp = *src_addr;
        coeff_temp = *coeff_addr;
        acc += src_temp * coeff_temp;
        coeff_addr += 1;
        delay_addr += 1;
    }
    dst[sample_idx] = acc;
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

void fir_sparse_cuda(config_t *config,
                     input_t *input,
                     output_t *output) {

    fir_sparse_config_t *fir_sparse_config = (fir_sparse_config_t *)config;
    fir_sparse_input_t *fir_sparse_input = (fir_sparse_input_t *)input;
    fir_sparse_output_t *fir_sparse_output = (fir_sparse_output_t *)output;

    int input_count = fir_sparse_config->input_count;
    int sample_count = fir_sparse_config->sample_count;
    int coeff_count = fir_sparse_config->coeff_count;
    int effective_coeff_count = fir_sparse_config->effective_coeff_count;
    int32_t *src = fir_sparse_input->src;
    int32_t *coeff = fir_sparse_input->coeff;
    int32_t *dst = fir_sparse_output->dst;
    int32_t *delay = fir_sparse_input->delay;
    size_t src_size = input_count * sizeof(int32_t);
    size_t coeff_size = coeff_count * sizeof(int32_t);
    size_t delay_size = effective_coeff_count * sizeof(int32_t);
    size_t dst_size = sample_count * sizeof(int32_t);

    int *d_src, *d_coeff, *d_dst;
    int *d_delay;
    dim3 blocks((sample_count + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_src, src_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_coeff, coeff_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_delay, delay_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_dst, dst_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_coeff, coeff, coeff_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_delay, delay, delay_size, cudaMemcpyHostToDevice));

    fir_sparse_gpu<<<blocks, NUM_THREADS, sizeof(int) * (NUM_THREADS + 1)>>>(sample_count, effective_coeff_count, d_src, d_coeff, d_delay, d_dst);
    CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_src));
    CUDA_SAFE_CALL(cudaFree(d_coeff));
    CUDA_SAFE_CALL(cudaFree(d_dst));
}