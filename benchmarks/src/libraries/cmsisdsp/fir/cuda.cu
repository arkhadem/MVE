#include "fir.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
__global__ void fir_cuda_kernel(int sample_count, int coeff_count, int *src, int *coeff, int *dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sample_idx = idx;
    int acc = 0;
    int *src_addr = src + sample_idx;
    int *coeff_addr = coeff;
    int src_temp;
    int coeff_temp;
    if (sample_idx < sample_count) {
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            src_temp = *src_addr;
            coeff_temp = *coeff_addr;
            acc += src_temp * coeff_temp;
            src_addr += 1;
            coeff_addr += 1;
        }
        dst[sample_idx] = acc;
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

void fir_cuda(config_t *config,
              input_t *input,
              output_t *output) {

    fir_config_t *fir_config = (fir_config_t *)config;
    fir_input_t *fir_input = (fir_input_t *)input;
    fir_output_t *fir_output = (fir_output_t *)output;

    int sample_count = fir_config->sample_count;
    int coeff_count = fir_config->coeff_count;
    int32_t *src = fir_input->src;
    int32_t *coeff = fir_input->coeff;
    int32_t *dst = fir_output->dst;

    int *d_src, *d_coeff, *d_dst;
    dim3 blocks((sample_count + NUM_THREADS - 1) / NUM_THREADS);

    size_t src_size = (sample_count + coeff_count - 1) * sizeof(int32_t);
    size_t coeff_size = coeff_count * sizeof(int32_t);
    size_t dst_size = sample_count * sizeof(int32_t);

    CUDA_SAFE_CALL(cudaMalloc(&d_src, src_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_coeff, coeff_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_dst, dst_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_coeff, coeff, coeff_size, cudaMemcpyHostToDevice));

    fir_cuda_kernel<<<blocks, NUM_THREADS>>>(sample_count, coeff_count, d_src, d_coeff, d_dst);
    CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_src));
    CUDA_SAFE_CALL(cudaFree(d_coeff));
    CUDA_SAFE_CALL(cudaFree(d_dst));
}