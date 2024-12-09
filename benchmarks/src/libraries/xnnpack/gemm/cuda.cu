#include "gemm.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gemm_cuda_kernel(
    int M,
    int N,
    int K,
    int *input,
    int *bias,
    int *weight,
    int *output,
    int min,
    int max) {
    // output stationary reversed
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int n = idx / M;
    int m = idx - (n * M);

    if (m < M) {
        if (n < N) {
            int *bias_addr = bias + n;
            int accum = *bias_addr;
            int *weight_addr = weight + K * n;
            int *input_addr = input + m;
            for (int k = 0; k < K; ++k) {
                accum += *weight_addr * *input_addr;
                weight_addr++;
                input_addr += M;
            }
            int *output_addr = output + idx;
            *output_addr = accum;
        }
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

void gemm_cuda(config_t *config,
               input_t *input,
               output_t *output) {
    gemm_config_t *gemm_config = (gemm_config_t *)config;
    gemm_input_t *gemm_input = (gemm_input_t *)input;
    gemm_output_t *gemm_output = (gemm_output_t *)output;

    int M = gemm_config->M;
    int N = gemm_config->N;
    int K = gemm_config->K;
    int32_t min = gemm_config->min;
    int32_t max = gemm_config->max;
    int32_t *in = gemm_input->input;
    int32_t *bias = gemm_input->bias;
    int32_t *weights = gemm_input->weights;
    int32_t *out = gemm_output->output;

    const int weight_elements = N * K;
    const int bias_elements = N;
    const int input_elements = K * M;
    const int output_elements = N * M;

    int *d_input, *d_bias, *d_weight, *d_output;
    dim3 blocks((M * N + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_input, input_elements * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_bias, bias_elements * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_weight, weight_elements * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_output, output_elements * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(d_input, in, input_elements * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bias, bias, bias_elements * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_weight, weights, weight_elements * sizeof(int), cudaMemcpyHostToDevice));

    gemm_cuda_kernel<<<blocks, NUM_THREADS>>>(M, N, K, d_input, d_bias, d_weight, d_output, min, max);
    CUDA_SAFE_CALL(cudaMemcpy(out, d_output, output_elements * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_bias));
    CUDA_SAFE_CALL(cudaFree(d_weight));
    CUDA_SAFE_CALL(cudaFree(d_output));
}