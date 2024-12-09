#include "spmm.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void spmm_cuda_kernel(
    int M,
    int N,
    int *input,
    int *bias,
    int *weights,
    int32_t *IDX,
    uint32_t *NNZ,
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
            int *weight_addr = weights + NNZ[n];
            for (int k = NNZ[n]; k < NNZ[n + 1]; ++k) {
                int *input_addr = input + M * IDX[k] + m;
                accum += *weight_addr * *input_addr;
                weight_addr++;
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

void spmm_cuda(config_t *config,
               input_t *input,
               output_t *output) {
    spmm_config_t *spmm_config = (spmm_config_t *)config;
    spmm_input_t *spmm_input = (spmm_input_t *)input;
    spmm_output_t *spmm_output = (spmm_output_t *)output;

    int M = spmm_config->M;
    int N = spmm_config->N;
    int K = spmm_config->K;
    int32_t min = spmm_config->min;
    int32_t max = spmm_config->max;
    int32_t *in = spmm_input->input;
    int32_t *bias = spmm_input->bias;
    int32_t *weights = spmm_input->weights;
    int32_t *IDX = spmm_input->IDX;
    uint32_t *NNZ = spmm_input->NNZ;
    int32_t *out = spmm_output->output;

    const int weight_elements = (float)(spmm_config->N * spmm_config->K) * (1.0 - spmm_config->sparsity) + 1;
    const int bias_elements = spmm_config->N;
    const int input_elements = spmm_config->K * spmm_config->M;
    const int output_elements = spmm_config->N * spmm_config->M;

    int32_t *d_input, *d_bias, *d_weight, *d_output;
    uint32_t *d_nmap;
    int32_t *d_dmap;
    dim3 blocks((M * N + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_input, input_elements * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_bias, bias_elements * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_weight, weight_elements * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_nmap, (N + 1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dmap, weight_elements * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_output, output_elements * sizeof(int32_t)));

    CUDA_SAFE_CALL(cudaMemcpy(d_input, in, input_elements * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bias, bias, bias_elements * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_weight, weights, weight_elements * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_nmap, NNZ, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_dmap, IDX, weight_elements * sizeof(int32_t), cudaMemcpyHostToDevice));

    spmm_cuda_kernel<<<blocks, NUM_THREADS>>>(M, N, d_input, d_bias, d_weight, d_dmap, d_nmap, d_output, min, max);
    CUDA_SAFE_CALL(cudaMemcpy(out, d_output, output_elements * sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_bias));
    CUDA_SAFE_CALL(cudaFree(d_weight));
    CUDA_SAFE_CALL(cudaFree(d_nmap));
    CUDA_SAFE_CALL(cudaFree(d_dmap));
    CUDA_SAFE_CALL(cudaFree(d_output));
}