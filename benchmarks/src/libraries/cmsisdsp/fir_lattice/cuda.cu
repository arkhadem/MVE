#include "fir_lattice.hpp"
#include "stdio.h"
#include <cassert>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
__global__ void fir_lattice_gpu(int sample_count, int coeff_count, int *src, int *coeff, int *dst) {
    // @DAICHI implement GPU code here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sample_idx = idx;
    extern __shared__ int G[];
    G[idx] = sample_idx == 0 ? 0 : src[sample_idx - 1];
    __syncthreads();
    int f_val = src[sample_idx];
    int G_prev_temp = G[idx];
    int G_curr_temp;
    int coeff_temp;

    if (sample_idx < sample_count) {
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            coeff_temp = coeff[coeff_idx];
            G_curr_temp = coeff_temp * f_val + G_prev_temp;
            f_val += coeff_temp * G_prev_temp;

            // send curr_temp to tid+1
            G[idx + 1] = G_curr_temp;
            __syncthreads();
            // read curr_temp and store it for the next loop
            G_prev_temp = G[idx];
            __syncthreads();
        }
        dst[sample_idx] = f_val;
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

void fir_lattice_cuda(config_t *config,
                      input_t *input,
                      output_t *output) {

    fir_lattice_config_t *fir_lattice_config = (fir_lattice_config_t *)config;
    fir_lattice_input_t *fir_lattice_input = (fir_lattice_input_t *)input;
    fir_lattice_output_t *fir_lattice_output = (fir_lattice_output_t *)output;

    int sample_count = fir_lattice_config->sample_count;
    int coeff_count = fir_lattice_config->coeff_count;
    int32_t *src = fir_lattice_input->src;
    int32_t *coeff = fir_lattice_input->coeff;
    int32_t *dst = fir_lattice_output->dst;
    size_t src_size = (sample_count + coeff_count - 1) * sizeof(int32_t);
    size_t coeff_size = coeff_count * sizeof(int32_t);
    size_t dst_size = sample_count * sizeof(int32_t);

    int32_t *d_src, *d_coeff, *d_dst;
    dim3 blocks((sample_count + NUM_THREADS - 1) / NUM_THREADS);
    assert(blocks.x == 1 && "Inter-block communication is not supported.");

    CUDA_SAFE_CALL(cudaMalloc(&d_src, src_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_coeff, coeff_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_dst, dst_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_coeff, coeff, coeff_size, cudaMemcpyHostToDevice));

    fir_lattice_gpu<<<blocks, NUM_THREADS, sizeof(int32_t) * (NUM_THREADS + 1)>>>(sample_count, coeff_count, d_src, d_coeff, d_dst);
    CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_src));
    CUDA_SAFE_CALL(cudaFree(d_coeff));
    CUDA_SAFE_CALL(cudaFree(d_dst));
}