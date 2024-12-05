#include "lpack.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void lpack_cuda_kernel(int n, int32_t *da, int32_t *dx, int32_t *dyin, int32_t *dyout) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        dyout[idx] = dyin[idx] + da[0] * dx[idx];
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

void lpack_cuda(config_t *config,
                input_t *input,
                output_t *output) {
    lpack_config_t *lpack_config = (lpack_config_t *)config;
    lpack_input_t *lpack_input = (lpack_input_t *)input;
    lpack_output_t *lpack_output = (lpack_output_t *)output;

    int n = lpack_config->n;
    int32_t *da = lpack_input->da;
    int32_t *dx = lpack_input->dx;
    int32_t *dyin = lpack_input->dyin;
    int32_t *dyout = lpack_output->dyout;

    int *d_da, *d_dx, *d_dyin, *d_dyout;
    dim3 blocks((n + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_da, 1 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dx, n * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dyin, n * sizeof(int32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dyout, n * sizeof(int32_t)));

    CUDA_SAFE_CALL(cudaMemcpy(d_da, da, 1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_dx, dx, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_dyin, dyin, n * sizeof(int32_t), cudaMemcpyHostToDevice));

    lpack_cuda_kernel<<<blocks, NUM_THREADS>>>(n, d_da, d_dx, d_dyin, d_dyout);
    CUDA_SAFE_CALL(cudaMemcpy(dyout, d_dyout, n * sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_da));
    CUDA_SAFE_CALL(cudaFree(d_dx));
    CUDA_SAFE_CALL(cudaFree(d_dyin));
    CUDA_SAFE_CALL(cudaFree(d_dyout));
}