#include "satd.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void satd_cuda_kernel(int count, uint8_t *piOrg, uint8_t *piCur, int *result) {
    __shared__ int smem[1024];

    int global_tid = threadIdx.x + blockDim.x * blockIdx.x; //      @ CONTROL
    int global_input_id = global_tid >> 6;                  // / 64 @ CONTROL
    int local_thread_id = global_tid & 63;                  // % 64 @ ADDRESS
    int row = local_thread_id >> 3;                         // / 8  @ ADDRESS
    int col = local_thread_id & 7;                          // % 8  @ ADDRESS

    int inout_offset = global_input_id << 6;         // @ ADDRESS
    piOrg += inout_offset;                           // @ ADDRESS
    piCur += inout_offset;                           // @ ADDRESS
    result += inout_offset;                          // @ ADDRESS
    int *my_smem = smem + ((threadIdx.x >> 6) << 6); // @ ADDRESS

    int input_offset = row << 3;                   // @ ADDRESS
    uint8_t *my_piOrg_addr = piOrg + input_offset; // @ ADDRESS
    uint8_t *my_piCur_addr = piCur + input_offset; // @ ADDRESS
    int *my_result = my_smem + input_offset + col; // @ ADDRESS
    if (global_input_id < count) {                 // @ CONTROL

        // Column 0 (+, +, +, +, +, +, +, +)
        int acc = *my_piOrg_addr - *my_piCur_addr; // @ COMPUTE
        my_piOrg_addr += 1;                        // @ ADDRESS
        my_piCur_addr += 1;                        // @ ADDRESS

        // Column 1 (+, -, +, -, +, -, +, -)
        int diff = *my_piOrg_addr - *my_piCur_addr;                 // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 2) || (col == 4) || (col == 6)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 2 (+, +, -, -, +, +, -, -)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 1) || (col == 4) || (col == 5)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 3 (+, -, -, +, +, -, -, +)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 3) || (col == 4) || (col == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 4 (+, +, +, +, -, -, -, -)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 1) || (col == 2) || (col == 3)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 5 (+, -, +, -, -, +, -, +)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 2) || (col == 5) || (col == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 6 (+, +, -, -, -, -, +, +)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        my_piOrg_addr += 1;                                         // @ ADDRESS
        my_piCur_addr += 1;                                         // @ ADDRESS
        if ((col == 0) || (col == 1) || (col == 6) || (col == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 7 (+, -, -, +, -, +, +, -)
        diff = *my_piOrg_addr - *my_piCur_addr;                     // @ COMPUTE
        if ((col == 0) || (col == 3) || (col == 5) || (col == 6)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        *my_result = acc; // @ COMPUTE
    }

    __syncthreads(); // @ CONTROL

    int *my_input = my_smem + col;           // @ ADDRESS
    my_result = result + input_offset + col; // @ ADDRESS
    if (global_input_id < count) {           // @ CONTROL

        // Column 0 (+, +, +, +, +, +, +, +)
        int acc = *my_input; // @ COMPUTE
        my_input += 8;       // @ ADDRESS

        // Column 1 (+, -, +, -, +, -, +, -)
        int diff = *my_input;                                       // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 2) || (row == 4) || (row == 6)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 2 (+, +, -, -, +, +, -, -)
        diff = *my_input;                                           // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 1) || (row == 4) || (row == 5)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 3 (+, -, -, +, +, -, -, +)
        diff = *my_input;                                           // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 3) || (row == 4) || (row == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 4 (+, +, +, +, -, -, -, -)
        diff = *my_input;                                           // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 1) || (row == 2) || (row == 3)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 5 (+, -, +, -, -, +, -, +)
        diff = *my_input;                                           // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 2) || (row == 5) || (row == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 6 (+, +, -, -, -, -, +, +)
        diff = *my_input;                                           // @ COMPUTE
        my_input += 8;                                              // @ ADDRESS
        if ((row == 0) || (row == 1) || (row == 6) || (row == 7)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        // Column 7 (+, -, -, +, -, +, +, -)
        diff = *my_input;                                           // @ COMPUTE
        if ((row == 0) || (row == 3) || (row == 5) || (row == 6)) { // @ CONTROL
            acc += diff;                                            // @ COMPUTE
        } else {                                                    // @ CONTROL
            acc -= diff;                                            // @ COMPUTE
        }

        *my_result = acc; // @ COMPUTE
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

void satd_cuda(config_t *config,
               input_t *input,
               output_t *output) {
    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t *satd_input = (satd_input_t *)input;
    satd_output_t *satd_output = (satd_output_t *)output;

    int count = satd_config->count;
    uint8_t *piOrg = satd_input->piOrg;
    uint8_t *piCur = satd_input->piCur;
    int32_t *result = satd_output->result;

    uint8_t *d_piOrg, *d_piCur;
    int32_t *d_result;
    dim3 blocks((count * 64 + NUM_THREADS - 1) / NUM_THREADS);

    size_t input_size = count * 64 * sizeof(uint8_t);
    size_t output_size = count * 64 * sizeof(int32_t);

    CUDA_SAFE_CALL(cudaMalloc(&d_piOrg, input_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_piCur, input_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_result, output_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_piOrg, piOrg, input_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_piCur, piCur, input_size, cudaMemcpyHostToDevice));

    satd_cuda_kernel<<<blocks, NUM_THREADS>>>(count, d_piOrg, d_piCur, d_result);

    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, output_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_piOrg));
    CUDA_SAFE_CALL(cudaFree(d_piCur));
    CUDA_SAFE_CALL(cudaFree(d_result));
}