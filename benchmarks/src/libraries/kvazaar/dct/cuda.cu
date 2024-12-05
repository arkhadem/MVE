#include "dct.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ short kvz_coeff[64] = {
    64, 64, 64, 64, 64, 64, 64, 64,
    89, 75, 50, 18, -18, -50, -75, -89,
    83, 36, -36, -83, -83, -36, 36, 83,
    75, -18, -89, -50, 50, 89, 18, -75,
    64, -64, -64, 64, 64, -64, -64, 64,
    50, -89, 18, 75, -75, -18, 89, -50,
    36, -83, 83, -36, -36, 83, -83, 36,
    18, -50, 75, -89, 89, -75, 50, -18};

__constant__ short convert_coeff[64 + 1] = {
    -1, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 2,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4};

__device__ void partial_butterfly_8_adreno(int row, int col, short *input, short *output, int shift) {

    bool row_is_1357 = (row & 0x1);            // @ CONTROL
    bool row_is_04 = (row == 0) || (row == 4); // @ CONTROL

    short *my_input = input + (col << 3);     // @ ADDRESS
    short *my_output = output + (row << 3);   // @ ADDRESS
    short *my_coeff = kvz_coeff + (row << 3); // @ ADDRESS

    int result = 1 << (shift - 1); // @ COMPUTE

    int ee_eo_o_0 = 0; // @ COMPUTE
    int ee_eo_o_1 = 0; // @ COMPUTE

    if (row_is_1357) {                                 // @ CONTROL
        short *my_input_2_addr = my_input + 2;         // @ ADDRESS
        short *my_input_5_addr = my_input + 5;         // @ ADDRESS
        int o_2 = *my_input_2_addr - *my_input_5_addr; // @ COMPUTE

        result += o_2 * my_coeff[2];

        short *my_input_3_addr = my_input + 3;         // @ ADDRESS
        short *my_input_4_addr = my_input + 4;         // @ ADDRESS
        int o_3 = *my_input_3_addr - *my_input_4_addr; // @ COMPUTE

        result += o_3 * my_coeff[3];

        short *my_input_0_addr = my_input + 0;           // @ ADDRESS
        short *my_input_7_addr = my_input + 7;           // @ ADDRESS
        ee_eo_o_0 = *my_input_0_addr - *my_input_7_addr; // @ COMPUTE

        short *my_input_1_addr = my_input + 1;           // @ ADDRESS
        short *my_input_6_addr = my_input + 6;           // @ ADDRESS
        ee_eo_o_1 = *my_input_1_addr - *my_input_6_addr; // @ COMPUTE
    } else {
        short *my_input_0_addr = my_input + 0;           // @ ADDRESS
        short *my_input_7_addr = my_input + 7;           // @ ADDRESS
        ee_eo_o_0 = *my_input_0_addr + *my_input_7_addr; // @ COMPUTE

        short *my_input_1_addr = my_input + 1;           // @ ADDRESS
        short *my_input_6_addr = my_input + 6;           // @ ADDRESS
        ee_eo_o_1 = *my_input_1_addr + *my_input_6_addr; // @ COMPUTE

        short *my_input_2_addr = my_input + 2;         // @ ADDRESS
        short *my_input_5_addr = my_input + 5;         // @ ADDRESS
        int e_2 = *my_input_2_addr + *my_input_5_addr; // @ COMPUTE

        short *my_input_3_addr = my_input + 3;         // @ ADDRESS
        short *my_input_4_addr = my_input + 4;         // @ ADDRESS
        int e_3 = *my_input_3_addr + *my_input_4_addr; // @ COMPUTE

        if (row_is_04) {      // @ CONTROL
            ee_eo_o_1 += e_2; // @ COMPUTE
            ee_eo_o_0 += e_3; // @ COMPUTE
        } else {              // @ CONTROL
            ee_eo_o_1 -= e_2; // @ COMPUTE
            ee_eo_o_0 -= e_3; // @ COMPUTE
        }
    }

    short *my_coeff_0_address = my_coeff + 0;  // @ ADDRESS
    result += ee_eo_o_0 * *my_coeff_0_address; // @ COMPUTE

    short *my_coeff_1_address = my_coeff + 1;  // @ ADDRESS
    result += ee_eo_o_1 * *my_coeff_1_address; // @ COMPUTE

    short *my_output_addr = my_output + col; // @ ADDRESS
    *my_output_addr = result >> shift;       // @ COMPUTE
}

__global__ void dct_cuda_kernel(int count, int8_t *bitdepth, short *input, short *output) {
    __shared__ short smem[1024];

    int input_per_block = blockDim.x >> 6;                        // @ CONTROL / 64 threads for each input
    int global_block_input_id = blockIdx.x * input_per_block;     // @ CONTROL global input block id
    int local_input_id = threadIdx.x >> 6;                        // @ CONTROL / 64 threads for each input
    int global_input_id = global_block_input_id + local_input_id; // @ CONTROL ID of the input for the current thread
    int local_thread_id = threadIdx.x & 0x3f;                     // @ CONTROL % 64
    int local_thread_row = local_thread_id >> 3;                  // @ CONTROL / 8
    int local_thread_col = local_thread_id & 0x7;                 // @ CONTROL % 8

    int shift_1 = convert_coeff[8] + 1 + (bitdepth[0] - 8);                                               // @ ADDRESS
    if (global_input_id < count) {                                                                        // @ CONTROL
        short *input_addr = input + (global_input_id << 6);                                               // @ ADDRESS
        short *output_addr = smem + (local_input_id << 6);                                                // @ ADDRESS
        partial_butterfly_8_adreno(local_thread_row, local_thread_col, input_addr, output_addr, shift_1); // @ CONTROL
    }

    __syncthreads(); // @ CONTROL

    int shift_2 = convert_coeff[8] + 8;                                                                   // @ ADDRESS
    if (global_input_id < count) {                                                                        // @ CONTROL
        short *input_addr = smem + (local_input_id << 6);                                                 // @ ADDRESS
        short *output_addr = output + (global_input_id << 6);                                             // @ ADDRESS
        partial_butterfly_8_adreno(local_thread_row, local_thread_col, input_addr, output_addr, shift_2); // @ CONTROL
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

void dct_cuda(config_t *config,
              input_t *input,
              output_t *output) {

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t *dct_input = (dct_input_t *)input;
    dct_output_t *dct_output = (dct_output_t *)output;

    int count = dct_config->count;
    int8_t *bitdepth = dct_config->bitdepth;
    int16_t *in = dct_input->input;
    int16_t *out = dct_output->output;
    size_t data_size = count * (8 * 8);

    int16_t *d_input, *d_output;
    int8_t *d_bitdepth;
    dim3 blocks((count * 64 + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_input, sizeof(int16_t) * data_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_bitdepth, sizeof(int8_t) * count));
    CUDA_SAFE_CALL(cudaMalloc(&d_output, sizeof(int16_t) * data_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_input, input, sizeof(int16_t) * data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bitdepth, bitdepth, sizeof(int8_t) * count, cudaMemcpyHostToDevice));

    dct_cuda_kernel<<<blocks, NUM_THREADS>>>(count, d_bitdepth, d_input, d_output);

    CUDA_SAFE_CALL(cudaMemcpy(output, d_output, sizeof(int16_t) * data_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_bitdepth));
    CUDA_SAFE_CALL(cudaFree(d_output));
}