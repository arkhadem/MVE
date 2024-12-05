#include "idct.hpp"
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

__device__ void partial_butterfly_inverse_adreno(int row, int col, short *input, short *output, int shift) {

    bool col_is_07 = (col == 0) || (col == 7); // @ CONTROL
    bool col_is_16 = (col == 1) || (col == 6); // @ CONTROL
    bool col_is_25 = (col == 2) || (col == 5); // @ CONTROL
    bool col_is_34 = (col == 3) || (col == 4); // @ CONTROL

    short *my_input = input + row;          // @ ADDRESS
    short *my_output = output + (row << 3); // @ ADDRESS

    short kvz_0, kvz_1, kvz_2, kvz_3, kvz_4, kvz_5, kvz_6, kvz_7;

    if (col_is_07 || col_is_34) {                      // @ CONTROL
        short *kvz_coeff_0_addr = kvz_coeff + 0;       // @ ADDRESS
        kvz_0 = *kvz_coeff_0_addr;                     // @ COMPUTE
        short *kvz_coeff_16_addr = kvz_coeff + 16;     // @ ADDRESS
        kvz_2 = *kvz_coeff_16_addr;                    // @ COMPUTE
        short *kvz_coeff_32_addr = kvz_coeff + 32;     // @ ADDRESS
        kvz_4 = *kvz_coeff_32_addr;                    // @ COMPUTE
        short *kvz_coeff_48_addr = kvz_coeff + 48;     // @ ADDRESS
        kvz_6 = *kvz_coeff_48_addr;                    // @ COMPUTE
        if (col_is_07) {                               // @ CONTROL
            short *kvz_coeff_8_addr = kvz_coeff + 8;   // @ ADDRESS
            kvz_1 = *kvz_coeff_8_addr;                 // @ COMPUTE
            short *kvz_coeff_24_addr = kvz_coeff + 24; // @ ADDRESS
            kvz_3 = *kvz_coeff_24_addr;                // @ COMPUTE
            short *kvz_coeff_40_addr = kvz_coeff + 40; // @ ADDRESS
            kvz_5 = *kvz_coeff_40_addr;                // @ COMPUTE
            short *kvz_coeff_56_addr = kvz_coeff + 56; // @ ADDRESS
            kvz_7 = *kvz_coeff_56_addr;                // @ COMPUTE
        } else {                                       // @ CONTROL
            short *kvz_coeff_11_addr = kvz_coeff + 11; // @ ADDRESS
            kvz_1 = *kvz_coeff_11_addr;                // @ COMPUTE
            short *kvz_coeff_27_addr = kvz_coeff + 27; // @ ADDRESS
            kvz_3 = *kvz_coeff_27_addr;                // @ COMPUTE
            short *kvz_coeff_43_addr = kvz_coeff + 43; // @ ADDRESS
            kvz_5 = *kvz_coeff_43_addr;                // @ COMPUTE
            short *kvz_coeff_59_addr = kvz_coeff + 59; // @ ADDRESS
            kvz_7 = *kvz_coeff_59_addr;                // @ COMPUTE
        }
    } else {                                           // @ CONTROL
        short *kvz_coeff_1_addr = kvz_coeff + 1;       // @ ADDRESS
        kvz_0 = *kvz_coeff_1_addr;                     // @ COMPUTE
        short *kvz_coeff_17_addr = kvz_coeff + 17;     // @ ADDRESS
        kvz_2 = *kvz_coeff_17_addr;                    // @ COMPUTE
        short *kvz_coeff_33_addr = kvz_coeff + 33;     // @ ADDRESS
        kvz_4 = *kvz_coeff_33_addr;                    // @ COMPUTE
        short *kvz_coeff_49_addr = kvz_coeff + 49;     // @ ADDRESS
        kvz_6 = *kvz_coeff_49_addr;                    // @ COMPUTE
        if (col_is_16) {                               // @ CONTROL
            short *kvz_coeff_9_addr = kvz_coeff + 9;   // @ ADDRESS
            kvz_1 = *kvz_coeff_9_addr;                 // @ COMPUTE
            short *kvz_coeff_25_addr = kvz_coeff + 25; // @ ADDRESS
            kvz_3 = *kvz_coeff_25_addr;                // @ COMPUTE
            short *kvz_coeff_41_addr = kvz_coeff + 41; // @ ADDRESS
            kvz_5 = *kvz_coeff_41_addr;                // @ COMPUTE
            short *kvz_coeff_57_addr = kvz_coeff + 57; // @ ADDRESS
            kvz_7 = *kvz_coeff_57_addr;                // @ COMPUTE
        } else {                                       // @ CONTROL
            short *kvz_coeff_10_addr = kvz_coeff + 10; // @ ADDRESS
            kvz_1 = *kvz_coeff_10_addr;                // @ COMPUTE
            short *kvz_coeff_26_addr = kvz_coeff + 26; // @ ADDRESS
            kvz_3 = *kvz_coeff_26_addr;                // @ COMPUTE
            short *kvz_coeff_42_addr = kvz_coeff + 42; // @ ADDRESS
            kvz_5 = *kvz_coeff_42_addr;                // @ COMPUTE
            short *kvz_coeff_58_addr = kvz_coeff + 58; // @ ADDRESS
            kvz_7 = *kvz_coeff_58_addr;                // @ COMPUTE
        }
    }

    short *my_input_8_addr = my_input + 8;                                                                                // @ ADDRESS
    short *my_input_24_addr = my_input + 24;                                                                              // @ ADDRESS
    short *my_input_40_addr = my_input + 40;                                                                              // @ ADDRESS
    short *my_input_56_addr = my_input + 56;                                                                              // @ ADDRESS
    int o = kvz_1 * *my_input_8_addr + kvz_3 * *my_input_24_addr + kvz_5 * *my_input_40_addr + kvz_7 * *my_input_56_addr; // @ COMPUTE

    short *my_input_16_addr = my_input + 16;                        // @ ADDRESS
    short *my_input_48_addr = my_input + 48;                        // @ ADDRESS
    short *my_input_0_addr = my_input + 0;                          // @ ADDRESS
    short *my_input_32_addr = my_input + 32;                        // @ ADDRESS
    int eo = kvz_2 * *my_input_16_addr + kvz_6 * *my_input_48_addr; // @ COMPUTE
    int ee = kvz_0 * *my_input_0_addr + kvz_4 * *my_input_32_addr;  // @ COMPUTE

    int e;

    if (col_is_07 || col_is_16) { // @ CONTROL
        e = eo + ee;              // @ COMPUTE
    } else {                      // @ CONTROL
        e = ee - eo;              // @ COMPUTE
    }

    int result = 1 << (shift - 1);

    if (col < 4) {         // @ CONTROL
        result += (e + o); // @ COMPUTE
    } else {               // @ CONTROL
        result += (e - o); // @ COMPUTE
    }

    result >>= shift; // @ COMPUTE

    int min = -32768; // @ COMPUTE
    int max = 32767;  // @ COMPUTE

    result = (result > max) ? max : result; // @ COMPUTE
    result = (result < min) ? min : result; // @ COMPUTE

    short *my_output_addr = my_output + col; // @ ADDRESS
    *my_output_addr = result;                // @ COMPUTE
}

__global__ void idct_cuda_kernel(int count, int8_t *bitdepth, short *input, short *output) {
    __shared__ short smem[1024];

    int input_per_block = blockDim.x >> 6;                        // @ CONTROL / 64 threads for each input
    int global_block_input_id = blockIdx.x * input_per_block;     // @ CONTROL global input block id
    int local_input_id = threadIdx.x >> 6;                        // @ CONTROL / 64 threads for each input
    int global_input_id = global_block_input_id + local_input_id; // @ CONTROL ID of the input for the current thread
    int local_thread_id = threadIdx.x & 0x3f;                     // @ CONTROL % 64
    int local_thread_row = local_thread_id >> 3;                  // @ CONTROL / 8
    int local_thread_col = local_thread_id & 0x7;                 // @ CONTROL % 8

    int shift_1 = 7;                                                                                            // @ ADDRESS
    if (global_input_id < count) {                                                                              // @ CONTROL
        short *input_addr = input + (global_input_id << 6);                                                     // @ ADDRESS
        short *output_addr = smem + (local_input_id << 6);                                                      // @ ADDRESS
        partial_butterfly_inverse_adreno(local_thread_row, local_thread_col, input_addr, output_addr, shift_1); // @ CONTROL
    }

    __syncthreads(); // @ CONTROL

    int shift_2 = 12 - (bitdepth[0] - 8);                                                                       // @ ADDRESS
    if (global_input_id < count) {                                                                              // @ CONTROL
        short *input_addr = smem + (local_input_id << 6);                                                       // @ ADDRESS
        short *output_addr = output + (global_input_id << 6);                                                   // @ ADDRESS
        partial_butterfly_inverse_adreno(local_thread_row, local_thread_col, input_addr, output_addr, shift_2); // @ CONTROL
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

void idct_cuda(config_t *config,
               input_t *input,
               output_t *output) {

    idct_config_t *idct_config = (idct_config_t *)config;
    idct_input_t *idct_input = (idct_input_t *)input;
    idct_output_t *idct_output = (idct_output_t *)output;

    int count = idct_config->count;
    int8_t *bitdepth = idct_config->bitdepth;
    int16_t *in = idct_input->input;
    int16_t *out = idct_output->output;
    size_t data_size = count * (8 * 8);

    int16_t *d_input, *d_output;
    int8_t *d_bitdepth;
    dim3 blocks((count * 64 + NUM_THREADS - 1) / NUM_THREADS);

    CUDA_SAFE_CALL(cudaMalloc(&d_input, sizeof(int16_t) * data_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_bitdepth, sizeof(int8_t) * count));
    CUDA_SAFE_CALL(cudaMalloc(&d_output, sizeof(int16_t) * data_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_input, input, sizeof(int16_t) * data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bitdepth, bitdepth, sizeof(int8_t) * count, cudaMemcpyHostToDevice));

    idct_cuda_kernel<<<blocks, NUM_THREADS>>>(count, d_bitdepth, d_input, d_output);

    CUDA_SAFE_CALL(cudaMemcpy(output, d_output, sizeof(int16_t) * data_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_bitdepth));
    CUDA_SAFE_CALL(cudaFree(d_output));
}