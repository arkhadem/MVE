#include "intra.hpp"
#include "stdio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void intra_cuda_kernel(int width, int log2_width, int count,
                                  unsigned char *ref_top, unsigned char *ref_left,
                                  unsigned char *dst) {

    int global_tid = threadIdx.x + blockDim.x * blockIdx.x;            //      @ CONTROL
    int global_input_id = global_tid >> (log2_width << 1);             // / 64 @ CONTROL
    int local_thread_id = global_tid & ((1 << (log2_width << 1)) - 1); // % 64 @ CONTROL
    int row = local_thread_id >> log2_width;                           // / 8  @ CONTROL
    int col = local_thread_id & ((1 << log2_width) - 1);               // % 8  @ CONTROL

    int w2p1 = (width << 1) + 1;                                      // @ ADDRESS
    unsigned char *my_ref_top = ref_top + (global_input_id * w2p1);   // @ ADDRESS
    unsigned char *my_ref_left = ref_left + (global_input_id * w2p1); // @ ADDRESS

    if (global_input_id < count) { // @ CONTROL
        // rl[y+1]
        unsigned char *left_address = my_ref_left + row + 1; // @ ADDRESS
        short left = *left_address;                          // @ COMPUTE

        // rt[x+1]
        unsigned char *top_address = my_ref_top + col + 1; // @ ADDRESS
        short top = *top_address;                          // @ COMPUTE

        // rl[w+1]
        unsigned char *bottom_left_address = my_ref_left + width + 1; // @ ADDRESS
        short bottom_left = *bottom_left_address;                     // @ COMPUTE

        // rt[w+1]
        unsigned char *top_right_address = my_ref_top + width + 1; // @ ADDRESS
        short top_right = *top_right_address;                      // @ COMPUTE

        // (w - (x+1)) * rl[y+1]
        short left_val = (width - (col + 1)) * left; // @ COMPUTE

        // (w - (y+1)) * rt[x+1]
        short top_val = (width - (row + 1)) * top; // @ COMPUTE

        // rl[w+1] * (y+1)
        short bottom_left_val = (row + 1) * bottom_left; // @ COMPUTE

        // rt[w+1] * (x+1)
        short top_right_val = (col + 1) * top_right; // @ COMPUTE

        // R5 = (w - (x+1)) * rl[y+1] + rt[w+1] * (x+1) +
        //      (w - (y+1)) * rt[x+1] + rl[w+1] * (y+1) + w
        short total = width + left_val + top_val + bottom_left_val + top_right_val; // @ COMPUTE

        total = total >> (log2_width + 1); // @ COMPUTE

        unsigned char *dst_addr = dst + global_tid; // @ ADDRESS
        *dst_addr = total;                          // @ COMPUTE
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

void intra_cuda(config_t *config,
                input_t *input,
                output_t *output) {

    intra_config_t *intra_config = (intra_config_t *)config;
    intra_input_t *intra_input = (intra_input_t *)input;
    intra_output_t *intra_output = (intra_output_t *)output;

    int count = intra_config->count;
    const int_fast8_t log2_width = intra_config->log2_width;
    const int_fast8_t width = intra_config->width;
    kvz_pixel *ref_top = intra_input->ref_top;
    kvz_pixel *ref_left = intra_input->ref_left;
    kvz_pixel *dst = intra_output->dst;

    uint8_t *d_ref_top, *d_ref_left, *d_dst;
    dim3 blocks((count * 64 + NUM_THREADS - 1) / NUM_THREADS);
    size_t input_size = count * (sizeof(uint8_t) * 17);
    if (input_size % 16) {
        input_size += (16 - (input_size % 16));
    }
    size_t output_size = count * (sizeof(uint8_t) * 64);
    if (output_size % 16) {
        output_size += (16 - (output_size % 16));
    }

    CUDA_SAFE_CALL(cudaMalloc(&d_ref_top, input_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_ref_left, input_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_dst, output_size));

    CUDA_SAFE_CALL(cudaMemcpy(d_ref_top, ref_top, input_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_ref_left, ref_left, input_size, cudaMemcpyHostToDevice));

    intra_cuda_kernel<<<blocks, NUM_THREADS>>>(8, 3, count, d_ref_top, d_ref_left, d_dst);

    CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, output_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_ref_top));
    CUDA_SAFE_CALL(cudaFree(d_ref_left));
    CUDA_SAFE_CALL(cudaFree(d_dst));
}