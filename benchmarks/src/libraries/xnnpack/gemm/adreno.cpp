#include "CL/cl.h"
#include "clutil.hpp"
#include "gemm.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)
#define GEMMK 1 // Kernel to choose: 0 regular, 1 with 2D register tiling
#define MWG 8   // Tile-size in dimension M (e.g. 64, 128)
#define NWG 8   // Tile-size in dimension N (e.g. 64, 128)
#define KWG 8   // Tile-size in dimension K (e.g. 8, 16)
#define MDIMC 8 // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#define NDIMC 8 // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#define MDIMA 8 // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#define NDIMB 8 // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#define KWI 1   // Unroll factor of the KWG loop (smaller or equal than KWG)
#define KREG 1  // Amount of register tiling in second dimension, multiple of 1 (kernel 1 only)

cl_platform_id gemm_cpPlatform; // OpenCL platform
cl_device_id gemm_device_id;    // device ID
cl_context gemm_context;        // gemm_context
cl_command_queue gemm_queue;    // command gemm_queue
cl_program gemm_program;        // gemm_program
cl_kernel gemm_kernel;          // kernel

void gemm_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading kernel
    fp = fopen("gemm.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &gemm_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(gemm_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &gemm_device_id, NULL);
    printErrorString(4, err);

    // Create a gemm_context
    gemm_context = clCreateContext(0, 1, &gemm_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command gemm_queue
    gemm_queue = clCreateCommandQueue(gemm_context, gemm_device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    printErrorString(6, err);

    // Create the compute gemm_program from the source buffer
    gemm_program = clCreateProgramWithSource(gemm_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL kernel gemm_program
    std::string build_option = "";
    build_option += " -DGEMMK=" + std::to_string(GEMMK);
    build_option += " -DMWG=" + std::to_string(MWG);
    build_option += " -DNWG=" + std::to_string(NWG);
    build_option += " -DKWG=" + std::to_string(KWG);
    build_option += " -DMDIMC=" + std::to_string(MDIMC);
    build_option += " -DNDIMC=" + std::to_string(NDIMC);
    build_option += " -DMDIMA=" + std::to_string(MDIMA);
    build_option += " -DNDIMB=" + std::to_string(NDIMB);
    build_option += " -DKWI=" + std::to_string(KWI);
    build_option += " -DKREG=" + std::to_string(KREG);
    err = clBuildProgram(gemm_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);

    // Create the compute kernel in the gemm_program we wish to run
    gemm_kernel = clCreateKernel(gemm_program, "Xgemm", &err);

    printErrorString(9, err);
}

void gemm_DestroyGPU(config_t *config) {
    clReleaseProgram(gemm_program);
    clReleaseKernel(gemm_kernel);
    clReleaseCommandQueue(gemm_queue);
    clReleaseContext(gemm_context);
}

timing_t gemm_adreno(config_t *config,
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

    cl_int err;
    clock_t start, end;
    timing_t timing;
    cl_event event;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 2;
    size_t global_divider_one = GEMMK ? NWG : MWG;
    size_t global_divider_two = GEMMK ? MWG : NWG;
    size_t m_ceiled = ceil(N / float(MWG)) * MWG;
    size_t n_ceiled = ceil(M / float(NWG)) * NWG;
    size_t c_one_i = GEMMK ? n_ceiled : m_ceiled;
    size_t c_two_i = GEMMK ? m_ceiled : n_ceiled;
    size_t global_item_size_gemm[] = {
        (c_one_i * MDIMC) / global_divider_one,
        (c_two_i * NDIMC) / global_divider_two,
        1};
    size_t local_item_size[] = {MDIMC, NDIMC, 1};

    // Create the input and output arrays in device memory for our calculation
    size_t input_size = K * M * sizeof(float);
    size_t bias_size = N * sizeof(float);
    size_t weights_size = N * K * sizeof(float);
    size_t output_size = N * M * sizeof(float);

    CLOCK_START()
    cl_mem d_input = clCreateBuffer(gemm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, input_size, NULL, NULL);
    cl_mem d_bias = clCreateBuffer(gemm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bias_size, NULL, NULL);
    cl_mem d_weights = clCreateBuffer(gemm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, weights_size, NULL, NULL);
    cl_mem d_output = clCreateBuffer(gemm_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, output_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    float *h_input = (float *)clEnqueueMapBuffer(gemm_queue, d_input, CL_FALSE, CL_MAP_READ, 0, input_size, 0, NULL, NULL, &err);
    float *h_bias = (float *)clEnqueueMapBuffer(gemm_queue, d_bias, CL_FALSE, CL_MAP_READ, 0, bias_size, 0, NULL, NULL, &err);
    float *h_weights = (float *)clEnqueueMapBuffer(gemm_queue, d_weights, CL_FALSE, CL_MAP_READ, 0, weights_size, 0, NULL, NULL, &err);
    float *h_output = (float *)clEnqueueMapBuffer(gemm_queue, d_output, CL_FALSE, CL_MAP_WRITE, 0, output_size, 0, NULL, NULL, &err);
    clFinish(gemm_queue);
    CLOCK_FINISH(timing.map_buffer)

    printErrorString(-2, err);

    // Set the arguments to our compute kernel
    err = clSetKernelArg(gemm_kernel, 0, sizeof(int), &N);
    err |= clSetKernelArg(gemm_kernel, 1, sizeof(int), &M);
    err |= clSetKernelArg(gemm_kernel, 2, sizeof(int), &K);
    err |= clSetKernelArg(gemm_kernel, 3, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(gemm_kernel, 4, sizeof(cl_mem), &d_bias);
    err |= clSetKernelArg(gemm_kernel, 5, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(gemm_kernel, 6, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(gemm_kernel, 7, sizeof(float), &min);
    err |= clSetKernelArg(gemm_kernel, 8, sizeof(float), &max);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_input, input, input_size);
    memcpy(h_bias, bias, bias_size);
    memcpy(h_weights, weights, weights_size);
    CLOCK_FINISH(timing.memcpy)

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(gemm_queue, gemm_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, NULL);
    printErrorString(2, err);
    // Wait for the command gemm_queue to get serviced before reading back results
    clFinish(gemm_queue);

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(gemm_queue, gemm_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, &event);
    PROF_FINISH(gemm_queue)

    CLOCK_START()
    memcpy(output, h_output, output_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_output);

    return timing;
}