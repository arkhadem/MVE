#include "CL/cl.h"
#include "clutil.hpp"
#include "spmm.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stddef.h>
#include <string.h>
#include <string>

#define MAX_SOURCE_SIZE (0x100000)
#define TILE_SIZE_M 32
#define TILE_SIZE_K 4

cl_platform_id spmm_cpPlatform; // OpenCL platform
cl_device_id spmm_device_id;    // device ID
cl_context spmm_context;        // spmm_context
cl_command_queue spmm_queue;    // command spmm_queue
cl_program spmm_program;        // spmm_program
cl_kernel spmm_kernel;          // spmm_kernel

void spmm_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading spmm_kernel
    fp = fopen("spmm.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load spmm_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &spmm_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(spmm_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &spmm_device_id, NULL);
    printErrorString(4, err);

    // Create a spmm_context
    spmm_context = clCreateContext(0, 1, &spmm_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command spmm_queue
    spmm_queue = clCreateCommandQueue(spmm_context, spmm_device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    printErrorString(6, err);

    // Create the compute spmm_program from the source buffer
    spmm_program = clCreateProgramWithSource(spmm_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL spmm_kernel spmm_program
    std::string build_option = "";
    build_option += " -DTILE_SIZE_K=" + std::to_string(TILE_SIZE_K);
    build_option += " -DTILE_SIZE_M=" + std::to_string(TILE_SIZE_M);
    err = clBuildProgram(spmm_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);

    // Create the compute spmm_kernel in the spmm_program we wish to run
    spmm_kernel = clCreateKernel(spmm_program, "x32_spmm", &err);
    printErrorString(9, err);
}

void spmm_DestroyGPU(config_t *config) {
    clReleaseProgram(spmm_program);
    clReleaseKernel(spmm_kernel);
    clReleaseCommandQueue(spmm_queue);
    clReleaseContext(spmm_context);
}

timing_t spmm_adreno(config_t *config,
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

    cl_int err;
    clock_t start, end;
    timing_t timing;
    cl_event event;
    CLOCK_INIT(timing)

    // Device input buffers
    cl_mem d_input;
    size_t input_size = K * M * sizeof(int32_t);
    cl_mem d_bias;
    size_t bias_size = N * sizeof(int32_t);
    cl_mem d_weights;
    size_t weights_size = weight_elements * sizeof(int32_t);
    cl_mem d_IDX;
    size_t IDX_size = weight_elements * sizeof(int32_t);
    cl_mem d_NNZ;
    size_t NNZ_size = (N + 1) * sizeof(uint32_t);

    // Device output buffer
    cl_mem d_output;
    size_t output_size = N * M * sizeof(int32_t);

    int dimention = 3;
    size_t global_0 = ceil(M / (float)TILE_SIZE_M) * TILE_SIZE_M;
    size_t global_1 = TILE_SIZE_K;
    size_t global_2 = N;
    size_t global_item_size[] = {global_0, global_1, global_2};
    size_t local_item_size[] = {TILE_SIZE_M, TILE_SIZE_K, 1};

    // Create the input and output arrays in device memory for our calculation
    CLOCK_START()
    d_input = clCreateBuffer(spmm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, input_size, NULL, NULL);
    d_bias = clCreateBuffer(spmm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bias_size, NULL, NULL);
    d_weights = clCreateBuffer(spmm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, weights_size, NULL, NULL);
    d_IDX = clCreateBuffer(spmm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, IDX_size, NULL, NULL);
    d_NNZ = clCreateBuffer(spmm_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, NNZ_size, NULL, NULL);
    d_output = clCreateBuffer(spmm_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, output_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    int32_t *h_input = (int32_t *)clEnqueueMapBuffer(spmm_queue, d_input, CL_FALSE, CL_MAP_READ, 0, input_size, 0, NULL, NULL, &err);
    int32_t *h_bias = (int32_t *)clEnqueueMapBuffer(spmm_queue, d_bias, CL_FALSE, CL_MAP_READ, 0, bias_size, 0, NULL, NULL, &err);
    int32_t *h_weights = (int32_t *)clEnqueueMapBuffer(spmm_queue, d_weights, CL_FALSE, CL_MAP_READ, 0, weights_size, 0, NULL, NULL, &err);
    int32_t *h_IDX = (int32_t *)clEnqueueMapBuffer(spmm_queue, d_IDX, CL_FALSE, CL_MAP_READ, 0, IDX_size, 0, NULL, NULL, &err);
    uint32_t *h_NNZ = (uint32_t *)clEnqueueMapBuffer(spmm_queue, d_NNZ, CL_FALSE, CL_MAP_READ, 0, NNZ_size, 0, NULL, NULL, &err);
    int32_t *h_output = (int32_t *)clEnqueueMapBuffer(spmm_queue, d_output, CL_FALSE, CL_MAP_WRITE, 0, output_size, 0, NULL, NULL, &err);
    clFinish(spmm_queue);
    CLOCK_FINISH(timing.map_buffer)
    printErrorString(0, err);

    // Set the arguments to our compute spmm_kernel
    err = clSetKernelArg(spmm_kernel, 0, sizeof(int), &M);
    err |= clSetKernelArg(spmm_kernel, 1, sizeof(int), &N);
    err |= clSetKernelArg(spmm_kernel, 2, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(spmm_kernel, 3, sizeof(cl_mem), &d_bias);
    err |= clSetKernelArg(spmm_kernel, 4, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(spmm_kernel, 5, sizeof(cl_mem), &d_IDX);
    err |= clSetKernelArg(spmm_kernel, 6, sizeof(cl_mem), &d_NNZ);
    err |= clSetKernelArg(spmm_kernel, 7, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(spmm_kernel, 8, sizeof(int32_t), &min);
    err |= clSetKernelArg(spmm_kernel, 9, sizeof(int32_t), &max);
    printErrorString(1, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_input, input, input_size);
    memcpy(h_bias, bias, bias_size);
    memcpy(h_weights, weights, weights_size);
    memcpy(h_IDX, IDX, IDX_size);
    memcpy(h_NNZ, NNZ, NNZ_size);
    CLOCK_FINISH(timing.memcpy)

    // Execute the spmm_kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(spmm_queue, spmm_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &event);
    printErrorString(2, err);
    // Wait for the command spmm_queue to get serviced before reading back results
    clFinish(spmm_queue);

    // Execute the spmm_kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(spmm_queue, spmm_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &event);
    PROF_FINISH(spmm_queue)

    CLOCK_START()
    memcpy(output, h_output, output_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_IDX);
    clReleaseMemObject(d_NNZ);
    clReleaseMemObject(d_output);

    return timing;
}