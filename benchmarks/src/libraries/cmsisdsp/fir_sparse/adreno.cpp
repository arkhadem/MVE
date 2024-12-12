#include "CL/cl.h"
#include "clutil.hpp"
#include "fir_sparse.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)
#define BLOCK_SIZE 950

cl_platform_id fir_sparse_cpPlatform; // OpenCL platform
cl_device_id fir_sparse_device_id;    // device ID
cl_context fir_sparse_context;        // fir_sparse_context
cl_command_queue fir_sparse_queue;    // command fir_sparse_queue
cl_program fir_sparse_program;        // fir_sparse_program
cl_kernel fir_sparse_kernel;          // fir_sparse_kernel

void fir_sparse_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading fir_sparse_kernel
    fp = fopen("fir_sparse.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load fir_sparse_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &fir_sparse_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(fir_sparse_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &fir_sparse_device_id, NULL);
    printErrorString(4, err);

    // Create a fir_sparse_context
    fir_sparse_context = clCreateContext(0, 1, &fir_sparse_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command fir_sparse_queue
    fir_sparse_queue = clCreateCommandQueue(fir_sparse_context, fir_sparse_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute fir_sparse_program from the source buffer
    fir_sparse_program = clCreateProgramWithSource(fir_sparse_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL fir_sparse_kernel fir_sparse_program
    std::string build_option = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);
    err = clBuildProgram(fir_sparse_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);

    // Create the compute fir_sparse_kernel in the fir_sparse_program we wish to run
    fir_sparse_kernel = clCreateKernel(fir_sparse_program, "fir_sparse_adreno_kernel", &err);

    printErrorString(9, err);
}

void fir_sparse_DestroyGPU(config_t *config) {
    clReleaseProgram(fir_sparse_program);
    clReleaseKernel(fir_sparse_kernel);
    clReleaseCommandQueue(fir_sparse_queue);
    clReleaseContext(fir_sparse_context);
}

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
timing_t fir_sparse_adreno(config_t *config,
                           input_t *input,
                           output_t *output) {

    fir_sparse_config_t *fir_sparse_config = (fir_sparse_config_t *)config;
    fir_sparse_input_t *fir_sparse_input = (fir_sparse_input_t *)input;
    fir_sparse_output_t *fir_sparse_output = (fir_sparse_output_t *)output;

    int sample_count = fir_sparse_config->sample_count;
    int coeff_count = fir_sparse_config->coeff_count;
    int effective_coeff_count = fir_sparse_config->effective_coeff_count;
    int input_count = fir_sparse_config->input_count;
    int32_t *src = fir_sparse_input->src;
    int32_t *coeff = fir_sparse_input->coeff;
    int32_t *delay = fir_sparse_input->delay;
    int32_t *dst = fir_sparse_output->dst;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 1;
    size_t global_size = std::ceil(sample_count / (float)BLOCK_SIZE) * BLOCK_SIZE;
    size_t global_item_size[] = {global_size, 1, 1};
    size_t local_item_size[] = {BLOCK_SIZE, 1, 1};

    size_t src_size = input_count * sizeof(float);
    size_t coeff_size = coeff_count * sizeof(float);
    size_t delay_size = effective_coeff_count * sizeof(int);
    size_t dst_size = sample_count * sizeof(float);

    // Create the input and output arrays in device memory for our calculation
    CLOCK_START()
    cl_mem d_src = clCreateBuffer(fir_sparse_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, src_size, NULL, &err);
    cl_mem d_coeff = clCreateBuffer(fir_sparse_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, coeff_size, NULL, &err);
    cl_mem d_delay = clCreateBuffer(fir_sparse_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, delay_size, NULL, &err);
    cl_mem d_dst = clCreateBuffer(fir_sparse_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, dst_size, NULL, &err);
    CLOCK_FINISH(timing.create_buffer)
    printErrorString(-1, err);

    // Write our data set into the input array in device memory
    CLOCK_START()
    float *h_src = (float *)clEnqueueMapBuffer(fir_sparse_queue, d_src, CL_FALSE, CL_MAP_READ, 0, src_size, 0, NULL, NULL, &err);
    float *h_coeff = (float *)clEnqueueMapBuffer(fir_sparse_queue, d_coeff, CL_FALSE, CL_MAP_READ, 0, coeff_size, 0, NULL, NULL, &err);
    int *h_delay = (int *)clEnqueueMapBuffer(fir_sparse_queue, d_delay, CL_FALSE, CL_MAP_READ, 0, delay_size, 0, NULL, NULL, &err);
    float *h_dst = (float *)clEnqueueMapBuffer(fir_sparse_queue, d_dst, CL_FALSE, CL_MAP_WRITE, 0, dst_size, 0, NULL, NULL, &err);
    clFinish(fir_sparse_queue);
    CLOCK_FINISH(timing.map_buffer)
    printErrorString(-2, err);

    err = clSetKernelArg(fir_sparse_kernel, 0, sizeof(int), &sample_count);
    err |= clSetKernelArg(fir_sparse_kernel, 1, sizeof(int), &effective_coeff_count);
    err |= clSetKernelArg(fir_sparse_kernel, 2, sizeof(cl_mem), &d_src);
    err |= clSetKernelArg(fir_sparse_kernel, 3, sizeof(cl_mem), &d_coeff);
    err |= clSetKernelArg(fir_sparse_kernel, 4, sizeof(cl_mem), &d_delay);
    err |= clSetKernelArg(fir_sparse_kernel, 5, sizeof(cl_mem), &d_dst);
    printErrorString(0, err);

    CLOCK_START()
    memcpy(h_src, src, src_size);
    memcpy(h_coeff, coeff, coeff_size);
    memcpy(h_delay, delay, delay_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(fir_sparse_queue, fir_sparse_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(fir_sparse_queue);

    // Execute the fir_sparse_kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(fir_sparse_queue, fir_sparse_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(fir_sparse_queue);
    CLOCK_FINISH(timing.kernel_execute)

    CLOCK_START()
    clEnqueueUnmapMemObject(fir_sparse_queue, d_src, h_src, 0, NULL, NULL);
    clEnqueueUnmapMemObject(fir_sparse_queue, d_coeff, h_coeff, 0, NULL, NULL);
    clEnqueueUnmapMemObject(fir_sparse_queue, d_delay, h_coeff, 0, NULL, NULL);
    clEnqueueUnmapMemObject(fir_sparse_queue, d_dst, h_dst, 0, NULL, NULL);
    CLOCK_FINISH(timing.map_buffer)

    // Read the results from the device
    CLOCK_START()
    memcpy(dst, h_dst, dst_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_coeff);
    clReleaseMemObject(d_delay);
    clReleaseMemObject(d_dst);

    return timing;
}