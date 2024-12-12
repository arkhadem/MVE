#include "CL/cl.h"
#include "clutil.hpp"
#include "csum.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)
#define BLOCK_SIZE 1024

cl_platform_id csum_cpPlatform; // OpenCL platform
cl_device_id csum_device_id;    // device ID
cl_context csum_context;        // csum_context
cl_command_queue csum_queue;    // command csum_queue
cl_program csum_program;        // csum_program
cl_kernel csum_kernel;          // csum_kernel

void csum_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading csum_kernel
    fp = fopen("csum.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load csum_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &csum_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(csum_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &csum_device_id, NULL);
    printErrorString(4, err);

    // Create a csum_context
    csum_context = clCreateContext(0, 1, &csum_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command csum_queue
    csum_queue = clCreateCommandQueue(csum_context, csum_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute csum_program from the source buffer
    csum_program = clCreateProgramWithSource(csum_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL csum_kernel csum_program
    std::string build_option = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);
    err = clBuildProgram(csum_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);
    if (err != CL_SUCCESS) {
        std::vector<char> buildLog;
        size_t logSize;
        clGetProgramBuildInfo(csum_program, csum_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        buildLog.resize(logSize);
        clGetProgramBuildInfo(csum_program, csum_device_id, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

        std::cout << &buildLog[0] << "\n";
    }

    // Create the compute csum_kernel in the csum_program we wish to run
    csum_kernel = clCreateKernel(csum_program, "csum_adreno_kernel", &err);

    printErrorString(9, err);
}

void csum_DestroyGPU(config_t *config) {
    clReleaseProgram(csum_program);
    clReleaseKernel(csum_kernel);
    clReleaseCommandQueue(csum_queue);
    clReleaseContext(csum_context);
}

timing_t csum_adreno(config_t *config,
                     input_t *input,
                     output_t *output) {
    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t *csum_input = (csum_input_t *)input;
    csum_output_t *csum_output = (csum_output_t *)output;

    int count = csum_config->count;
    int32_t *ptr = csum_input->ptr;
    int16_t *sum = csum_output->sum;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 2;
    size_t count_size = count;
    size_t global_item_size_gemm[] = {BLOCK_SIZE, count_size, 1};
    size_t local_item_size[] = {BLOCK_SIZE, 1, 1};

    // Create the input and output arrays in device memory for our calculation
    size_t ptr_size = count * BLOCK_16K * sizeof(int32_t);
    size_t sum_size = count * sizeof(int16_t);

    CLOCK_START()
    cl_mem d_ptr = clCreateBuffer(csum_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, ptr_size, NULL, NULL);
    cl_mem d_sum = clCreateBuffer(csum_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sum_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    int32_t *h_ptr = (int32_t *)clEnqueueMapBuffer(csum_queue, d_ptr, CL_FALSE, CL_MAP_READ, 0, ptr_size, 0, NULL, NULL, &err);
    int16_t *h_sum = (int16_t *)clEnqueueMapBuffer(csum_queue, d_sum, CL_FALSE, CL_MAP_WRITE, 0, sum_size, 0, NULL, NULL, &err);
    clFinish(csum_queue);
    CLOCK_FINISH(timing.map_buffer)
    printErrorString(-2, err);

    err = clSetKernelArg(csum_kernel, 0, sizeof(cl_mem), &d_ptr);
    err |= clSetKernelArg(csum_kernel, 1, sizeof(cl_mem), &d_sum);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_ptr, ptr, ptr_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(csum_queue, csum_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, NULL);
    clFinish(csum_queue);

    // Execute the csum_kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(csum_queue, csum_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, NULL);
    clFinish(csum_queue);
    CLOCK_FINISH(timing.kernel_execute)

    CLOCK_START()
    memcpy(sum, h_sum, sum_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_ptr);
    clReleaseMemObject(d_sum);

    return timing;
}