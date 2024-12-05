#include "CL/cl.h"
#include "clutil.hpp"
#include "dct.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id dct_cpPlatform; // OpenCL platform
cl_device_id dct_device_id;    // device ID
cl_context dct_context;        // dct_context
cl_command_queue dct_queue;    // command dct_queue
cl_program dct_program;        // dct_program
cl_kernel dct_kernel;          // dct_kernel

void dct_InitGPU() {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading dct_kernel
    fp = fopen("dct.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load dct_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &dct_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(dct_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &dct_device_id, NULL);
    printErrorString(4, err);

    // Create a dct_context
    dct_context = clCreateContext(0, 1, &dct_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command dct_queue
    dct_queue = clCreateCommandQueue(dct_context, dct_device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    printErrorString(6, err);

    // Create the compute dct_program from the source buffer
    dct_program = clCreateProgramWithSource(dct_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL dct_kernel dct_program
    std::string build_option = "";
    err = clBuildProgram(dct_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);
    if (err != CL_SUCCESS) {
        std::vector<char> buildLog;
        size_t logSize;
        clGetProgramBuildInfo(dct_program, dct_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        buildLog.resize(logSize);
        clGetProgramBuildInfo(dct_program, dct_device_id, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

        std::cout << &buildLog[0] << "\n";
    }

    // Create the compute dct_kernel in the dct_program we wish to run
    dct_kernel = clCreateKernel(dct_program, "dct_adreno_kernel", &err);

    printErrorString(9, err);
}

void dct_DestroyGPU() {
    clReleaseProgram(dct_program);
    clReleaseKernel(dct_kernel);
    clReleaseCommandQueue(dct_queue);
    clReleaseContext(dct_context);
}

timing_t dct_adreno(config_t *config,
                    input_t *input,
                    output_t *output) {
    const int16_t coeff[64] = {
        64, 64, 64, 64, 64, 64, 64, 64,
        89, 75, 50, 18, -18, -50, -75, -89,
        83, 36, -36, -83, -83, -36, 36, 83,
        75, -18, -89, -50, 50, 89, 18, -75,
        64, -64, -64, 64, 64, -64, -64, 64,
        50, -89, 18, 75, -75, -18, 89, -50,
        36, -83, 83, -36, -36, 83, -83, 36,
        18, -50, 75, -89, 89, -75, 50, -18};

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t *dct_input = (dct_input_t *)input;
    dct_output_t *dct_output = (dct_output_t *)output;

    int count = dct_config->count;
    int8_t *bitdepth = dct_config->bitdepth;
    int16_t *in = dct_input->input;
    int16_t *out = dct_output->output;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    cl_event event;
    dct_InitGPU();
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 3;
    size_t count_size = (size_t)ceil((float)count / 8.00);
    size_t global_item_size[] = {8, 8, count_size};
    size_t local_item_size[] = {8, 8, 1};

    // Create the input and output arrays in device memory for our calculation
    size_t inout_size = 64 * count * sizeof(int16_t);
    size_t bitdepth_size = count * sizeof(int8_t);
    size_t coeff_size = 64 * sizeof(int16_t);

    CLOCK_START()
    // Create the input and output arrays in device memory for our calculation
    cl_mem d_input = clCreateBuffer(dct_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, inout_size, NULL, NULL);
    cl_mem d_output = clCreateBuffer(dct_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, inout_size, NULL, NULL);
    cl_mem d_bitdepth = clCreateBuffer(dct_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bitdepth_size, NULL, NULL);
    cl_mem d_coeff = clCreateBuffer(dct_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, coeff_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    int16_t *h_input = (int16_t *)clEnqueueMapBuffer(dct_queue, d_input, CL_FALSE, CL_MAP_READ, 0, inout_size, 0, NULL, NULL, &err);
    int16_t *h_output = (int16_t *)clEnqueueMapBuffer(dct_queue, d_output, CL_FALSE, CL_MAP_WRITE, 0, inout_size, 0, NULL, NULL, &err);
    int8_t *h_bitdepth = (int8_t *)clEnqueueMapBuffer(dct_queue, d_bitdepth, CL_FALSE, CL_MAP_READ, 0, bitdepth_size, 0, NULL, NULL, &err);
    int16_t *h_coeff = (int16_t *)clEnqueueMapBuffer(dct_queue, d_coeff, CL_FALSE, CL_MAP_READ, 0, coeff_size, 0, NULL, NULL, &err);
    clFinish(dct_queue);
    CLOCK_FINISH(timing.map_buffer)

    printErrorString(-2, err);

    err = clSetKernelArg(dct_kernel, 0, sizeof(int), &count);
    err |= clSetKernelArg(dct_kernel, 1, sizeof(cl_mem), &d_bitdepth);
    err |= clSetKernelArg(dct_kernel, 2, sizeof(cl_mem), &d_coeff);
    err |= clSetKernelArg(dct_kernel, 3, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(dct_kernel, 4, sizeof(cl_mem), &d_output);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_input, input, inout_size);
    memcpy(h_bitdepth, bitdepth, bitdepth_size);
    memcpy(h_coeff, coeff, coeff_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(dct_queue, dct_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(dct_queue);

    // Execute the dct_kernel over the entire range of the data set
    // CLOCK_START()
    err = clEnqueueNDRangeKernel(dct_queue, dct_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &event);
    // clFinish(dct_queue);
    // CLOCK_FINISH(timing.kernel_execute)
    PROF_FINISH(dct_queue)

    CLOCK_START()
    memcpy(output, h_output, inout_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_bitdepth);
    clReleaseMemObject(d_coeff);
    clReleaseMemObject(d_output);

    dct_DestroyGPU();

    return timing;
}