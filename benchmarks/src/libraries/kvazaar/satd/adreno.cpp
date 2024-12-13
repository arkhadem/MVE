#include "CL/cl.h"
#include "clutil.hpp"
#include "satd.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id satd_cpPlatform; // OpenCL platform
cl_device_id satd_device_id;    // device ID
cl_context satd_context;        // satd_context
cl_command_queue satd_queue;    // command satd_queue
cl_program satd_program;        // satd_program
cl_kernel satd_kernel;          // satd_kernel

void satd_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading satd_kernel
    fp = fopen("satd.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load satd_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &satd_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(satd_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &satd_device_id, NULL);
    printErrorString(4, err);

    // Create a satd_context
    satd_context = clCreateContext(0, 1, &satd_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command satd_queue
    satd_queue = clCreateCommandQueue(satd_context, satd_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute satd_program from the source buffer
    satd_program = clCreateProgramWithSource(satd_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL satd_kernel satd_program
    std::string build_option = "";
    err = clBuildProgram(satd_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);
    if (err != CL_SUCCESS) {
        std::vector<char> buildLog;
        size_t logSize;
        clGetProgramBuildInfo(satd_program, satd_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        buildLog.resize(logSize);
        clGetProgramBuildInfo(satd_program, satd_device_id, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

        std::cout << &buildLog[0] << "\n";
    }

    // Create the compute satd_kernel in the satd_program we wish to run
    satd_kernel = clCreateKernel(satd_program, "satd_adreno_kernel", &err);

    printErrorString(9, err);
}

void satd_DestroyGPU(config_t *config) {
    clReleaseProgram(satd_program);
    clReleaseKernel(satd_kernel);
    clReleaseCommandQueue(satd_queue);
    clReleaseContext(satd_context);
}

timing_t satd_adreno(config_t *config,
                     input_t *input,
                     output_t *output) {
    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t *satd_input = (satd_input_t *)input;
    satd_output_t *satd_output = (satd_output_t *)output;

    int count = satd_config->count;
    uint8_t *piOrg = satd_input->piOrg;
    uint8_t *piCur = satd_input->piCur;
    int32_t *result = satd_output->result;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 3;
    size_t count_size = (size_t)count;
    size_t global_item_size[] = {8, 8, count_size};
    size_t local_item_size[] = {8, 8, 1};

    // Create the input and output arrays in device memory for our calculation
    int32_t coeff[64] = {
        +1, +1, +1, +1, +1, +1, +1, +1,
        +1, -1, +1, -1, +1, -1, +1, -1,
        +1, +1, -1, -1, +1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, -1, +1,
        +1, +1, +1, +1, -1, -1, -1, -1,
        +1, -1, +1, -1, -1, +1, -1, +1,
        +1, +1, -1, -1, -1, -1, +1, +1,
        +1, -1, -1, +1, -1, +1, +1, -1};

    size_t pi_size = 64 * count * sizeof(uint8_t);
    size_t coeff_size = 64 * sizeof(int32_t);
    size_t result_size = 64 * count * sizeof(int32_t);

    CLOCK_START()
    cl_mem d_piOrg = clCreateBuffer(satd_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, pi_size, NULL, NULL);
    cl_mem d_piCur = clCreateBuffer(satd_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, pi_size, NULL, NULL);
    cl_mem d_coeff = clCreateBuffer(satd_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, coeff_size, NULL, NULL);
    cl_mem d_result = clCreateBuffer(satd_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, result_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    uint8_t *h_piOrg = (uint8_t *)clEnqueueMapBuffer(satd_queue, d_piOrg, CL_FALSE, CL_MAP_READ, 0, pi_size, 0, NULL, NULL, &err);
    uint8_t *h_piCur = (uint8_t *)clEnqueueMapBuffer(satd_queue, d_piCur, CL_FALSE, CL_MAP_READ, 0, pi_size, 0, NULL, NULL, &err);
    int32_t *h_coeff = (int32_t *)clEnqueueMapBuffer(satd_queue, d_coeff, CL_FALSE, CL_MAP_READ, 0, coeff_size, 0, NULL, NULL, &err);
    int32_t *h_result = (int32_t *)clEnqueueMapBuffer(satd_queue, d_result, CL_FALSE, CL_MAP_WRITE, 0, result_size, 0, NULL, NULL, &err);
    clFinish(satd_queue);
    CLOCK_FINISH(timing.map_buffer)
    printErrorString(-2, err);

    err = clSetKernelArg(satd_kernel, 0, sizeof(cl_mem), &d_piOrg);
    err |= clSetKernelArg(satd_kernel, 1, sizeof(cl_mem), &d_piCur);
    err |= clSetKernelArg(satd_kernel, 2, sizeof(cl_mem), &d_coeff);
    err |= clSetKernelArg(satd_kernel, 3, sizeof(cl_mem), &d_result);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_piOrg, piOrg, pi_size);
    memcpy(h_piCur, piCur, pi_size);
    memcpy(h_coeff, coeff, coeff_size);
    CLOCK_FINISH(timing.memcpy)

    // // Execute the satd_kernel over the entire range of the data set
    // err = clEnqueueNDRangeKernel(satd_queue, satd_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    // clFinish(satd_queue);

    // Execute the satd_kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(satd_queue, satd_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(satd_queue);
    CLOCK_FINISH(timing.kernel_execute)

    // Read the results from the device
    CLOCK_START()
    memcpy(result, h_result, result_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_piOrg);
    clReleaseMemObject(d_piCur);
    clReleaseMemObject(d_coeff);
    clReleaseMemObject(d_result);

    return timing;
}