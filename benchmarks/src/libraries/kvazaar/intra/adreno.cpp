#include "clutil.hpp"
#include "intra.hpp"
#include <CL/cl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id intra_cpPlatform; // OpenCL platform
cl_device_id intra_device_id;    // device ID
cl_context intra_context;        // intra_context
cl_command_queue intra_queue;    // command intra_queue
cl_program intra_program;        // intra_program
cl_kernel intra_kernel;          // intra_kernel

void intra_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading intra_kernel
    fp = fopen("intra.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load intra_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &intra_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(intra_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &intra_device_id, NULL);
    printErrorString(4, err);

    // Create a intra_context
    intra_context = clCreateContext(0, 1, &intra_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command intra_queue
    intra_queue = clCreateCommandQueue(intra_context, intra_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute intra_program from the source buffer
    intra_program = clCreateProgramWithSource(intra_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL intra_kernel intra_program
    std::string build_option = "";
    err = clBuildProgram(intra_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);
    if (err != CL_SUCCESS) {
        std::vector<char> buildLog;
        size_t logSize;
        clGetProgramBuildInfo(intra_program, intra_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        buildLog.resize(logSize);
        clGetProgramBuildInfo(intra_program, intra_device_id, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

        std::cout << &buildLog[0] << "\n";
    }

    // Create the compute intra_kernel in the intra_program we wish to run
    intra_kernel = clCreateKernel(intra_program, "intra_adreno_kernel", &err);

    printErrorString(9, err);
}

void intra_DestroyGPU(config_t *config) {
    clReleaseProgram(intra_program);
    clReleaseKernel(intra_kernel);
    clReleaseCommandQueue(intra_queue);
    clReleaseContext(intra_context);
}

timing_t intra_adreno(config_t *config,
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

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 3;
    size_t count_size = (size_t)count;
    size_t global_item_size[] = {8, 8, count_size};
    size_t local_item_size[] = {8, 8, 1};

    size_t ref_size = 17 * count * sizeof(uint8_t);
    size_t dst_size = 64 * count * sizeof(uint8_t);

    CLOCK_START()
    // Create the input and output arrays in device memory for our calculation
    cl_mem d_ref_top = clCreateBuffer(intra_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, ref_size, NULL, NULL);
    cl_mem d_ref_left = clCreateBuffer(intra_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, ref_size, NULL, NULL);
    cl_mem d_dst = clCreateBuffer(intra_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, dst_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    uint8_t *h_ref_top = (uint8_t *)clEnqueueMapBuffer(intra_queue, d_ref_top, CL_FALSE, CL_MAP_READ, 0, ref_size, 0, NULL, NULL, &err);
    uint8_t *h_ref_left = (uint8_t *)clEnqueueMapBuffer(intra_queue, d_ref_left, CL_FALSE, CL_MAP_READ, 0, ref_size, 0, NULL, NULL, &err);
    uint8_t *h_dst = (uint8_t *)clEnqueueMapBuffer(intra_queue, d_dst, CL_FALSE, CL_MAP_WRITE, 0, dst_size, 0, NULL, NULL, &err);
    clFinish(intra_queue);
    CLOCK_FINISH(timing.map_buffer)

    printErrorString(-2, err);

    err = clSetKernelArg(intra_kernel, 0, sizeof(cl_mem), &d_ref_top);
    err |= clSetKernelArg(intra_kernel, 1, sizeof(cl_mem), &d_ref_left);
    err |= clSetKernelArg(intra_kernel, 2, sizeof(cl_mem), &d_dst);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_ref_top, ref_top, ref_size);
    memcpy(h_ref_left, ref_left, ref_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(intra_queue, intra_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(intra_queue);

    // Execute the intra_kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(intra_queue, intra_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(intra_queue);
    CLOCK_FINISH(timing.kernel_execute)

    // Read the dsts from the device
    CLOCK_START()
    memcpy(dst, h_dst, dst_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_ref_top);
    clReleaseMemObject(d_ref_left);
    clReleaseMemObject(d_dst);

    return timing;
}