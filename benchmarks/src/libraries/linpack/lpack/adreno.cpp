#include "CL/cl.h"
#include "clutil.hpp"
#include "lpack.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id lpack_cpPlatform; // OpenCL platform
cl_device_id lpack_device_id;    // device ID
cl_context lpack_context;        // lpack_context
cl_command_queue lpack_queue;    // command lpack_queue
cl_program lpack_program;        // lpack_program
cl_kernel lpack_kernel;          // lpack_kernel

void lpack_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading lpack_kernel
    fp = fopen("lpack.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load lpack_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &lpack_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(lpack_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &lpack_device_id, NULL);
    printErrorString(4, err);

    // Create a lpack_context
    lpack_context = clCreateContext(0, 1, &lpack_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command lpack_queue
    lpack_queue = clCreateCommandQueue(lpack_context, lpack_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute lpack_program from the source buffer
    lpack_program = clCreateProgramWithSource(lpack_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL lpack_kernel lpack_program
    std::string build_option = "";
    err = clBuildProgram(lpack_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);

    // Create the compute lpack_kernel in the lpack_program we wish to run
    lpack_kernel = clCreateKernel(lpack_program, "lpack_adreno_kernel", &err);

    printErrorString(9, err);
}

void lpack_DestroyGPU(config_t *config) {
    clReleaseProgram(lpack_program);
    clReleaseKernel(lpack_kernel);
    clReleaseCommandQueue(lpack_queue);
    clReleaseContext(lpack_context);
}

timing_t lpack_adreno(config_t *config,
                      input_t *input,
                      output_t *output) {
    lpack_config_t *lpack_config = (lpack_config_t *)config;
    lpack_input_t *lpack_input = (lpack_input_t *)input;
    lpack_output_t *lpack_output = (lpack_output_t *)output;

    int n = lpack_config->n;
    int32_t *da = lpack_input->da;
    int32_t *dx = lpack_input->dx;
    int32_t *dyin = lpack_input->dyin;
    int32_t *dyout = lpack_output->dyout;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 1;
    size_t global_size = ceil(n / 128.00) * 128.00;
    size_t global_item_size_gemm[] = {global_size, 1, 1};
    size_t local_item_size[] = {128, 1, 1};

    // Create the input and output arrays in device memory for our calculation
    size_t d_size = n * sizeof(int32_t);
    CLOCK_START()
    cl_mem d_dx = clCreateBuffer(lpack_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, d_size, NULL, NULL);
    cl_mem d_dyin = clCreateBuffer(lpack_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, d_size, NULL, NULL);
    cl_mem d_dyout = clCreateBuffer(lpack_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, d_size, NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    CLOCK_START()
    int32_t *h_dx = (int32_t *)clEnqueueMapBuffer(lpack_queue, d_dx, CL_FALSE, CL_MAP_READ, 0, d_size, 0, NULL, NULL, &err);
    int32_t *h_dyin = (int32_t *)clEnqueueMapBuffer(lpack_queue, d_dyin, CL_FALSE, CL_MAP_READ, 0, d_size, 0, NULL, NULL, &err);
    int32_t *h_dyout = (int32_t *)clEnqueueMapBuffer(lpack_queue, d_dyout, CL_FALSE, CL_MAP_WRITE, 0, d_size, 0, NULL, NULL, &err);
    clFinish(lpack_queue);
    CLOCK_FINISH(timing.map_buffer)
    printErrorString(-2, err);

    err = clSetKernelArg(lpack_kernel, 0, sizeof(int32_t), &da[0]);
    err |= clSetKernelArg(lpack_kernel, 1, sizeof(cl_mem), &d_dx);
    err |= clSetKernelArg(lpack_kernel, 2, sizeof(cl_mem), &d_dyin);
    err |= clSetKernelArg(lpack_kernel, 3, sizeof(cl_mem), &d_dyout);
    printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_dx, dx, d_size);
    memcpy(h_dyin, dyin, d_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(lpack_queue, lpack_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, NULL);
    clFinish(lpack_queue);

    CLOCK_START()
    err = clEnqueueNDRangeKernel(lpack_queue, lpack_kernel, dimention, NULL, global_item_size_gemm, local_item_size, 0, NULL, NULL);
    clFinish(lpack_queue);
    CLOCK_FINISH(timing.kernel_execute)

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(dyout, h_dyout, d_size);
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    clReleaseMemObject(d_dx);
    clReleaseMemObject(d_dyin);
    clReleaseMemObject(d_dyout);

    return timing;
}