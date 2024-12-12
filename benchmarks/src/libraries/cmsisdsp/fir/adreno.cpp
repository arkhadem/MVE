#include "CL/cl.h"
#include "clutil.hpp"
#include "fir.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stddef.h>
#include <string.h>
#include <string>

#define MAX_SOURCE_SIZE (0x100000)
#define BLOCK_SIZE 950

cl_platform_id fir_cpPlatform; // OpenCL platform
cl_device_id fir_device_id;    // device ID
cl_context fir_context;        // fir_context
cl_command_queue fir_queue;    // command fir_queue
cl_program fir_program;        // fir_program
cl_kernel fir_kernel;          // fir_kernel

void fir_InitGPU(config_t *config) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading fir_kernel
    fp = fopen("fir.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load fir_kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &fir_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(fir_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &fir_device_id, NULL);
    printErrorString(4, err);

    // Create a fir_context
    fir_context = clCreateContext(0, 1, &fir_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command fir_queue
    fir_queue = clCreateCommandQueue(fir_context, fir_device_id, 0, &err);
    printErrorString(6, err);

    // Create the compute fir_program from the source buffer
    fir_program = clCreateProgramWithSource(fir_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL fir_kernel fir_program
    std::string build_option = "";
    err = clBuildProgram(fir_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);

    // Create the compute fir_kernel in the fir_program we wish to run
    fir_kernel = clCreateKernel(fir_program, "fir_adreno_kernel", &err);

    printErrorString(9, err);
}

void fir_DestroyGPU(config_t *config) {
    clReleaseProgram(fir_program);
    clReleaseKernel(fir_kernel);
    clReleaseCommandQueue(fir_queue);
    clReleaseContext(fir_context);
}

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
timing_t fir_adreno(config_t *config,
                    input_t *input,
                    output_t *output) {

    fir_config_t *fir_config = (fir_config_t *)config;
    fir_input_t *fir_input = (fir_input_t *)input;
    fir_output_t *fir_output = (fir_output_t *)output;

    int sample_count = fir_config->sample_count;
    int coeff_count = fir_config->coeff_count;
    int32_t *src = fir_input->src;
    int32_t *coeff = fir_input->coeff;
    int32_t *dst = fir_output->dst;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int dimention = 1;
    size_t input_per_BLOCK = BLOCK_SIZE + coeff_count - 1;
    size_t global_size = std::ceil(sample_count / (float)BLOCK_SIZE) * input_per_BLOCK;
    size_t global_item_size[] = {global_size, 1, 1};
    size_t local_item_size[] = {input_per_BLOCK, 1, 1};

    size_t src_size = (sample_count + coeff_count - 1) * sizeof(int32_t);
    size_t coeff_size = coeff_count * sizeof(int32_t);
    size_t dst_size = sample_count * sizeof(int32_t);

    // Create the input and output arrays in device memory for our calculation
    // Note, since we are using NULL for the data pointer, we HAVE to use CL_MEM_ALLOC_HOST_PTR
    // This allocates memory on the devices
    CLOCK_START()
    cl_mem d_src = clCreateBuffer(fir_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, src_size, NULL, NULL);
    // printErrorString(-5, err);
    cl_mem d_coeff = clCreateBuffer(fir_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, coeff_size, NULL, NULL);
    // printErrorString(-4, err);
    cl_mem d_dst = clCreateBuffer(fir_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, dst_size, NULL, NULL);
    // printErrorString(-3, err);
    CLOCK_FINISH(timing.create_buffer)

    //Map the Device memory to host memory, aka pinning it
    CLOCK_START()
    int32_t *h_src = (int32_t *)clEnqueueMapBuffer(fir_queue, d_src, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, src_size, 0, NULL, NULL, &err);
    int32_t *h_coeff = (int32_t *)clEnqueueMapBuffer(fir_queue, d_coeff, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, coeff_size, 0, NULL, NULL, &err);
    int32_t *h_dst = (int32_t *)clEnqueueMapBuffer(fir_queue, d_dst, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, dst_size, 0, NULL, NULL, &err);
    // printErrorString(-2, err);
    clFinish(fir_queue);
    CLOCK_FINISH(timing.map_buffer)

    err = clSetKernelArg(fir_kernel, 0, sizeof(int), &sample_count);
    err |= clSetKernelArg(fir_kernel, 1, sizeof(int), &coeff_count);
    err |= clSetKernelArg(fir_kernel, 2, sizeof(cl_mem), &d_src);
    err |= clSetKernelArg(fir_kernel, 3, sizeof(cl_mem), &d_coeff);
    err |= clSetKernelArg(fir_kernel, 4, sizeof(cl_mem), &d_dst);
    err |= clSetKernelArg(fir_kernel, 5, input_per_BLOCK * sizeof(int32_t), NULL);
    // printErrorString(0, err);

    // Copy from host memory to pinned host memory which copies to the card automatically
    CLOCK_START()
    memcpy(h_src, src, src_size);
    memcpy(h_coeff, coeff, coeff_size);
    CLOCK_FINISH(timing.memcpy)

    err = clEnqueueNDRangeKernel(fir_queue, fir_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(fir_queue);

    // Execute the fir_kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(fir_queue, fir_kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(fir_queue);
    CLOCK_FINISH(timing.kernel_execute)

    CLOCK_START()
    clEnqueueUnmapMemObject(fir_queue, d_src, h_src, 0, NULL, NULL);
    clEnqueueUnmapMemObject(fir_queue, d_coeff, h_coeff, 0, NULL, NULL);
    clEnqueueUnmapMemObject(fir_queue, d_dst, h_dst, 0, NULL, NULL);
    CLOCK_FINISH(timing.map_buffer)

    CLOCK_START()
    memcpy(dst, h_dst, dst_size);
    CLOCK_FINISH(timing.memcpy)

    // CLOCK_START()
    // release OpenCL resources
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_coeff);
    clReleaseMemObject(d_dst);
    // CLOCK_FINISH(timing.create_buffer)

    return timing;
}