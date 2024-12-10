#include "CL/cl.h"
#include "clutil.hpp"
#include "fir_lattice.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)
#define BLOCK_SIZE 1024

cl_platform_id fir_lattice_cpPlatform; // OpenCL platform
cl_device_id fir_lattice_device_id;    // device ID
cl_context fir_lattice_context;        // fir_lattice_context
cl_command_queue fir_lattice_queue;    // command fir_lattice_queue
cl_program fir_lattice_program;        // fir_lattice_program
cl_kernel *fir_lattice_kernels;        // kernel

void fir_lattice_InitGPU(config_t *config) {
    cl_int err;

    int iteration = (int)(std::ceil(((fir_lattice_config_t *)config)->sample_count / (float)BLOCK_SIZE));

    fir_lattice_kernels = (cl_kernel *)malloc(iteration * sizeof(cl_kernel)); // kernel

    FILE *fp;
    char *source_str;
    size_t source_size;

    // Reading kernel
    fp = fopen("fir_lattice.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose(fp);

    // Bind to platform
    err = clGetPlatformIDs(1, &fir_lattice_cpPlatform, NULL);
    printErrorString(3, err);

    // Get ID for the device
    err = clGetDeviceIDs(fir_lattice_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &fir_lattice_device_id, NULL);
    printErrorString(4, err);

    // Create a fir_lattice_context
    fir_lattice_context = clCreateContext(0, 1, &fir_lattice_device_id, NULL, NULL, &err);
    printErrorString(5, err);

    // Create a command fir_lattice_queue
    fir_lattice_queue = clCreateCommandQueue(fir_lattice_context, fir_lattice_device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    printErrorString(6, err);

    // Create the compute fir_lattice_program from the source buffer
    fir_lattice_program = clCreateProgramWithSource(fir_lattice_context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    printErrorString(7, err);

    // Build and compile the OpenCL kernel fir_lattice_program
    std::string build_option = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);
    err = clBuildProgram(fir_lattice_program, 0, NULL, build_option.c_str(), NULL, NULL);
    printErrorString(8, err);
    if (err != CL_SUCCESS) {
        std::vector<char> buildLog;
        size_t logSize;
        clGetProgramBuildInfo(fir_lattice_program, fir_lattice_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        buildLog.resize(logSize);
        clGetProgramBuildInfo(fir_lattice_program, fir_lattice_device_id, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

        printf("%s\n", buildLog.data());
    }

    // Create the compute kernel in the fir_lattice_program we wish to run
    for (int itr = 0; itr < iteration; itr++) {
        fir_lattice_kernels[itr] = clCreateKernel(fir_lattice_program, "fir_lattice_adreno_kernel", &err);
    }

    printErrorString(9, err);
}

void fir_lattice_DestroyGPU(config_t *config) {
    int iteration = (int)(std::ceil(((fir_lattice_config_t *)config)->sample_count / (float)BLOCK_SIZE));
    for (int i = 0; i < iteration; i++) {
        clReleaseKernel(fir_lattice_kernels[i]);
    }
    clReleaseProgram(fir_lattice_program);
    clReleaseCommandQueue(fir_lattice_queue);
    clReleaseContext(fir_lattice_context);
}

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
timing_t fir_lattice_adreno(config_t *config,
                            input_t *input,
                            output_t *output) {

    fir_lattice_config_t *fir_lattice_config = (fir_lattice_config_t *)config;
    fir_lattice_input_t *fir_lattice_input = (fir_lattice_input_t *)input;
    fir_lattice_output_t *fir_lattice_output = (fir_lattice_output_t *)output;

    int sample_count = fir_lattice_config->sample_count;
    int coeff_count = fir_lattice_config->coeff_count;
    int32_t *src = fir_lattice_input->src;
    int32_t *coeff = fir_lattice_input->coeff;
    int32_t *dst = fir_lattice_output->dst;

    cl_int err;
    clock_t start, end;
    timing_t timing;
    cl_event event1, event2;
    CLOCK_INIT(timing)

    // Computes the global and local thread sizes
    int iteration = (int)(std::ceil(sample_count / (float)BLOCK_SIZE));
    int dimention = 1;
    size_t fm_global_item_size[] = {BLOCK_SIZE, 1, 1};
    size_t fm_local_item_size[] = {BLOCK_SIZE, 1, 1};
    size_t last_size;
    if (sample_count % BLOCK_SIZE == 0) {
        last_size = BLOCK_SIZE;
    } else {
        last_size = sample_count % BLOCK_SIZE;
    }
    size_t l_global_item_size[] = {last_size, 1, 1};
    size_t l_local_item_size[] = {last_size, 1, 1};
    size_t coeff_size = coeff_count * sizeof(float);

    // Create the input and output arrays in device memory for our calculation
    cl_mem *d_src = (cl_mem *)malloc(iteration * sizeof(cl_mem));
    cl_mem *d_dst = (cl_mem *)malloc(iteration * sizeof(cl_mem));

    CLOCK_START()
    for (int itr = 0; itr < iteration - 1; itr++) {
        d_src[itr] = clCreateBuffer(fir_lattice_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BLOCK_SIZE * sizeof(float), NULL, NULL);
        d_dst[itr] = clCreateBuffer(fir_lattice_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, BLOCK_SIZE * sizeof(float), NULL, NULL);
    }
    d_src[iteration - 1] = clCreateBuffer(fir_lattice_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, last_size * sizeof(float), NULL, NULL);
    d_dst[iteration - 1] = clCreateBuffer(fir_lattice_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, last_size * sizeof(float), NULL, NULL);

    cl_mem d_coeff = clCreateBuffer(fir_lattice_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, coeff_size, NULL, NULL);

    cl_mem d_initial_g[2];
    d_initial_g[0] = clCreateBuffer(fir_lattice_context, CL_MEM_READ_WRITE, (coeff_count + 1) * sizeof(float), NULL, NULL);
    d_initial_g[1] = clCreateBuffer(fir_lattice_context, CL_MEM_READ_WRITE, (coeff_count + 1) * sizeof(float), NULL, NULL);
    CLOCK_FINISH(timing.create_buffer)

    // Write our data set into the input array in device memory
    err = CL_SUCCESS;
    CLOCK_START()

    float **h_src = (float **)malloc(iteration * sizeof(float *));
    float **h_dst = (float **)malloc(iteration * sizeof(float *));
    for (int itr = 0; itr < iteration - 1; itr++) {
        h_src[itr] = (float *)clEnqueueMapBuffer(fir_lattice_queue, d_src[itr], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, BLOCK_SIZE * sizeof(float), 0, NULL, NULL, &err);
        h_dst[itr] = (float *)clEnqueueMapBuffer(fir_lattice_queue, d_dst[itr], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, BLOCK_SIZE * sizeof(float), 0, NULL, NULL, &err);
    }
    h_src[iteration - 1] = (float *)clEnqueueMapBuffer(fir_lattice_queue, d_src[iteration - 1], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, last_size * sizeof(float), 0, NULL, NULL, &err);
    h_dst[iteration - 1] = (float *)clEnqueueMapBuffer(fir_lattice_queue, d_dst[iteration - 1], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, last_size * sizeof(float), 0, NULL, NULL, &err);

    float *h_coeff = (float *)clEnqueueMapBuffer(fir_lattice_queue, d_coeff, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, coeff_size, 0, NULL, NULL, &err);

    float zero = 0.00;
    err = clEnqueueFillBuffer(fir_lattice_queue, d_initial_g[0], &zero, sizeof(float), 0, (coeff_count + 1) * sizeof(float), 0, NULL, NULL);
    clFinish(fir_lattice_queue);
    CLOCK_FINISH(timing.map_buffer)

    printErrorString(-2, err);

    err = CL_SUCCESS;
    int block_size = BLOCK_SIZE;
    for (int itr = 0; itr < iteration - 1; itr++) {
        err |= clSetKernelArg(fir_lattice_kernels[itr], 0, sizeof(int), &block_size);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 1, sizeof(int), &coeff_count);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 2, sizeof(cl_mem), &d_src[itr]);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 3, sizeof(cl_mem), &d_coeff);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 4, sizeof(cl_mem), &d_dst[itr]);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 5, sizeof(cl_mem), &d_initial_g[itr % 2]);
        err |= clSetKernelArg(fir_lattice_kernels[itr], 6, sizeof(cl_mem), &d_initial_g[(itr + 1) % 2]);
    }
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 0, sizeof(int), &last_size);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 1, sizeof(int), &coeff_count);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 2, sizeof(cl_mem), &d_src[iteration - 1]);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 3, sizeof(cl_mem), &d_coeff);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 4, sizeof(cl_mem), &d_dst[iteration - 1]);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 5, sizeof(cl_mem), &d_initial_g[(iteration - 1) % 2]);
    err |= clSetKernelArg(fir_lattice_kernels[iteration - 1], 6, sizeof(cl_mem), &d_initial_g[iteration % 2]);
    printErrorString(0, err);

    CLOCK_START()
    for (int itr = 0; itr < iteration - 1; itr++) {
        memcpy(h_src[itr], src + itr * BLOCK_SIZE, BLOCK_SIZE * sizeof(float));
    }
    memcpy(h_src[iteration - 1], src + (iteration - 1) * BLOCK_SIZE, last_size * sizeof(float));
    memcpy(h_coeff, coeff, coeff_size);
    CLOCK_FINISH(timing.memcpy)

    // Dummy kernel launch
    err = CL_SUCCESS;
    for (int itr = 0; itr < iteration - 1; itr++) {
        err |= clEnqueueNDRangeKernel(fir_lattice_queue, fir_lattice_kernels[itr], dimention, NULL, fm_global_item_size, fm_local_item_size, 0, NULL, NULL);
    }
    err |= clEnqueueNDRangeKernel(fir_lattice_queue, fir_lattice_kernels[iteration - 1], dimention, NULL, l_global_item_size, l_local_item_size, 0, NULL, NULL);
    printErrorString(2, err);
    // Wait for the command fir_lattice_queue to get serviced before reading back results
    clFinish(fir_lattice_queue);

    // Execute the kernel over the entire range of the data set
    CLOCK_START()
    err = clEnqueueNDRangeKernel(fir_lattice_queue, fir_lattice_kernels[0], dimention, NULL, fm_global_item_size, fm_local_item_size, 0, NULL, &event1);
    for (int itr = 1; itr < iteration - 1; itr++) {
        err |= clEnqueueNDRangeKernel(fir_lattice_queue, fir_lattice_kernels[itr], dimention, NULL, fm_global_item_size, fm_local_item_size, 0, NULL, NULL);
    }
    err |= clEnqueueNDRangeKernel(fir_lattice_queue, fir_lattice_kernels[iteration - 1], dimention, NULL, l_global_item_size, l_local_item_size, 0, NULL, &event2);
    clFinish(fir_lattice_queue);
    cl_ulong time_submit;
    cl_ulong time_start;
    cl_ulong time_end;
    err = clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &time_submit, NULL);
    err |= clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    err |= clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    printErrorString(2, err);
    timing.kernel_execute = (double)(time_end - time_start) * 1.0e-9f;
    timing.kernel_launch = (double)(time_start - time_submit) * 1.0e-9f;

    CLOCK_START()
    for (int itr = 0; itr < iteration; itr++) {
        clEnqueueUnmapMemObject(fir_lattice_queue, d_src[itr], h_src, 0, NULL, NULL);
        clEnqueueUnmapMemObject(fir_lattice_queue, d_dst[itr], h_dst, 0, NULL, NULL);
    }
    clEnqueueUnmapMemObject(fir_lattice_queue, d_coeff, h_coeff, 0, NULL, NULL);
    CLOCK_FINISH(timing.map_buffer)

    // Read the results from the device
    CLOCK_START()
    for (int itr = 0; itr < iteration - 1; itr++) {
        memcpy(dst + itr * BLOCK_SIZE, h_dst[itr], BLOCK_SIZE * sizeof(float));
    }
    memcpy(dst + (iteration - 1) * BLOCK_SIZE, h_dst[iteration - 1], last_size * sizeof(float));
    CLOCK_FINISH(timing.memcpy)

    // release OpenCL resources
    for (int itr = 0; itr < iteration; itr++) {
        clReleaseMemObject(d_src[itr]);
        clReleaseMemObject(d_dst[itr]);
    }
    clReleaseMemObject(d_coeff);
    clReleaseMemObject(d_initial_g[0]);
    clReleaseMemObject(d_initial_g[1]);

    return timing;
}