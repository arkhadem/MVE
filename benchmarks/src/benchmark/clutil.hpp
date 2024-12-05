#ifndef B6B793D2_8E53_4BDA_B5D1_A13E1ECD5C6B
#define B6B793D2_8E53_4BDA_B5D1_A13E1ECD5C6B

#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

void printErrorString(int pointer, cl_int error);

typedef struct timing_t {
    double create_buffer;
    double map_buffer;
    double memcpy;
    double kernel_launch;
    double kernel_execute;
} timing_t;

#define CLOCK_START() \
    start = clock();

#define CLOCK_FINISH(FIELD)                              \
    do {                                                 \
        end = clock();                                   \
        double clock_spent = end - start;                \
        FIELD += ((double)clock_spent) / CLOCKS_PER_SEC; \
    } while (0);

#define CLOCK_INIT(TIMING)     \
    TIMING.create_buffer = 0;  \
    TIMING.map_buffer = 0;     \
    TIMING.kernel_launch = 0;  \
    TIMING.kernel_execute = 0; \
    TIMING.memcpy = 0;

#define CLOCK_ACC(DST, SRC)                   \
    DST.create_buffer += SRC.create_buffer;   \
    DST.map_buffer += SRC.map_buffer;         \
    DST.kernel_launch += SRC.kernel_launch;   \
    DST.kernel_execute += SRC.kernel_execute; \
    DST.memcpy += SRC.memcpy;

#define CLOCK_DIV(DST, SRC)            \
    DST.create_buffer /= (double)SRC;  \
    DST.map_buffer /= (double)SRC;     \
    DST.kernel_launch /= (double)SRC;  \
    DST.kernel_execute /= (double)SRC; \
    DST.memcpy /= (double)SRC;

#define PROF_FINISH(QUEUE)                                                                                   \
    clFinish(QUEUE);                                                                                         \
    cl_ulong time_submit;                                                                                    \
    cl_ulong time_start;                                                                                     \
    cl_ulong time_end;                                                                                       \
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &time_submit, NULL); \
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);  \
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);      \
    printErrorString(2, err);                                                                                \
    timing.kernel_execute = (double)(time_end - time_start) * 1.0e-9f;                                       \
    timing.kernel_launch = (double)(time_start - time_submit) * 1.0e-9f;

#define PROF_PRINT()                                                                                                                                                                                                             \
    printf("Total (create buffer: %lf), (map buffer: %lf), (mem cpy: %lf), (kernel launch: %lf), (kernel execute: %lf)\n", timing.create_buffer, timing.map_buffer, timing.memcpy, timing.kernel_launch, timing.kernel_execute); \
    CLOCK_DIV(timing, iterations)                                                                                                                                                                                                \
    printf("Individual (create buffer: %lf), (map buffer: %lf), (mem cpy: %lf), (kernel launch: %lf), (kernel execute: %lf)\n", timing.create_buffer, timing.map_buffer, timing.memcpy, timing.kernel_launch, timing.kernel_execute);

#endif /* B6B793D2_8E53_4BDA_B5D1_A13E1ECD5C6B */
