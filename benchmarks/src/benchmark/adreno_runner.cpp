#include "benchmark.hpp"

#include "init.hpp"
#include "runner.hpp"
#include <cstddef>
#include <cstdio>

void register_kernels();

void benchmark_runner(char *library, char *kernel, int rounds, bool execute, int LANE_NUM) {
    register_inits();
    register_kernels();
    std::string lib_str = std::string(library);
    std::string ker_str = std::string(kernel);
    if (init_functions.find(lib_str) == init_functions.end()) {
        printf("Error: library \"%s\" does not exist in inits!\n", library);
        exit(-1);
    }
    if (init_functions[lib_str].find(ker_str) == init_functions[lib_str].end()) {
        printf("Error: kernel \"%s\" does not exist in inits!\n", kernel);
        exit(-1);
    }
    if (adreno_kernel_functions.find(lib_str) == adreno_kernel_functions.end()) {
        printf("Error: library \"%s\" does not exist in kernels!\n", library);
        exit(-1);
    }
    if (adreno_kernel_functions[lib_str].find(ker_str) == adreno_kernel_functions[lib_str].end()) {
        printf("Error: kernel \"%s\" does not exist in kernels!\n", kernel);
        exit(-1);
    }
    initfunc init_func = init_functions[lib_str][ker_str];
    adrenokernelfunc kernel_func = adreno_kernel_functions[lib_str][ker_str];
    config_s *config = nullptr;
    input_s **input = nullptr;
    output_s **output = nullptr;

    int count = init_func(sizeof(long) * CACHE_SIZE, 32, config, input, output);

    long *tmp = pollute_cache(sizeof(long) * CACHE_SIZE);

    std::printf("Cache polluted (%ld) | Count (%d) | LANE_NUM(%d) | Dummy (%d) | Dummy (%d)\n", tmp[rand() % CACHE_SIZE], count, LANE_NUM, input[rand() % count]->dummy[0], output[rand() % count]->dummy[0]);

    int idx = 0;
    int iterations = 0;
    // in us
    timing_t timing;
    cl_event event;
    CLOCK_INIT(timing)
    while (rounds != 0) {

        iterations++;

        timing_t temp_timing;
        CLOCK_INIT(temp_timing)
        if (execute)
            temp_timing = kernel_func(config, input[idx], output[idx]);
        CLOCK_ACC(timing, temp_timing)

        if (timing.kernel_execute >= 1.00) {
            if (rounds < 0)
                break;
        }

        idx++;
        rounds--;
        idx %= count;
    }
    printf("Successfully finished run in experiment mode!\n");
    printf("iterations: %d (number of processing whole domain input)\n", iterations);
    printf("total_time: %lf usec (total execution time of all iterations)\n",
           timing.create_buffer + timing.map_buffer + timing.memcpy + timing.kernel_launch + timing.kernel_execute);
    CLOCK_DIV(timing, iterations)
    printf("iteration_time: %lf usec (execution time of one iterations)\n",
           timing.create_buffer + timing.map_buffer + timing.memcpy + timing.kernel_launch + timing.kernel_execute);
    printf("create_buffer_time: %lf usec (execution time of clCreateBuffer APIs)\n", timing.create_buffer);
    printf("map_buffer_time: %lf usec (execution time of clEnqueueMapBuffer and clEnqueueUnmapMemObject APIs)\n", timing.map_buffer);
    printf("memcpy_time: %lf usec (execution time of memcpy APIs)\n", timing.memcpy);
    printf("kernel_launch_time: %lf usec (execution time of clEnqueueNDRangeKernel APIs)\n", timing.kernel_launch);
    printf("kernel_execute_time: %lf usec (kernel time)\n", timing.kernel_execute);
}