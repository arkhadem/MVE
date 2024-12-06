#include "benchmark.hpp"

#include "init.hpp"
#include "runner.hpp"
#include <cstddef>
#include <cstdio>

void register_kernels();

void benchmark_runner(const char *library, const char *kernel, int rounds, bool execute, int LANE_NUM) {
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
    if (kernel_functions.find(lib_str) == kernel_functions.end()) {
        printf("Error: library \"%s\" does not exist in kernels!\n", library);
        exit(-1);
    }
    if (kernel_functions[lib_str].find(ker_str) == kernel_functions[lib_str].end()) {
        printf("Error: kernel \"%s\" does not exist in kernels!\n", kernel);
        exit(-1);
    }
    initfunc init_func = init_functions[lib_str][ker_str];
    kernelfunc kernel_func = kernel_functions[lib_str][ker_str];
    config_s *config = nullptr;
    input_s **input = nullptr;
    output_s **output = nullptr;

    int count = init_func(sizeof(long) * CACHE_SIZE, 32, config, input, output);

    long *tmp = pollute_cache(sizeof(long) * CACHE_SIZE);

    std::printf("Cache polluted (%ld) | Count (%d) | LANE_NUM(%d) | Dummy (%d) | Dummy (%d)\n", tmp[rand() % CACHE_SIZE], count, LANE_NUM, input[rand() % count]->dummy[0], output[rand() % count]->dummy[0]);

    int idx = 0;
    int iterations = 0;
    // in us
    double time_spent = 0.0000;
    while (rounds != 0) {

        iterations++;

        clock_t start = clock();
        if (execute)
            kernel_func(LANE_NUM, config, input[idx], output[idx]);
        clock_t end = clock();

        clock_t clock_spent = end - start;
        time_spent += ((double)clock_spent) * 1000000.0000 / CLOCKS_PER_SEC;

        if (time_spent >= 1000000.0000) {
            if (rounds < 0)
                break;
        }

        idx++;
        rounds--;
        idx %= count;
    }
    printf("Successfully finished run in experiment mode!\n");
    printf("iterations: %d (number of processing whole domain input)\n", iterations);
    printf("total_time: %lf usec (total execution time of all iterations)\n", time_spent);
    printf("iteration_time: %lf usec (execution time of one iterations)\n", time_spent / (double)iterations);
}