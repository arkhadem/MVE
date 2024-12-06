#include "benchmark.hpp"

#include "init.hpp"
#include "mve.hpp"
#include "runner.hpp"
#include <cassert>
#include <cstddef>
#include <cstdio>

void register_kernels();

void benchmark_runner(const char *library, const char *kernel, int rounds, bool execute, int LANE_NUM) {
    register_inits();
    register_kernels();
    assert(rounds == -1);
    assert(execute == true);
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
    kernelfunc mve_func = kernel_functions[lib_str][ker_str];
    config_s *config = nullptr;
    input_s **input = nullptr;
    output_s **output = nullptr;
    init_func(0, LANE_NUM, config, input, output);
    char graph_name[100];
    sprintf(graph_name, "%s_%s_%d", library, kernel, LANE_NUM);
    mve_initializer(graph_name, LANE_NUM);
    mve_func(LANE_NUM, config, input[0], output[0]);
    mve_finisher();
}