#include "benchmark.hpp"
#include <cstdlib>
#include <string.h>

#include "argparse.hpp"
#include <iostream>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string_view>

int XNNPACK_M;
int XNNPACK_N;
int XNNPACK_K;

long *pollute_cache(size_t size) {
    const size_t bigger_than_cachesize = size / sizeof(long);
    long *p = (long *)malloc(size);
    for (int i = 0; i < bigger_than_cachesize; i++) {
        p[i] = i;
    }
    return p;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program(argv[0]);

    program.add_argument("-l", "--library")
        .help("Name of the library. Use --list for a list of libraries and kernels")
        .metavar("LIBRARY");

    program.add_argument("-k", "--kernel")
        .help("Name of the kernel. Use --list for a list of libraries and kernels")
        .metavar("KERNEL");

    program.add_argument("-n", "--lane_num")
        .help("Number of SIMD lanes for MVE or RVV simulations")
        .default_value(-1)
        .scan<'i', int>()
        .metavar("LANE_NUM");

    program.add_argument("-r", "--rounds")
        .help("Number of execution round. If not specified, the execution will repeat for 1 second")
        .default_value(-1)
        .scan<'i', int>()
        .metavar("ROUNDS");

    program.add_argument("-e", "--execute")
        .help("When specifying this option, kernels will not be invoked (used for energy measurements)")
        .default_value(true)
        .implicit_value(true);

    program.add_argument("-xm", "--xnnpack_m")
        .help("M dimension for the XNNPACK's GEMM and SPMM kernel")
        .default_value(-1)
        .scan<'i', int>()
        .metavar("XM");

    program.add_argument("-xn", "--xnnpack_n")
        .help("N dimension for the XNNPACK's GEMM and SPMM kernel")
        .default_value(-1)
        .scan<'i', int>()
        .metavar("XN");

    program.add_argument("-xk", "--xnnpack_k")
        .help("K dimension for the XNNPACK's GEMM and SPMM kernel")
        .default_value(-1)
        .scan<'i', int>()
        .metavar("XK");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string library, kernel;
    if ((program.present("--library") == std::nullopt) ||
        (program.present("--kernel") == std::nullopt)) {
        printf("Error: library and kernel must be provided!\n");
        printf("%s\n", program.help().str().c_str());
        exit(-1);
    }
    library = program.get<std::string>("--library").c_str();
    kernel = program.get<std::string>("--kernel").c_str();
    int lane_num = program.get<int>("--lane_num");
    int rounds = program.get<int>("--rounds");
    bool execute = (program["--execute"] == true);

    XNNPACK_M = program.get<int>("--xnnpack_m");
    XNNPACK_N = program.get<int>("--xnnpack_n");
    XNNPACK_K = program.get<int>("--xnnpack_k");
    if (library == "xnnpack") {
        if (XNNPACK_M == -1 || XNNPACK_N == -1 || XNNPACK_K == -1) {
            printf("Error: M, N, and K must be provided for XNNPACK!\n");
            exit(-1);
        }
    } else {
        if (XNNPACK_M != -1 || XNNPACK_N != -1 || XNNPACK_K != -1) {
            printf("Error: M, N, and K are only applicable for XNNPACK!\n");
            exit(-1);
        }
    }

    benchmark_runner(library.c_str(), kernel.c_str(), rounds, execute, lane_num);
    return 0;
}