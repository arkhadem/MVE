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

    benchmark_runner(library.c_str(), kernel.c_str(), rounds, execute, lane_num);
    return 0;
}