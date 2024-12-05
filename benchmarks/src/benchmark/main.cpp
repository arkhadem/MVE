#include "benchmark.hpp"
#include <cstdlib>
#include <string.h>

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
    char *library;
    char *kernel;
    int rounds = -1;
    bool execute = true;
    int LANE_NUM = -1;
    if (argc < 4) {
        printf("missing argument: <library> <kernel> <LANE_NUM> [ <rounds [<execute=false|true>] ]\n");
        exit(-1);
    } else {
        library = argv[1];
        kernel = argv[2];
        LANE_NUM = atoi(argv[3]);
        if (argc > 4) {
            rounds = atoi(argv[4]);
            if (argc > 5) {
                if (argc == 6) {
                    if (strcmp(argv[5], "true") == 0) {
                        execute = true;
                    } else if (strcmp(argv[5], "false") == 0) {
                        execute = false;
                    } else {
                        printf("Error: execute must be either \"true\" or \"false\". Unrecognized \"%s\"\n", argv[4]);
                        exit(-1);
                    }
                } else {
                    printf("Error: at most 6 arguments are acceptable.\n");
                    exit(-1);
                }
            }
        }
    }
    benchmark_runner(library, kernel, rounds, execute, LANE_NUM);
    return 0;
}