#include "memcmp.hpp"

#include "optroutines.hpp"

#include "benchmark.hpp"

#include "init.hpp"
#include <cstring>

#include <stdio.h>
#include <string.h>

char *cmp_mve_const;

int memcmp_init(size_t cache_size,
                int LANE_NUM,
                config_t *&config,
                input_t **&input,
                output_t **&output) {

    memcmp_config_t *memcmp_config = (memcmp_config_t *)config;
    memcmp_input_t **memcmp_input = (memcmp_input_t **)input;
    memcmp_output_t **memcmp_output = (memcmp_output_t **)output;

    // configuration
    int size = 128 * 1024;

    init_1D<memcmp_config_t>(1, memcmp_config);
    memcmp_config->size = size;

    // in/output versions
    size_t input_size = 2 * size * sizeof(char);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<memcmp_input_t *>(count, memcmp_input);
    init_1D<memcmp_output_t *>(count, memcmp_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<memcmp_input_t>(1, memcmp_input[i]);
        init_1D<memcmp_output_t>(1, memcmp_output[i]);

        random_init_1D<char>(size, memcmp_input[i]->src1);
        init_1D<char>(size, memcmp_input[i]->src2);
        memcpy(memcmp_input[i]->src2, memcmp_input[i]->src1, size);
        init_1D<char *>(size, memcmp_input[i]->src1_addr);
        init_1D<char *>(size, memcmp_input[i]->src2_addr);
        for (int j = 0; j < 128; j++) {
            memcmp_input[i]->src1_addr[j] = memcmp_input[i]->src1 + rand_index[j];
            memcmp_input[i]->src2_addr[j] = memcmp_input[i]->src2 + rand_index[j];
        }
        memcmp_input[i]->src1[size - 1]++;
        init_1D<int>(1, memcmp_output[i]->return_val);
    }

    init_1D<char>(LANE_NUM * 256, cmp_mve_const);
    for (int i = 0; i < LANE_NUM * 2; i++) {
        for (int j = 0; j < 128; j++) {
            cmp_mve_const[i * 128 + j] = j;
        }
    }

    config = (config_t *)memcmp_config;
    input = (input_t **)memcmp_input;
    output = (output_t **)memcmp_output;

    return count;
}