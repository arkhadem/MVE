#include "csum.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int csum_init(size_t cache_size,
              int LANE_NUM,
              config_t *&config,
              input_t **&input,
              output_t **&output) {

    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t **csum_input = (csum_input_t **)input;
    csum_output_t **csum_output = (csum_output_t **)output;

    // configuration
    init_1D<csum_config_t>(1, csum_config);
    csum_config->count = 64;
    int n = csum_config->count * BLOCK_16K;
    int count = cache_size / (n * sizeof(int32_t) + csum_config->count * sizeof(int16_t)) + 1;

    // initializing in/output
    init_1D<csum_input_t *>(count, csum_input);
    init_1D<csum_output_t *>(count, csum_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<csum_input_t>(1, csum_input[i]);
        init_1D<csum_output_t>(1, csum_output[i]);

        random_init_1D<int32_t>(n, csum_input[i]->ptr);
        random_init_1D<int16_t>(csum_config->count, csum_output[i]->sum);
    }

    config = (config_t *)csum_config;
    input = (input_t **)csum_input;
    output = (output_t **)csum_output;

    return count;
}