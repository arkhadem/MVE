#include "memset.hpp"

#include "optroutines.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int memset_init(size_t cache_size,
                int LANE_NUM,
                config_t *&config,
                input_t **&input,
                output_t **&output) {

    memset_config_t *memset_config = (memset_config_t *)config;
    memset_input_t **memset_input = (memset_input_t **)input;
    memset_output_t **memset_output = (memset_output_t **)output;

    // configuration
    int size = 65536;

    init_1D<memset_config_t>(1, memset_config);
    memset_config->size = size;

    // in/output versions
    size_t input_size = 0;
    size_t output_size = size * sizeof(char);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<memset_input_t *>(count, memset_input);
    init_1D<memset_output_t *>(count, memset_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<memset_input_t>(1, memset_input[i]);
        init_1D<memset_output_t>(1, memset_output[i]);

        random_init_1D<char>(1, memset_input[i]->value);
        init_1D<char>(size, memset_output[i]->dst);
    }

    config = (config_t *)memset_config;
    input = (input_t **)memset_input;
    output = (output_t **)memset_output;

    return count;
}