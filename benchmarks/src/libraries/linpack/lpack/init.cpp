#include "lpack.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int lpack_init(size_t cache_size,
               int LANE_NUM,
               config_t *&config,
               input_t **&input,
               output_t **&output) {

    lpack_config_t *lpack_config = (lpack_config_t *)config;
    lpack_input_t **lpack_input = (lpack_input_t **)input;
    lpack_output_t **lpack_output = (lpack_output_t **)output;

    // configuration
    init_1D<lpack_config_t>(1, lpack_config);
    lpack_config->n = 512 * 1024 * 1024;
    int count = cache_size / (lpack_config->n * 3 * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<lpack_input_t *>(count, lpack_input);
    init_1D<lpack_output_t *>(count, lpack_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<lpack_input_t>(1, lpack_input[i]);
        init_1D<lpack_output_t>(1, lpack_output[i]);

        random_init_1D<int32_t>(1, lpack_input[i]->da);
        random_init_1D<int32_t>(lpack_config->n * 64, lpack_input[i]->dx);
        random_init_1D<int32_t>(lpack_config->n * 64, lpack_input[i]->dyin);
        random_init_1D<int32_t>(lpack_config->n * 64, lpack_output[i]->dyout);
    }

    config = (config_t *)lpack_config;
    input = (input_t **)lpack_input;
    output = (output_t **)lpack_output;

    return count;
}