#include "satd.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int satd_init(size_t cache_size,
              int LANE_NUM,
              config_t *&config,
              input_t **&input,
              output_t **&output) {

    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t **satd_input = (satd_input_t **)input;
    satd_output_t **satd_output = (satd_output_t **)output;

    // configuration
    init_1D<satd_config_t>(1, satd_config);
    satd_config->count = 14400;
    int count = cache_size / (satd_config->count * 64 * 2 * sizeof(uint8_t) + satd_config->count * 64 * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<satd_input_t *>(count, satd_input);
    init_1D<satd_output_t *>(count, satd_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<satd_input_t>(1, satd_input[i]);
        init_1D<satd_output_t>(1, satd_output[i]);

        random_init_1D<uint8_t>(satd_config->count * 64, satd_input[i]->piOrg);
        random_init_1D<uint8_t>(satd_config->count * 64, satd_input[i]->piCur);
        random_init_1D<int32_t>(satd_config->count * 64, satd_output[i]->result);
    }

    config = (config_t *)satd_config;
    input = (input_t **)satd_input;
    output = (output_t **)satd_output;

    return count;
}