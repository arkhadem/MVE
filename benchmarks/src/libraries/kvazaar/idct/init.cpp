#include "idct.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int idct_init(size_t cache_size,
              int LANE_NUM,
              config_t *&config,
              input_t **&input,
              output_t **&output) {

    idct_config_t *idct_config = (idct_config_t *)config;
    idct_input_t **idct_input = (idct_input_t **)input;
    idct_output_t **idct_output = (idct_output_t **)output;

    // configuration
    init_1D<idct_config_t>(1, idct_config);
    idct_config->count = 8192;
    init_1D<int8_t>(idct_config->count, idct_config->bitdepth);
    for (int i = 0; i < idct_config->count; i++) {
        idct_config->bitdepth[i] = 12;
    }
    int count = cache_size / (idct_config->count * 8 * 8 * 2 * sizeof(int16_t)) + 1;

    // initializing in/output
    init_1D<idct_input_t *>(count, idct_input);
    init_1D<idct_output_t *>(count, idct_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<idct_input_t>(1, idct_input[i]);
        init_1D<idct_output_t>(1, idct_output[i]);

        random_init_1D<int16_t>(idct_config->count * 8 * 8, idct_input[i]->input);
        random_init_1D<int16_t>(idct_config->count * 8 * 8, idct_output[i]->output);
    }

    config = (config_t *)idct_config;
    input = (input_t **)idct_input;
    output = (output_t **)idct_output;

    return count;
}