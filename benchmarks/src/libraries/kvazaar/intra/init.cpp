#include "intra.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int intra_init(size_t cache_size,
               int LANE_NUM,
               config_t *&config,
               input_t **&input,
               output_t **&output) {

    intra_config_t *intra_config = (intra_config_t *)config;
    intra_input_t **intra_input = (intra_input_t **)input;
    intra_output_t **intra_output = (intra_output_t **)output;

    // configuration
    init_1D<intra_config_t>(1, intra_config);
    intra_config->count = 8192;
    intra_config->log2_width = 3;
    intra_config->width = 1 << intra_config->log2_width;
    int input_count = intra_config->count * 17;
    if (input_count % 16) {
        input_count += (16 - (input_count % 16));
    }
    int output_count = intra_config->count;
    if (output_count % 16) {
        output_count += (16 - (output_count % 16));
    }
    int count = cache_size / ((input_count * 2 + output_count) * sizeof(kvz_pixel)) + 1;

    // initializing in/output
    init_1D<intra_input_t *>(count, intra_input);
    init_1D<intra_output_t *>(count, intra_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<intra_input_t>(1, intra_input[i]);
        init_1D<intra_output_t>(1, intra_output[i]);

        random_init_1D<kvz_pixel>(input_count, intra_input[i]->ref_top);
        random_init_1D<kvz_pixel>(input_count, intra_input[i]->ref_left);
        random_init_1D<kvz_pixel>(output_count, intra_output[i]->dst);
    }

    config = (config_t *)intra_config;
    input = (input_t **)intra_input;
    output = (output_t **)intra_output;

    return count;
}