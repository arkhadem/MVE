#include "dispatch_alpha.hpp"

#include "libwebp.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int dispatch_alpha_init(size_t cache_size,
                        int LANE_NUM,
                        config_t *&config,
                        input_t **&input,
                        output_t **&output) {

    dispatch_alpha_config_t *dispatch_alpha_config = (dispatch_alpha_config_t *)config;
    dispatch_alpha_input_t **dispatch_alpha_input = (dispatch_alpha_input_t **)input;
    dispatch_alpha_output_t **dispatch_alpha_output = (dispatch_alpha_output_t **)output;

    // configuration
    int rows = 16;
    int columns = 1024;

    init_1D<dispatch_alpha_config_t>(1, dispatch_alpha_config);
    dispatch_alpha_config->num_rows = rows;
    dispatch_alpha_config->num_cols = columns;

    // in/output versions
    size_t input_size = 4 * (rows * columns) * sizeof(uint8_t);
    size_t output_size = (rows * columns) * sizeof(uint8_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<dispatch_alpha_input_t *>(count, dispatch_alpha_input);
    init_1D<dispatch_alpha_output_t *>(count, dispatch_alpha_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<dispatch_alpha_input_t>(1, dispatch_alpha_input[i]);
        init_1D<dispatch_alpha_output_t>(1, dispatch_alpha_output[i]);

        random_init_1D<uint8_t>(rows * columns, dispatch_alpha_input[i]->alpha);
        random_init_1D<uint8_t>(4 * rows * columns, dispatch_alpha_output[i]->dst);
        random_init_1D<int>(1, dispatch_alpha_output[i]->return_val);
    }

    config = (config_t *)dispatch_alpha_config;
    input = (input_t **)dispatch_alpha_input;
    output = (output_t **)dispatch_alpha_output;

    return count;
}