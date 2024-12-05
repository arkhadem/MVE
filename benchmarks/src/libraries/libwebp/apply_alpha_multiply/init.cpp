#include "apply_alpha_multiply.hpp"

#include "libwebp.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int apply_alpha_multiply_init(size_t cache_size,
                              int LANE_NUM,
                              config_t *&config,
                              input_t **&input,
                              output_t **&output) {

    apply_alpha_multiply_config_t *apply_alpha_multiply_config = (apply_alpha_multiply_config_t *)config;
    apply_alpha_multiply_input_t **apply_alpha_multiply_input = (apply_alpha_multiply_input_t **)input;
    apply_alpha_multiply_output_t **apply_alpha_multiply_output = (apply_alpha_multiply_output_t **)output;

    // configuration
    int rows = 16;
    int columns = 1024;

    init_1D<apply_alpha_multiply_config_t>(1, apply_alpha_multiply_config);
    apply_alpha_multiply_config->num_rows = rows;
    apply_alpha_multiply_config->num_cols = columns;

    // in/output versions
    size_t input_size = 4 * (rows * columns) * sizeof(uint8_t);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<apply_alpha_multiply_input_t *>(count, apply_alpha_multiply_input);
    init_1D<apply_alpha_multiply_output_t *>(count, apply_alpha_multiply_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<apply_alpha_multiply_input_t>(1, apply_alpha_multiply_input[i]);
        init_1D<apply_alpha_multiply_output_t>(1, apply_alpha_multiply_output[i]);

        random_init_1D<uint8_t>(4 * rows * columns, apply_alpha_multiply_input[i]->rgba);
    }

    config = (config_t *)apply_alpha_multiply_config;
    input = (input_t **)apply_alpha_multiply_input;
    output = (output_t **)apply_alpha_multiply_output;

    return count;
}