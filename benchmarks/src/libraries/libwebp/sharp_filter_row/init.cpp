#include "sharp_filter_row.hpp"

#include "libwebp.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int sharp_filter_row_init(size_t cache_size,
                          int LANE_NUM,
                          config_t *&config,
                          input_t **&input,
                          output_t **&output) {

    sharp_filter_row_config_t *sharp_filter_row_config = (sharp_filter_row_config_t *)config;
    sharp_filter_row_input_t **sharp_filter_row_input = (sharp_filter_row_input_t **)input;
    sharp_filter_row_output_t **sharp_filter_row_output = (sharp_filter_row_output_t **)output;

    // configuration
    int rows = 16;
    int columns = 1024;

    init_1D<sharp_filter_row_config_t>(1, sharp_filter_row_config);
    sharp_filter_row_config->num_rows = rows;
    sharp_filter_row_config->num_cols = columns;
    sharp_filter_row_config->bit_depth = 4;

    // in/output versions
    size_t input_size = 2 * rows * (columns + 1) * sizeof(int16_t);
    input_size += 2 * rows * columns * sizeof(uint16_t);
    size_t output_size = 2 * rows * columns * sizeof(uint16_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<sharp_filter_row_input_t *>(count, sharp_filter_row_input);
    init_1D<sharp_filter_row_output_t *>(count, sharp_filter_row_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<sharp_filter_row_input_t>(1, sharp_filter_row_input[i]);
        init_1D<sharp_filter_row_output_t>(1, sharp_filter_row_output[i]);

        random_init_1D<int16_t>(rows * (columns + 1), sharp_filter_row_input[i]->A);
        random_init_1D<int16_t>(rows * (columns + 1), sharp_filter_row_input[i]->B);
        random_init_1D<uint16_t>(2 * rows * columns, sharp_filter_row_input[i]->best_y);
        random_init_1D<uint16_t>(2 * rows * columns, sharp_filter_row_output[i]->out);
    }

    config = (config_t *)sharp_filter_row_config;
    input = (input_t **)sharp_filter_row_input;
    output = (output_t **)sharp_filter_row_output;

    return count;
}