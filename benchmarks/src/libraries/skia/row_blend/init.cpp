#include "row_blend.hpp"

#include "skia.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int row_blend_init(size_t cache_size,
                   int LANE_NUM,
                   config_t *&config,
                   input_t **&input,
                   output_t **&output) {

    row_blend_config_t *row_blend_config = (row_blend_config_t *)config;
    row_blend_input_t **row_blend_input = (row_blend_input_t **)input;
    row_blend_output_t **row_blend_output = (row_blend_output_t **)output;

    // configuration
    int rows = 4;
    int columns = 1024;

    init_1D<row_blend_config_t>(1, row_blend_config);
    row_blend_config->num_rows = rows;
    row_blend_config->num_cols = columns;
    row_blend_config->alpha = 32;

    // in/output versions
    size_t input_size = rows * columns * sizeof(uint32_t);
    size_t output_size = rows * columns * sizeof(uint32_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<row_blend_input_t *>(count, row_blend_input);
    init_1D<row_blend_output_t *>(count, row_blend_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<row_blend_input_t>(1, row_blend_input[i]);
        init_1D<row_blend_output_t>(1, row_blend_output[i]);

        random_init_1D<uint32_t>(rows * columns, row_blend_input[i]->src);
        random_init_1D<uint32_t>(rows * columns, row_blend_output[i]->dst);
    }

    config = (config_t *)row_blend_config;
    input = (input_t **)row_blend_input;
    output = (output_t **)row_blend_output;

    return count;
}