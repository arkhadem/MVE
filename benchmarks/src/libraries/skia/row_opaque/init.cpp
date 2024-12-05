#include "row_opaque.hpp"

#include "skia.hpp"

#include "benchmark.hpp"

#include "init.hpp"
#include <cstdint>

int row_opaque_init(size_t cache_size,
                    int LANE_NUM,
                    config_t *&config,
                    input_t **&input,
                    output_t **&output) {

    row_opaque_config_t *row_opaque_config = (row_opaque_config_t *)config;
    row_opaque_input_t **row_opaque_input = (row_opaque_input_t **)input;
    row_opaque_output_t **row_opaque_output = (row_opaque_output_t **)output;

    // configuration
    int rows = 4;
    int columns = 1024;

    init_1D<row_opaque_config_t>(1, row_opaque_config);
    row_opaque_config->num_rows = rows;
    row_opaque_config->num_cols = columns;

    // in/output versions
    size_t input_size = rows * columns * sizeof(uint32_t);
    input_size += 8 * sizeof(uint32_t);
    size_t output_size = rows * columns * sizeof(uint32_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<row_opaque_input_t *>(count, row_opaque_input);
    init_1D<row_opaque_output_t *>(count, row_opaque_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<row_opaque_input_t>(1, row_opaque_input[i]);
        init_1D<row_opaque_output_t>(1, row_opaque_output[i]);

        random_init_1D<uint16_t>(rows * columns, row_opaque_input[i]->src);
        random_init_1D<uint32_t>(4, row_opaque_input[i]->color);
        random_init_1D<uint32_t>(4, row_opaque_output[i]->opaque_dst);
        random_init_1D<uint32_t>(rows * columns, row_opaque_output[i]->dst);
    }

    config = (config_t *)row_opaque_config;
    input = (input_t **)row_opaque_input;
    output = (output_t **)row_opaque_output;

    return count;
}