#include <stdlib.h>

#include "gradient_filter.hpp"

#include "libwebp.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int gradient_filter_init(size_t cache_size,
                         int LANE_NUM,
                         config_t *&config,
                         input_t **&input,
                         output_t **&output) {

    gradient_filter_config_t *gradient_filter_config = (gradient_filter_config_t *)config;
    gradient_filter_input_t **gradient_filter_input = (gradient_filter_input_t **)input;
    gradient_filter_output_t **gradient_filter_output = (gradient_filter_output_t **)output;

    // configuration
    int rows = 64;
    int columns = 1024;

    init_1D<gradient_filter_config_t>(1, gradient_filter_config);
    gradient_filter_config->num_rows = rows;
    gradient_filter_config->num_cols = columns;
    gradient_filter_config->stride = (columns + 1);

    // in/output versions
    size_t input_size = ((rows + 1) * (columns + 1)) * sizeof(uint8_t);
    size_t output_size = (rows * columns) * sizeof(uint8_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<gradient_filter_input_t *>(count, gradient_filter_input);
    init_1D<gradient_filter_output_t *>(count, gradient_filter_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<gradient_filter_input_t>(1, gradient_filter_input[i]);
        init_1D<gradient_filter_output_t>(1, gradient_filter_output[i]);

        random_init_1D<uint8_t>((rows + 1) * (columns + 1), gradient_filter_input[i]->in);
        random_init_1D<uint8_t>(rows * columns, gradient_filter_output[i]->out);
    }

    config = (config_t *)gradient_filter_config;
    input = (input_t **)gradient_filter_input;
    output = (output_t **)gradient_filter_output;

    return count;
}