#include "convolve_vertically.hpp"

#include "skia.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int convolve_vertically_init(size_t cache_size,
                             int LANE_NUM,
                             config_t *&config,
                             input_t **&input,
                             output_t **&output) {

    convolve_vertically_config_t *convolve_vertically_config = (convolve_vertically_config_t *)config;
    convolve_vertically_input_t **convolve_vertically_input = (convolve_vertically_input_t **)input;
    convolve_vertically_output_t **convolve_vertically_output = (convolve_vertically_output_t **)output;

    // configuration
    int rows = 2;
    int columns = 1024;
    int filter_length = 32;

    init_1D<convolve_vertically_config_t>(1, convolve_vertically_config);
    convolve_vertically_config->num_rows = rows;
    convolve_vertically_config->num_cols = columns;
    convolve_vertically_config->filter_length = filter_length;
    convolve_vertically_config->shift_value = 2;

    // in/output versions
    size_t input_size = (rows + filter_length) * (columns * 4) * sizeof(uint8_t);
    input_size += filter_length * sizeof(int16_t);
    size_t output_size = (rows) * (4 * columns) * sizeof(uint8_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<convolve_vertically_input_t *>(count, convolve_vertically_input);
    init_1D<convolve_vertically_output_t *>(count, convolve_vertically_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<convolve_vertically_input_t>(1, convolve_vertically_input[i]);
        init_1D<convolve_vertically_output_t>(1, convolve_vertically_output[i]);

        random_init_2D<uint8_t>(rows + filter_length, columns * 4, convolve_vertically_input[i]->src_data);
        random_init_1D<int16_t>(filter_length, convolve_vertically_input[i]->filter_values);
        random_init_2D<uint8_t>(rows, 4 * columns, convolve_vertically_output[i]->out_row);
    }

    config = (config_t *)convolve_vertically_config;
    input = (input_t **)convolve_vertically_input;
    output = (output_t **)convolve_vertically_output;

    return count;
}