#include "convolve_horizontally.hpp"

#include "skia.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int convolve_horizontally_init(size_t cache_size,
                               int LANE_NUM,
                               config_t *&config,
                               input_t **&input,
                               output_t **&output) {

    convolve_horizontally_config_t *convolve_horizontally_config = (convolve_horizontally_config_t *)config;
    convolve_horizontally_input_t **convolve_horizontally_input = (convolve_horizontally_input_t **)input;
    convolve_horizontally_output_t **convolve_horizontally_output = (convolve_horizontally_output_t **)output;

    // configuration
    int rows = 1024;
    int columns = 2;
    int filter_length = 32;

    init_1D<convolve_horizontally_config_t>(1, convolve_horizontally_config);
    convolve_horizontally_config->num_rows = rows;
    convolve_horizontally_config->num_cols = columns;
    convolve_horizontally_config->filter_length = filter_length;
    convolve_horizontally_config->shift_value = 2;

    // in/output versions
    size_t input_size = 4 * rows * (columns + filter_length) * sizeof(uint8_t);
    input_size += filter_length * sizeof(int16_t);
    size_t output_size = 4 * rows * columns * sizeof(uint8_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<convolve_horizontally_input_t *>(count, convolve_horizontally_input);
    init_1D<convolve_horizontally_output_t *>(count, convolve_horizontally_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<convolve_horizontally_input_t>(1, convolve_horizontally_input[i]);
        init_1D<convolve_horizontally_output_t>(1, convolve_horizontally_output[i]);

        random_init_1D<uint8_t>(4 * rows * (columns + filter_length), convolve_horizontally_input[i]->src_data);
        random_init_1D<int16_t>(filter_length, convolve_horizontally_input[i]->filter_values);
        random_init_1D<uint8_t>(4 * rows * columns, convolve_horizontally_output[i]->out_row);
    }

    config = (config_t *)convolve_horizontally_config;
    input = (input_t **)convolve_horizontally_input;
    output = (output_t **)convolve_horizontally_output;

    return count;
}