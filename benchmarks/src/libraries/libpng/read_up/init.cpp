#include "read_up.hpp"

#include "libpng.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int read_up_init(size_t cache_size,
                 int LANE_NUM,
                 config_t *&config,
                 input_t **&input,
                 output_t **&output) {

    read_up_config_t *read_up_config = (read_up_config_t *)config;
    read_up_input_t **read_up_input = (read_up_input_t **)input;
    read_up_output_t **read_up_output = (read_up_output_t **)output;

    // configuration
    int rows = 32;
    int columns = 2048;

    init_1D<read_up_config_t>(1, read_up_config);
    read_up_config->num_rows = rows;
    read_up_config->num_cols = columns;

    // in/output versions
    size_t input_size = 2 * (rows * columns) * sizeof(png_byte);
    size_t output_size = (rows * columns) * sizeof(png_byte);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<read_up_input_t *>(count, read_up_input);
    init_1D<read_up_output_t *>(count, read_up_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<read_up_input_t>(1, read_up_input[i]);
        init_1D<read_up_output_t>(1, read_up_output[i]);

        random_init_2D<png_byte>(rows, columns, read_up_input[i]->input_buf);
        random_init_2D<png_byte>(rows, columns, read_up_input[i]->prev_input_buf);
        random_init_2D<png_byte>(rows, columns, read_up_output[i]->output_buf);
    }

    config = (config_t *)read_up_config;
    input = (input_t **)read_up_input;
    output = (output_t **)read_up_output;

    return count;
}