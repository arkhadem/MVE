#include "rgb_to_gray.hpp"

#include "libjpeg.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int rgb_to_gray_init(size_t cache_size,
                     int LANE_NUM,
                     config_t *&config,
                     input_t **&input,
                     output_t **&output) {

    rgb_to_gray_config_t *rgb_to_gray_config = (rgb_to_gray_config_t *)config;
    rgb_to_gray_input_t **rgb_to_gray_input = (rgb_to_gray_input_t **)input;
    rgb_to_gray_output_t **rgb_to_gray_output = (rgb_to_gray_output_t **)output;

    // configuration
    init_1D<rgb_to_gray_config_t>(1, rgb_to_gray_config);
    rgb_to_gray_config->num_rows = 16;
    rgb_to_gray_config->num_cols = 1024;

    // in/output versions
    size_t input_size = (rgb_to_gray_config->num_rows * rgb_to_gray_config->num_cols * RGB_PIXELSIZE) * sizeof(JSAMPLE);
    size_t output_size = (rgb_to_gray_config->num_rows * rgb_to_gray_config->num_cols) * sizeof(JSAMPLE);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<rgb_to_gray_input_t *>(count, rgb_to_gray_input);
    init_1D<rgb_to_gray_output_t *>(count, rgb_to_gray_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<rgb_to_gray_input_t>(1, rgb_to_gray_input[i]);
        init_1D<rgb_to_gray_output_t>(1, rgb_to_gray_output[i]);

        random_init_2D<JSAMPLE>(rgb_to_gray_config->num_rows, rgb_to_gray_config->num_cols * RGB_PIXELSIZE, rgb_to_gray_input[i]->input_buf);
        random_init_2D<JSAMPLE>(rgb_to_gray_config->num_rows, rgb_to_gray_config->num_cols, rgb_to_gray_output[i]->output_buf);
    }

    config = (config_t *)rgb_to_gray_config;
    input = (input_t **)rgb_to_gray_input;
    output = (output_t **)rgb_to_gray_output;

    return count;
}