#include "ycbcr_to_rgb.hpp"

#include "libjpeg.hpp"

#include "benchmark.hpp"

#include "init.hpp"

const int16_t ycbcr_to_rgb_const[4] = {-F_0_344, F_0_714, F_1_402, F_1_772};

int ycbcr_to_rgb_init(size_t cache_size,
                      int LANE_NUM,
                      config_t *&config,
                      input_t **&input,
                      output_t **&output) {

    ycbcr_to_rgb_config_t *ycbcr_to_rgb_config = (ycbcr_to_rgb_config_t *)config;
    ycbcr_to_rgb_input_t **ycbcr_to_rgb_input = (ycbcr_to_rgb_input_t **)input;
    ycbcr_to_rgb_output_t **ycbcr_to_rgb_output = (ycbcr_to_rgb_output_t **)output;

    // configuration
    init_1D<ycbcr_to_rgb_config_t>(1, ycbcr_to_rgb_config);
    ycbcr_to_rgb_config->num_rows = 16;
    ycbcr_to_rgb_config->num_cols = 1024;

    // in/output versions
    size_t input_size = ((RGB_PIXELSIZE - 1) * ycbcr_to_rgb_config->num_rows * ycbcr_to_rgb_config->num_cols) * sizeof(JSAMPLE);
    size_t output_size = (ycbcr_to_rgb_config->num_rows * ycbcr_to_rgb_config->num_cols * RGB_PIXELSIZE) * sizeof(JSAMPLE);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<ycbcr_to_rgb_input_t *>(count, ycbcr_to_rgb_input);
    init_1D<ycbcr_to_rgb_output_t *>(count, ycbcr_to_rgb_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<ycbcr_to_rgb_input_t>(1, ycbcr_to_rgb_input[i]);
        init_1D<ycbcr_to_rgb_output_t>(1, ycbcr_to_rgb_output[i]);

        random_init_3D<JSAMPLE>((RGB_PIXELSIZE - 1), ycbcr_to_rgb_config->num_rows, ycbcr_to_rgb_config->num_cols, ycbcr_to_rgb_input[i]->input_buf);
        random_init_2D<JSAMPLE>(ycbcr_to_rgb_config->num_rows, ycbcr_to_rgb_config->num_cols * RGB_PIXELSIZE, ycbcr_to_rgb_output[i]->output_buf);
    }

    config = (config_t *)ycbcr_to_rgb_config;
    input = (input_t **)ycbcr_to_rgb_input;
    output = (output_t **)ycbcr_to_rgb_output;

    return count;
}