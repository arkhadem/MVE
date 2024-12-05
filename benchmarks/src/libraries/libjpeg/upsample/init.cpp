#include "upsample.hpp"

#include "libjpeg.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int upsample_init(size_t cache_size,
                  int LANE_NUM,
                  config_t *&config,
                  input_t **&input,
                  output_t **&output) {

    upsample_config_t *upsample_config = (upsample_config_t *)config;
    upsample_input_t **upsample_input = (upsample_input_t **)input;
    upsample_output_t **upsample_output = (upsample_output_t **)output;

    // configuration
    init_1D<upsample_config_t>(1, upsample_config);
    upsample_config->num_rows = 8;
    upsample_config->num_cols = 512;

    // in/output versions
    size_t input_size = ((RGB_PIXELSIZE - 1) * 2 * upsample_config->num_rows * 2 * upsample_config->num_cols) * sizeof(JSAMPLE);
    size_t output_size = (2 * upsample_config->num_rows * 2 * upsample_config->num_cols * RGB_PIXELSIZE) * sizeof(JSAMPLE);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<upsample_input_t *>(count, upsample_input);
    init_1D<upsample_output_t *>(count, upsample_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<upsample_input_t>(1, upsample_input[i]);
        init_1D<upsample_output_t>(1, upsample_output[i]);

        random_init_3D<JSAMPLE>((RGB_PIXELSIZE - 1), 2 * upsample_config->num_rows, 2 * upsample_config->num_cols, upsample_input[i]->input_buf);
        random_init_2D<JSAMPLE>(2 * upsample_config->num_rows, 2 * upsample_config->num_cols * RGB_PIXELSIZE, upsample_output[i]->output_buf);
    }

    config = (config_t *)upsample_config;
    input = (input_t **)upsample_input;
    output = (output_t **)upsample_output;

    return count;
}

const int16_t upsample_consts[4] = {-F_0_344, F_0_714, F_1_402, F_1_772};