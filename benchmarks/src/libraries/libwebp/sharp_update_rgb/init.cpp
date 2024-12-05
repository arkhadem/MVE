#include "sharp_update_rgb.hpp"

#include "libwebp.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int sharp_update_rgb_init(size_t cache_size,
                          int LANE_NUM,
                          config_t *&config,
                          input_t **&input,
                          output_t **&output) {

    sharp_update_rgb_config_t *sharp_update_rgb_config = (sharp_update_rgb_config_t *)config;
    sharp_update_rgb_input_t **sharp_update_rgb_input = (sharp_update_rgb_input_t **)input;
    sharp_update_rgb_output_t **sharp_update_rgb_output = (sharp_update_rgb_output_t **)output;

    // configuration
    int rows = 16;
    int columns = 1024;

    init_1D<sharp_update_rgb_config_t>(1, sharp_update_rgb_config);
    sharp_update_rgb_config->num_rows = rows;
    sharp_update_rgb_config->num_cols = columns;

    // in/output versions
    size_t input_size = 2 * rows * columns * sizeof(int16_t);
    size_t output_size = rows * columns * sizeof(int16_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<sharp_update_rgb_input_t *>(count, sharp_update_rgb_input);
    init_1D<sharp_update_rgb_output_t *>(count, sharp_update_rgb_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<sharp_update_rgb_input_t>(1, sharp_update_rgb_input[i]);
        init_1D<sharp_update_rgb_output_t>(1, sharp_update_rgb_output[i]);

        random_init_1D<int16_t>(rows * columns, sharp_update_rgb_input[i]->ref);
        random_init_1D<int16_t>(rows * columns, sharp_update_rgb_input[i]->src);
        random_init_1D<int16_t>(rows * columns, sharp_update_rgb_output[i]->dst);
    }

    config = (config_t *)sharp_update_rgb_config;
    input = (input_t **)sharp_update_rgb_input;
    output = (output_t **)sharp_update_rgb_output;

    return count;
}