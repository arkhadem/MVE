#include "downsample.hpp"

#include "libjpeg.hpp"

#include "benchmark.hpp"

#include "init.hpp"
#include <cstddef>
#include <cstdio>

int downsample_init(size_t cache_size,
                    int LANE_NUM,
                    config_t *&config,
                    input_t **&input,
                    output_t **&output) {

    downsample_config_t *downsample_config = (downsample_config_t *)config;
    downsample_input_t **downsample_input = (downsample_input_t **)input;
    downsample_output_t **downsample_output = (downsample_output_t **)output;

    // configuration
    init_1D<downsample_config_t>(1, downsample_config);
    downsample_config->num_rows = 16;
    downsample_config->num_cols = 1024;

    // in/output versions
    size_t input_size = (downsample_config->num_rows * 2 * downsample_config->num_cols * 2) * sizeof(JSAMPLE);
    size_t output_size = (downsample_config->num_rows * downsample_config->num_cols) * sizeof(JSAMPLE);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<downsample_input_t *>(count, downsample_input);
    init_1D<downsample_output_t *>(count, downsample_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<downsample_input_t>(1, downsample_input[i]);
        init_1D<downsample_output_t>(1, downsample_output[i]);

        random_init_2D<JSAMPLE>(downsample_config->num_rows * 2, downsample_config->num_cols * 2, downsample_input[i]->input_buf);
        random_init_2D<JSAMPLE>(downsample_config->num_rows, downsample_config->num_cols, downsample_output[i]->output_buf);
    }

    config = (config_t *)downsample_config;
    input = (input_t **)downsample_input;
    output = (output_t **)downsample_output;

    return count;
}