#include "is_audible.hpp"

#include "webaudio.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int is_audible_init(size_t cache_size,
                    int LANE_NUM,
                    config_t *&config,
                    input_t **&input,
                    output_t **&output) {

    is_audible_config_t *is_audible_config = (is_audible_config_t *)config;
    is_audible_input_t **is_audible_input = (is_audible_input_t **)input;
    is_audible_output_t **is_audible_output = (is_audible_output_t **)output;

    // configuration
    int data_size = 4096;
    int number_of_channels = 4;

    init_1D<is_audible_config_t>(1, is_audible_config);
    is_audible_config->data_size = data_size;
    is_audible_config->number_of_channels = number_of_channels;

    // in/output versions
    size_t input_size = number_of_channels * data_size * sizeof(float);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<is_audible_input_t *>(count, is_audible_input);
    init_1D<is_audible_output_t *>(count, is_audible_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<is_audible_input_t>(1, is_audible_input[i]);
        init_1D<is_audible_output_t>(1, is_audible_output[i]);

        random_init_2D<float>(number_of_channels, data_size, is_audible_input[i]->data);
        random_init_1D<bool>(1, is_audible_output[i]->return_value);
    }

    config = (config_t *)is_audible_config;
    input = (input_t **)is_audible_input;
    output = (output_t **)is_audible_output;

    return count;
}