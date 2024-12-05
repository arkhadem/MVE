#include "copy_with_gain.hpp"

#include "webaudio.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int copy_with_gain_init(size_t cache_size,
                        int LANE_NUM,
                        config_t *&config,
                        input_t **&input,
                        output_t **&output) {

    copy_with_gain_config_t *copy_with_gain_config = (copy_with_gain_config_t *)config;
    copy_with_gain_input_t **copy_with_gain_input = (copy_with_gain_input_t **)input;
    copy_with_gain_output_t **copy_with_gain_output = (copy_with_gain_output_t **)output;

    // configuration
    int data_size = 4096;
    int number_of_channels = 4;

    init_1D<copy_with_gain_config_t>(1, copy_with_gain_config);
    copy_with_gain_config->data_size = data_size;
    copy_with_gain_config->gain = 236.2886;
    copy_with_gain_config->number_of_channels = number_of_channels;

    // in/output versions
    size_t input_size = number_of_channels * data_size * sizeof(float);
    size_t output_size = number_of_channels * data_size * sizeof(float);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<copy_with_gain_input_t *>(count, copy_with_gain_input);
    init_1D<copy_with_gain_output_t *>(count, copy_with_gain_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<copy_with_gain_input_t>(1, copy_with_gain_input[i]);
        init_1D<copy_with_gain_output_t>(1, copy_with_gain_output[i]);

        random_init_2D<float>(number_of_channels, data_size, copy_with_gain_input[i]->sources);
        random_init_2D<float>(number_of_channels, data_size, copy_with_gain_output[i]->destinations);
    }

    config = (config_t *)copy_with_gain_config;
    input = (input_t **)copy_with_gain_input;
    output = (output_t **)copy_with_gain_output;

    return count;
}