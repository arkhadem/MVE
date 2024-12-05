#include "sum_from.hpp"

#include "webaudio.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int sum_from_init(size_t cache_size,
                  int LANE_NUM,
                  config_t *&config,
                  input_t **&input,
                  output_t **&output) {

    sum_from_config_t *sum_from_config = (sum_from_config_t *)config;
    sum_from_input_t **sum_from_input = (sum_from_input_t **)input;
    sum_from_output_t **sum_from_output = (sum_from_output_t **)output;

    // configuration
    int data_size = 4096;
    int number_of_channels = 4;

    init_1D<sum_from_config_t>(1, sum_from_config);
    sum_from_config->data_size = data_size;
    sum_from_config->number_of_channels = number_of_channels;

    // in/output versions
    size_t input_size = 2 * number_of_channels * data_size * sizeof(float);
    size_t output_size = number_of_channels * data_size * sizeof(float);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<sum_from_input_t *>(count, sum_from_input);
    init_1D<sum_from_output_t *>(count, sum_from_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<sum_from_input_t>(1, sum_from_input[i]);
        init_1D<sum_from_output_t>(1, sum_from_output[i]);

        random_init_2D<float>(number_of_channels, data_size, sum_from_input[i]->source1);
        random_init_2D<float>(number_of_channels, data_size, sum_from_input[i]->source2);
        random_init_2D<float>(number_of_channels, data_size, sum_from_output[i]->destination);
    }

    config = (config_t *)sum_from_config;
    input = (input_t **)sum_from_input;
    output = (output_t **)sum_from_output;

    return count;
}