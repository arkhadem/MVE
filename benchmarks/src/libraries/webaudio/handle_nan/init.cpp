#include "handle_nan.hpp"

#include "webaudio.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int handle_nan_init(size_t cache_size,
                    int LANE_NUM,
                    config_t *&config,
                    input_t **&input,
                    output_t **&output) {

    handle_nan_config_t *handle_nan_config = (handle_nan_config_t *)config;
    handle_nan_input_t **handle_nan_input = (handle_nan_input_t **)input;
    handle_nan_output_t **handle_nan_output = (handle_nan_output_t **)output;

    // configuration
    int number_of_values = 16384;

    init_1D<handle_nan_config_t>(1, handle_nan_config);
    handle_nan_config->number_of_values = number_of_values;

    // in/output versions
    size_t input_size = number_of_values * sizeof(float);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<handle_nan_input_t *>(count, handle_nan_input);
    init_1D<handle_nan_output_t *>(count, handle_nan_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<handle_nan_input_t>(1, handle_nan_input[i]);
        init_1D<handle_nan_output_t>(1, handle_nan_output[i]);

        random_init_1D<float>(number_of_values, handle_nan_input[i]->values);
    }

    config = (config_t *)handle_nan_config;
    input = (input_t **)handle_nan_input;
    output = (output_t **)handle_nan_output;

    return count;
}