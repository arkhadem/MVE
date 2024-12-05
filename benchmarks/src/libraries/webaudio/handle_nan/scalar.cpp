#include "handle_nan.hpp"
#include "scalar_kernels.hpp"
#include <math.h>

void handle_nan_scalar(int LANE_NUM,
                       config_t *config,
                       input_t *input,
                       output_t *output) {
    handle_nan_config_t *handle_nan_config = (handle_nan_config_t *)config;
    handle_nan_input_t *handle_nan_input = (handle_nan_input_t *)input;

    float *values = handle_nan_input->values;
    unsigned number_of_values = handle_nan_config->number_of_values;
    float default_value = handle_nan_config->default_value;

    for (int k = 0; k < number_of_values; ++k) {
        if (isnan(values[k])) {
            values[k] = default_value;
        }
    }
}