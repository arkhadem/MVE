#include "mve.hpp"
#include "mve_kernels.hpp"

#include "handle_nan.hpp"

#ifndef MIN_BLOCK
#define MIN_BLOCK 64
#endif

void handle_nan_mve(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    handle_nan_config_t *handle_nan_config = (handle_nan_config_t *)config;
    handle_nan_input_t *handle_nan_input = (handle_nan_input_t *)input;

    float *values = handle_nan_input->values;
    unsigned number_of_values = handle_nan_config->number_of_values;
    float default_value = handle_nan_config->default_value;

    // DIM0: values
    _mve_set_dim_count(1);

    __mdvf default_value_f = _mve_set1_f(default_value);

    // Loading and storing everything sequentially
    __vidx_var stride = {1, 0, 0, 0};

    int value_pointer = 0;
    while (value_pointer < number_of_values) {
        int remaining_values = number_of_values - value_pointer;
        remaining_values = remaining_values > LANE_NUM ? LANE_NUM : remaining_values;
        if (remaining_values != LANE_NUM) {
            _mve_set_dim_length(1, remaining_values);
        }

        __mdvf value_f = _mve_load_f(values, stride);

        _mve_cmpneq_f(value_f, value_f);

        value_f = _mve_assign_f(value_f, default_value_f);
        // free value_f
        _mve_free_f();

        _mve_set_mask();

        _mve_store_f(values, value_f, stride);
        // free value_f
        _mve_free_f();

        values += LANE_NUM;
        value_pointer += LANE_NUM;
    }

    // free default_value_f
    _mve_free_f();
}