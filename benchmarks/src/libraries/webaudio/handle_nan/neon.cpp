#include "handle_nan.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <math.h>

void handle_nan_neon(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    handle_nan_config_t *handle_nan_config = (handle_nan_config_t *)config;
    handle_nan_input_t *handle_nan_input = (handle_nan_input_t *)input;

    float *values = handle_nan_input->values;
    unsigned number_of_values = handle_nan_config->number_of_values;
    float default_value = handle_nan_config->default_value;

    int k = 0;

    uint32x4_t defaults = reinterpret_cast<uint32x4_t>(vdupq_n_f32(default_value));
    for (k = 0; k < number_of_values; k += 4) {
        float32x4_t v = vld1q_f32(values + k);
        // Returns true (all ones) if v is not NaN
        uint32x4_t is_not_nan = vceqq_f32(v, v);
        // Get the parts that are not NaN
        uint32x4_t result = vandq_u32(is_not_nan, reinterpret_cast<uint32x4_t>(v));
        // Replace the parts that are NaN with the default and merge with previous
        // result.  (Note: vbic_u32(x, y) = x and not y)
        result = vorrq_u32(result, vbicq_u32(defaults, is_not_nan));
        vst1q_f32(values + k, reinterpret_cast<float32x4_t>(result));
    }

    for (; k < number_of_values; ++k) {
        if (isnan(values[k])) {
            values[k] = default_value;
        }
    }
}