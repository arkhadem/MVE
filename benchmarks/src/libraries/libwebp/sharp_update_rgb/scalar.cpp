#include "scalar_kernels.hpp"
#include "sharp_update_rgb.hpp"

void sharp_update_rgb_scalar(int LANE_NUM,
                             config_t *config,
                             input_t *input,
                             output_t *output) {
    sharp_update_rgb_config_t *sharp_update_rgb_config = (sharp_update_rgb_config_t *)config;
    sharp_update_rgb_input_t *sharp_update_rgb_input = (sharp_update_rgb_input_t *)input;
    sharp_update_rgb_output_t *sharp_update_rgb_output = (sharp_update_rgb_output_t *)output;

    int16_t *src = sharp_update_rgb_input->src;
    int16_t *ref = sharp_update_rgb_input->ref;
    int16_t *dst = sharp_update_rgb_output->dst;

    for (int i = 0; i < sharp_update_rgb_config->num_cols * sharp_update_rgb_config->num_rows; ++i, src++, ref++, dst++) {
        int diff_uv = *ref - *src;
        *dst = *dst + diff_uv;
    }
}