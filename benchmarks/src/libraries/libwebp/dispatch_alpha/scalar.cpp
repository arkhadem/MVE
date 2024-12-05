#include "dispatch_alpha.hpp"
#include "scalar_kernels.hpp"

void dispatch_alpha_scalar(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    dispatch_alpha_config_t *dispatch_alpha_config = (dispatch_alpha_config_t *)config;
    dispatch_alpha_input_t *dispatch_alpha_input = (dispatch_alpha_input_t *)input;
    dispatch_alpha_output_t *dispatch_alpha_output = (dispatch_alpha_output_t *)output;

    uint32_t alpha_mask = 0xff;
    int height = dispatch_alpha_config->num_rows;
    int width = dispatch_alpha_config->num_cols;
    int alpha_stride = width;
    int dst_stride = width << 2;
    uint8_t *alpha = dispatch_alpha_input->alpha;
    uint8_t *dst = dispatch_alpha_output->dst;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const uint32_t alpha_value = alpha[i];
            dst[4 * i] = alpha_value;
            alpha_mask &= alpha_value;
        }
        alpha += alpha_stride;
        dst += dst_stride;
    }

    dispatch_alpha_output->return_val[0] = (alpha_mask != 0xff);
}