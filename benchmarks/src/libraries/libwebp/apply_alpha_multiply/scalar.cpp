#include "apply_alpha_multiply.hpp"
#include "scalar_kernels.hpp"

void apply_alpha_multiply_scalar(int LANE_NUM,
                                 config_t *config,
                                 input_t *input,
                                 output_t *output) {
    apply_alpha_multiply_config_t *apply_alpha_multiply_config = (apply_alpha_multiply_config_t *)config;
    apply_alpha_multiply_input_t *apply_alpha_multiply_input = (apply_alpha_multiply_input_t *)input;

    int h = apply_alpha_multiply_config->num_rows;
    int w = apply_alpha_multiply_config->num_cols;
    int stride = w << 2;
    uint8_t *rgba = apply_alpha_multiply_input->rgba;

    while (h-- > 0) {
        uint8_t *const rgb = rgba + 1;
        const uint8_t *const alpha = rgba;
        for (int i = 0; i < w; ++i) {
            const uint32_t a = alpha[4 * i];
            if (a != 0xff) {
                const uint32_t mult = a * 32897U;
                rgb[4 * i + 0] = (rgb[4 * i + 0] * mult) >> 23;
                rgb[4 * i + 1] = (rgb[4 * i + 1] * mult) >> 23;
                rgb[4 * i + 2] = (rgb[4 * i + 2] * mult) >> 23;
            }
        }
        rgba += stride;
    }
}