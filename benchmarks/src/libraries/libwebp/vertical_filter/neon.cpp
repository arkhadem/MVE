#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdio>
#include <stdint.h>

#include "libwebp.hpp"
#include "vertical_filter.hpp"

void vertical_filter_neon(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    vertical_filter_config_t *vertical_filter_config = (vertical_filter_config_t *)config;
    vertical_filter_input_t *vertical_filter_input = (vertical_filter_input_t *)input;
    vertical_filter_output_t *vertical_filter_output = (vertical_filter_output_t *)output;

    int num_rows = vertical_filter_config->num_rows;
    int num_cols = vertical_filter_config->num_cols;
    int stride = vertical_filter_config->stride;
    uint8_t *out = vertical_filter_output->out;
    uint8_t *preds = vertical_filter_input->in;
    uint8_t *in = preds + stride;
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; col += 16) {
            const uint8x16_t A = vld1q_u8(&in[col]);
            const uint8x16_t B = vld1q_u8(&preds[col]);
            const uint8x16_t C = vsubq_u8(A, B);
            vst1q_u8(&out[col], C);
        }
        preds += stride;
        in += stride;
        out += stride;
    }
}
