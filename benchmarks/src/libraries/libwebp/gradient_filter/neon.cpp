#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "gradient_filter.hpp"
#include "libwebp.hpp"

#define U8_TO_S16(A) vreinterpretq_s16_u16(vmovl_u8(A))
#define LOAD_U8_TO_S16(A) U8_TO_S16(vld1_u8(A))

void gradient_filter_neon(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    gradient_filter_config_t *gradient_filter_config = (gradient_filter_config_t *)config;
    gradient_filter_input_t *gradient_filter_input = (gradient_filter_input_t *)input;
    gradient_filter_output_t *gradient_filter_output = (gradient_filter_output_t *)output;

    int num_rows = gradient_filter_config->num_rows;
    int num_cols = gradient_filter_config->num_cols;
    int stride = gradient_filter_config->stride;
    uint8_t *in = gradient_filter_input->in + stride + 1;
    uint8_t *preds = gradient_filter_input->in + 1;
    uint8_t *out = gradient_filter_output->out;
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; col += 8) {
            const uint8x8_t A = vld1_u8(&in[col - 1]);
            const uint8x8_t B = vld1_u8(&preds[col + 0]);
            const int16x8_t C = vreinterpretq_s16_u16(vaddl_u8(A, B));
            const int16x8_t D = LOAD_U8_TO_S16(&preds[col - 1]);
            const uint8x8_t E = vqmovun_s16(vsubq_s16(C, D));
            const uint8x8_t F = vld1_u8(&in[col + 0]);
            vst1_u8(&out[col], vsub_u8(F, E));
        }
        preds += stride;
        in += stride;
        out += stride;
    }
}
