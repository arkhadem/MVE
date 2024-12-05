#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "libwebp.hpp"
#include "sharp_update_rgb.hpp"

void sharp_update_rgb_neon(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    sharp_update_rgb_config_t *sharp_update_rgb_config = (sharp_update_rgb_config_t *)config;
    sharp_update_rgb_input_t *sharp_update_rgb_input = (sharp_update_rgb_input_t *)input;
    sharp_update_rgb_output_t *sharp_update_rgb_output = (sharp_update_rgb_output_t *)output;

    int16_t *src = sharp_update_rgb_input->src;
    int16_t *ref = sharp_update_rgb_input->ref;
    int16_t *dst = sharp_update_rgb_output->dst;

    for (int i = 0; i < sharp_update_rgb_config->num_cols * sharp_update_rgb_config->num_rows; i += 8, src += 8, ref += 8, dst += 8) {
        const int16x8_t A = vld1q_s16(ref);
        const int16x8_t B = vld1q_s16(src);
        const int16x8_t C = vld1q_s16(dst);
        const int16x8_t D = vsubq_s16(A, B); // diff_uv
        const int16x8_t E = vaddq_s16(C, D); // new_uv
        vst1q_s16(dst, E);
    }
}
