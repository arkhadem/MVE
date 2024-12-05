#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "apply_alpha_multiply.hpp"
#include "libwebp.hpp"

void apply_alpha_multiply_neon(int LANE_NUM,
                               config_t *config,
                               input_t *input,
                               output_t *output) {
    apply_alpha_multiply_config_t *apply_alpha_multiply_config = (apply_alpha_multiply_config_t *)config;
    apply_alpha_multiply_input_t *apply_alpha_multiply_input = (apply_alpha_multiply_input_t *)input;

    int h = apply_alpha_multiply_config->num_rows;
    int w = apply_alpha_multiply_config->num_cols;
    int stride = w << 2;
    uint8_t *rgba = apply_alpha_multiply_input->rgba;

    const uint16x8_t kOne = vdupq_n_u16(1u);
    while (h-- > 0) {
        uint32_t *const rgbx = (uint32_t *)rgba;
        ;
        for (int i = 0; i + 8 <= w; i += 8) {
            // load aaaa...|rrrr...|gggg...|bbbb...
            uint8x8x4_t RGBX = vld4_u8((const uint8_t *)(rgbx + i));

            const uint8x8_t alpha = RGBX.val[0];
            const uint16x8_t r1 = vmull_u8(RGBX.val[1], alpha);
            const uint16x8_t g1 = vmull_u8(RGBX.val[2], alpha);
            const uint16x8_t b1 = vmull_u8(RGBX.val[3], alpha);
            /* we use: v / 255 = (v + 1 + (v >> 8)) >> 8 */
            const uint16x8_t r2 = vsraq_n_u16(r1, r1, 8);
            const uint16x8_t g2 = vsraq_n_u16(g1, g1, 8);
            const uint16x8_t b2 = vsraq_n_u16(b1, b1, 8);
            const uint16x8_t r3 = vaddq_u16(r2, kOne);
            const uint16x8_t g3 = vaddq_u16(g2, kOne);
            const uint16x8_t b3 = vaddq_u16(b2, kOne);
            RGBX.val[1] = vshrn_n_u16(r3, 8);
            RGBX.val[2] = vshrn_n_u16(g3, 8);
            RGBX.val[3] = vshrn_n_u16(b3, 8);

            vst4_u8((uint8_t *)(rgbx + i), RGBX);
        }
        rgba += stride;
    }
}
