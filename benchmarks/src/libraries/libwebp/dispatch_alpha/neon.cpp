#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "dispatch_alpha.hpp"
#include "libwebp.hpp"

void dispatch_alpha_neon(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    dispatch_alpha_config_t *dispatch_alpha_config = (dispatch_alpha_config_t *)config;
    dispatch_alpha_input_t *dispatch_alpha_input = (dispatch_alpha_input_t *)input;
    dispatch_alpha_output_t *dispatch_alpha_output = (dispatch_alpha_output_t *)output;

    uint32_t alpha_mask = 0xffu;
    uint8x8_t mask8 = vdup_n_u8(0xff);
    uint32_t tmp[2];
    int height = dispatch_alpha_config->num_rows;
    int width = dispatch_alpha_config->num_cols;
    int alpha_stride = width;
    int dst_stride = width << 2;
    uint8_t *alpha = dispatch_alpha_input->alpha;
    uint8_t *dst = dispatch_alpha_output->dst;

    for (int j = 0; j < height; ++j) {
        // We don't know if alpha is first or last in dst[] (depending on rgbA/Argb
        // mode). So we must be sure dst[4*i + 8 - 1] is writable for the store.
        // Hence the test with 'width - 1' instead of just 'width'.
        for (int i = 0; i + 8 <= width - 1; i += 8) {
            uint8x8x4_t rgbX = vld4_u8((const uint8_t *)(dst + 4 * i));
            const uint8x8_t alphas = vld1_u8(alpha + i);
            rgbX.val[0] = alphas;
            vst4_u8((uint8_t *)(dst + 4 * i), rgbX);
            mask8 = vand_u8(mask8, alphas);
        }
        alpha += alpha_stride;
        dst += dst_stride;
    }
    vst1_u8((uint8_t *)tmp, mask8);
    alpha_mask *= 0x01010101;
    alpha_mask &= tmp[0];
    alpha_mask &= tmp[1];
    dispatch_alpha_output->return_val[0] = (alpha_mask != 0xffffffffu);
}