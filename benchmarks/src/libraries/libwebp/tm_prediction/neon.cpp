#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "libwebp.hpp"
#include "tm_prediction.hpp"

// Zero extend 'v' to an int16x8_t.
static inline int16x8_t ConvertU8ToS16_NEON(uint8x8_t v) {
    return vreinterpretq_s16_u16(vmovl_u8(v));
}

void tm_prediction_neon(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    tm_prediction_config_t *tm_prediction_config = (tm_prediction_config_t *)config;
    tm_prediction_output_t *tm_prediction_output = (tm_prediction_output_t *)output;

    int num_blocks = tm_prediction_config->num_blocks;
    int BPS = tm_prediction_config->pic_size;

    for (int i = 0; i < num_blocks; i++) {
        uint8_t *src = tm_prediction_output->block_dst[i];
        uint8_t *dst = tm_prediction_output->block_dst[i];
        uint8_t *src_tmp = src - BPS;
        const uint8x8_t TL = vld1_dup_u8(src_tmp - 1); // top-left pixel 'A[-1]'
        const uint8x16_t T = vld1q_u8(src_tmp);        // top row 'A[0..15]'
        // A[c] - A[-1]
        const int16x8_t d_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(T), TL));
        const int16x8_t d_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(T), TL));
        src--;
        for (int y = 0; y < 16; y += 4) {
            // left edge
            const int16x8_t L0 = vreinterpretq_s16_u16(vmovl_u8(vld1_dup_u8(src)));
            src += BPS;
            const int16x8_t L1 = vreinterpretq_s16_u16(vmovl_u8(vld1_dup_u8(src)));
            src += BPS;
            const int16x8_t L2 = vreinterpretq_s16_u16(vmovl_u8(vld1_dup_u8(src)));
            src += BPS;
            const int16x8_t L3 = vreinterpretq_s16_u16(vmovl_u8(vld1_dup_u8(src)));
            src += BPS;
            const int16x8_t r0_lo = vaddq_s16(L0, d_lo); // L[r]  +  A[c]  -  A[-1]
            const int16x8_t r1_lo = vaddq_s16(L1, d_lo);
            const int16x8_t r2_lo = vaddq_s16(L2, d_lo);
            const int16x8_t r3_lo = vaddq_s16(L3, d_lo);
            const int16x8_t r0_hi = vaddq_s16(L0, d_hi);
            const int16x8_t r1_hi = vaddq_s16(L1, d_hi);
            const int16x8_t r2_hi = vaddq_s16(L2, d_hi);
            const int16x8_t r3_hi = vaddq_s16(L3, d_hi);
            // Saturate and store the result.
            const uint8x16_t row0 = vcombine_u8(vqmovun_s16(r0_lo), vqmovun_s16(r0_hi));
            const uint8x16_t row1 = vcombine_u8(vqmovun_s16(r1_lo), vqmovun_s16(r1_hi));
            const uint8x16_t row2 = vcombine_u8(vqmovun_s16(r2_lo), vqmovun_s16(r2_hi));
            const uint8x16_t row3 = vcombine_u8(vqmovun_s16(r3_lo), vqmovun_s16(r3_hi));
            vst1q_u8(dst, row0);
            dst += BPS;
            vst1q_u8(dst, row1);
            dst += BPS;
            vst1q_u8(dst, row2);
            dst += BPS;
            vst1q_u8(dst, row3);
            dst += BPS;
        }
    }
}

void tm_prediction_original_neon(int LANE_NUM,
                                 config_t *config,
                                 input_t *input,
                                 output_t *output) {
    tm_prediction_config_t *tm_prediction_config = (tm_prediction_config_t *)config;
    tm_prediction_output_t *tm_prediction_output = (tm_prediction_output_t *)output;

    int num_blocks = tm_prediction_config->num_blocks;
    int BPS = tm_prediction_config->pic_size;

    for (int i = 0; i < num_blocks; i++) {
        uint8_t *dst = tm_prediction_output->block_dst[i];
        const uint8x8_t TL = vld1_dup_u8(dst - BPS - 1); // top-left pixel 'A[-1]'
        const uint8x16_t T = vld1q_u8(dst - BPS);        // top row 'A[0..15]'
        // A[c] - A[-1]
        const int16x8_t d_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(T), TL));
        const int16x8_t d_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(T), TL));
        int y;
        for (y = 0; y < 16; y += 4) {
            // left edge
            const int16x8_t L0 = ConvertU8ToS16_NEON(vld1_dup_u8(dst + 0 * BPS - 1));
            const int16x8_t L1 = ConvertU8ToS16_NEON(vld1_dup_u8(dst + 1 * BPS - 1));
            const int16x8_t L2 = ConvertU8ToS16_NEON(vld1_dup_u8(dst + 2 * BPS - 1));
            const int16x8_t L3 = ConvertU8ToS16_NEON(vld1_dup_u8(dst + 3 * BPS - 1));
            const int16x8_t r0_lo = vaddq_s16(L0, d_lo); // L[r] + A[c] - A[-1]
            const int16x8_t r1_lo = vaddq_s16(L1, d_lo);
            const int16x8_t r2_lo = vaddq_s16(L2, d_lo);
            const int16x8_t r3_lo = vaddq_s16(L3, d_lo);
            const int16x8_t r0_hi = vaddq_s16(L0, d_hi);
            const int16x8_t r1_hi = vaddq_s16(L1, d_hi);
            const int16x8_t r2_hi = vaddq_s16(L2, d_hi);
            const int16x8_t r3_hi = vaddq_s16(L3, d_hi);
            // Saturate and store the result.
            const uint8x16_t row0 = vcombine_u8(vqmovun_s16(r0_lo), vqmovun_s16(r0_hi));
            const uint8x16_t row1 = vcombine_u8(vqmovun_s16(r1_lo), vqmovun_s16(r1_hi));
            const uint8x16_t row2 = vcombine_u8(vqmovun_s16(r2_lo), vqmovun_s16(r2_hi));
            const uint8x16_t row3 = vcombine_u8(vqmovun_s16(r3_lo), vqmovun_s16(r3_hi));
            vst1q_u8(dst + 0 * BPS, row0);
            vst1q_u8(dst + 1 * BPS, row1);
            vst1q_u8(dst + 2 * BPS, row2);
            vst1q_u8(dst + 3 * BPS, row3);
            dst += 4 * BPS;
        }
    }
}
