/* YCbCr -> RGB conversion is defined by the following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.3441467 = 11277 * 2^-15
 *    0.7141418 = 23401 * 2^-15
 *    1.4020386 = 22971 * 2^-14
 *    1.7720337 = 29033 * 2^-14
 * These constants are defined in jdcolor-neon.c.
 *
 * To ensure correct results, rounding is used when descaling.
 */

#include "neon_kernels.hpp"
#include <arm_neon.h>

#include "ycbcr_to_rgb.hpp"

void ycbcr_to_rgb_neon(int LANE_NUM,
                       config_t *config,
                       input_t *input,
                       output_t *output) {
    ycbcr_to_rgb_config_t *ycbcr_to_rgb_config = (ycbcr_to_rgb_config_t *)config;
    ycbcr_to_rgb_input_t *ycbcr_to_rgb_input = (ycbcr_to_rgb_input_t *)input;
    ycbcr_to_rgb_output_t *ycbcr_to_rgb_output = (ycbcr_to_rgb_output_t *)output;

    JSAMPROW outptr;
    /* Pointers to Y, Cb, and Cr data */
    JSAMPROW inptr0;
    JSAMPROW inptr1;
    JSAMPROW inptr2;

    const int16x4_t consts = vld1_s16(ycbcr_to_rgb_const);
    const int16x8_t neg_128 = vdupq_n_s16(-128);

    for (JDIMENSION row = 0; row < ycbcr_to_rgb_config->num_rows; row++) {
        inptr0 = ycbcr_to_rgb_input->input_buf[0][row];
        inptr1 = ycbcr_to_rgb_input->input_buf[1][row];
        inptr2 = ycbcr_to_rgb_input->input_buf[2][row];
        outptr = ycbcr_to_rgb_output->output_buf[row];

        for (JDIMENSION col = 0; col < ycbcr_to_rgb_config->num_cols; col += 16) {
            uint8x16_t y = vld1q_u8(inptr0);
            uint8x16_t cb = vld1q_u8(inptr1);
            uint8x16_t cr = vld1q_u8(inptr2);
            /* Subtract 128 from Cb and Cr. */
            int16x8_t cr_128_l =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                               vget_low_u8(cr)));
            int16x8_t cr_128_h =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                               vget_high_u8(cr)));
            int16x8_t cb_128_l =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                               vget_low_u8(cb)));
            int16x8_t cb_128_h =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                               vget_high_u8(cb)));
            /* Compute G-Y: - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128) */
            int32x4_t g_sub_y_ll = vmull_lane_s16(vget_low_s16(cb_128_l), consts, 0);
            int32x4_t g_sub_y_lh = vmull_lane_s16(vget_high_s16(cb_128_l),
                                                  consts, 0);
            int32x4_t g_sub_y_hl = vmull_lane_s16(vget_low_s16(cb_128_h), consts, 0);
            int32x4_t g_sub_y_hh = vmull_lane_s16(vget_high_s16(cb_128_h),
                                                  consts, 0);
            g_sub_y_ll = vmlsl_lane_s16(g_sub_y_ll, vget_low_s16(cr_128_l),
                                        consts, 1);
            g_sub_y_lh = vmlsl_lane_s16(g_sub_y_lh, vget_high_s16(cr_128_l),
                                        consts, 1);
            g_sub_y_hl = vmlsl_lane_s16(g_sub_y_hl, vget_low_s16(cr_128_h),
                                        consts, 1);
            g_sub_y_hh = vmlsl_lane_s16(g_sub_y_hh, vget_high_s16(cr_128_h),
                                        consts, 1);
            /* Descale G components: shift right 15, round, and narrow to 16-bit. */
            int16x8_t g_sub_y_l = vcombine_s16(vrshrn_n_s32(g_sub_y_ll, 15),
                                               vrshrn_n_s32(g_sub_y_lh, 15));
            int16x8_t g_sub_y_h = vcombine_s16(vrshrn_n_s32(g_sub_y_hl, 15),
                                               vrshrn_n_s32(g_sub_y_hh, 15));
            /* Compute R-Y: 1.40200 * (Cr - 128) */
            int16x8_t r_sub_y_l = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_l, 1),
                                                     consts, 2);
            int16x8_t r_sub_y_h = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_h, 1),
                                                     consts, 2);
            /* Compute B-Y: 1.77200 * (Cb - 128) */
            int16x8_t b_sub_y_l = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_l, 1),
                                                     consts, 3);
            int16x8_t b_sub_y_h = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_h, 1),
                                                     consts, 3);
            /* Add Y. */
            int16x8_t r_l =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y_l),
                                               vget_low_u8(y)));
            int16x8_t r_h =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y_h),
                                               vget_high_u8(y)));
            int16x8_t b_l =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y_l),
                                               vget_low_u8(y)));
            int16x8_t b_h =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y_h),
                                               vget_high_u8(y)));
            int16x8_t g_l =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y_l),
                                               vget_low_u8(y)));
            int16x8_t g_h =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y_h),
                                               vget_high_u8(y)));

            uint8x16x4_t rgba;
            /* Convert each component to unsigned and narrow, clamping to [0-255]. */
            rgba.val[RGB_RED] = vcombine_u8(vqmovun_s16(r_l), vqmovun_s16(r_h));
            rgba.val[RGB_GREEN] = vcombine_u8(vqmovun_s16(g_l), vqmovun_s16(g_h));
            rgba.val[RGB_BLUE] = vcombine_u8(vqmovun_s16(b_l), vqmovun_s16(b_h));
            /* Set alpha channel to opaque (0xFF). */
            rgba.val[RGB_ALPHA] = vdupq_n_u8(0xFF);
            /* Store RGBA pixel data to memory. */
            vst4q_u8(outptr, rgba);

            /* Increment pointers. */
            inptr0 += 16;
            inptr1 += 16;
            inptr2 += 16;
            outptr += (RGB_PIXELSIZE * 16);
        }
    }
}