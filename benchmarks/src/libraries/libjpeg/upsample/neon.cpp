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

#include "libjpeg.hpp"
#include "upsample.hpp"

/* Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 *
 * See comments above for details regarding color conversion and safe memory
 * access.
 */

void upsample_neon(int LANE_NUM,
                   config_t *config,
                   input_t *input,
                   output_t *output) {
    upsample_config_t *upsample_config = (upsample_config_t *)config;
    upsample_input_t *upsample_input = (upsample_input_t *)input;
    upsample_output_t *upsample_output = (upsample_output_t *)output;
    JSAMPROW outptr0;
    JSAMPROW outptr1;
    /* Pointers to Y (both rows), Cb, and Cr data */
    JSAMPROW inptr0_0;
    JSAMPROW inptr0_1;
    JSAMPROW inptr1;
    JSAMPROW inptr2;

    const int16x4_t consts = vld1_s16(upsample_consts);
    const int16x8_t neg_128 = vdupq_n_s16(-128);

    for (JDIMENSION row = 0; row < upsample_config->num_rows; row++) {
        inptr0_0 = upsample_input->input_buf[0][row * 2];
        inptr0_1 = upsample_input->input_buf[0][row * 2 + 1];
        inptr1 = upsample_input->input_buf[1][row];
        inptr2 = upsample_input->input_buf[2][row];
        outptr0 = upsample_output->output_buf[row * 2];
        outptr1 = upsample_output->output_buf[row * 2 + 1];

        for (JDIMENSION col = 0; col < upsample_config->num_cols; col += 8) {
            /* For each row, de-interleave Y component values into two separate
            * vectors, one containing the component values with even-numbered indices
            * and one containing the component values with odd-numbered indices.
            */
            uint8x8x2_t y0 = vld2_u8(inptr0_0);
            uint8x8x2_t y1 = vld2_u8(inptr0_1);
            uint8x8_t cb = vld1_u8(inptr1);
            uint8x8_t cr = vld1_u8(inptr2);
            /* Subtract 128 from Cb and Cr. */
            int16x8_t cr_128 =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr));
            int16x8_t cb_128 =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb));
            /* Compute G-Y: - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128) */
            int32x4_t g_sub_y_l = vmull_lane_s16(vget_low_s16(cb_128), consts, 0);
            int32x4_t g_sub_y_h = vmull_lane_s16(vget_high_s16(cb_128), consts, 0);
            g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16(cr_128), consts, 1);
            g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16(cr_128), consts, 1);
            /* Descale G components: shift right 15, round, and narrow to 16-bit. */
            int16x8_t g_sub_y = vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15),
                                             vrshrn_n_s32(g_sub_y_h, 15));
            /* Compute R-Y: 1.40200 * (Cr - 128) */
            int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);
            /* Compute B-Y: 1.77200 * (Cb - 128) */
            int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);
            /* For each row, add the chroma-derived values (G-Y, R-Y, and B-Y) to both
        * the "even" and "odd" Y component values.  This effectively upsamples the
        * chroma components both horizontally and vertically.
        */
            int16x8_t g0_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                               y0.val[0]));
            int16x8_t r0_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                               y0.val[0]));
            int16x8_t b0_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                               y0.val[0]));
            int16x8_t g0_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                               y0.val[1]));
            int16x8_t r0_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                               y0.val[1]));
            int16x8_t b0_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                               y0.val[1]));
            int16x8_t g1_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                               y1.val[0]));
            int16x8_t r1_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                               y1.val[0]));
            int16x8_t b1_even =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                               y1.val[0]));
            int16x8_t g1_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                               y1.val[1]));
            int16x8_t r1_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                               y1.val[1]));
            int16x8_t b1_odd =
                vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                               y1.val[1]));
            /* Convert each component to unsigned and narrow, clamping to [0-255].
        * Re-interleave the "even" and "odd" component values.
        */
            uint8x8x2_t r0 = vzip_u8(vqmovun_s16(r0_even), vqmovun_s16(r0_odd));
            uint8x8x2_t r1 = vzip_u8(vqmovun_s16(r1_even), vqmovun_s16(r1_odd));
            uint8x8x2_t g0 = vzip_u8(vqmovun_s16(g0_even), vqmovun_s16(g0_odd));
            uint8x8x2_t g1 = vzip_u8(vqmovun_s16(g1_even), vqmovun_s16(g1_odd));
            uint8x8x2_t b0 = vzip_u8(vqmovun_s16(b0_even), vqmovun_s16(b0_odd));
            uint8x8x2_t b1 = vzip_u8(vqmovun_s16(b1_even), vqmovun_s16(b1_odd));

            uint8x16x3_t rgb0, rgb1;
            rgb0.val[RGB_RED] = vcombine_u8(r0.val[0], r0.val[1]);
            rgb1.val[RGB_RED] = vcombine_u8(r1.val[0], r1.val[1]);
            rgb0.val[RGB_GREEN] = vcombine_u8(g0.val[0], g0.val[1]);
            rgb1.val[RGB_GREEN] = vcombine_u8(g1.val[0], g1.val[1]);
            rgb0.val[RGB_BLUE] = vcombine_u8(b0.val[0], b0.val[1]);
            rgb1.val[RGB_BLUE] = vcombine_u8(b1.val[0], b1.val[1]);
            /* Store RGB pixel data to memory. */
            vst3q_u8(outptr0, rgb0);
            vst3q_u8(outptr1, rgb1);

            /* Increment pointers. */
            inptr0_0 += 16;
            inptr0_1 += 16;
            inptr1 += 8;
            inptr2 += 8;
            outptr0 += (RGB_PIXELSIZE * 16);
            outptr1 += (RGB_PIXELSIZE * 16);
        }
    }
}