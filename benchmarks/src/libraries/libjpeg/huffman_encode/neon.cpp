#include "neon_kernels.hpp"
#include <arm_neon.h>

#include "huffman_encode.hpp"
#include "libjpeg.hpp"
/* Data preparation for encode_mcu_AC_first().
 *
 * The equivalent scalar C function (encode_mcu_AC_first_prepare()) can be
 * found in jcphuff.c.
 */

void huffman_encode_neon(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    huffman_encode_config_t *huffman_encode_config = (huffman_encode_config_t *)config;
    huffman_encode_input_t *huffman_encode_input = (huffman_encode_input_t *)input;
    huffman_encode_output_t *huffman_encode_output = (huffman_encode_output_t *)output;

    int16_t *input_addr = huffman_encode_input->input_buf;
    uint16_t *output_addr = huffman_encode_output->output_buf;
    uint32_t *zero_addr = huffman_encode_output->zero_bits;

    for (int b_id = 0; b_id < huffman_encode_config->num_blocks; b_id++) {
        UJCOEF *values_ptr = output_addr;
        UJCOEF *diff_values_ptr = output_addr + 64;

        const unsigned char *my_huffman_encode_consts = huffman_encode_consts;

        for (int i = 0; i < 4; i++) {
            int16x8_t coefs1 = vld1q_dup_s16(input_addr + my_huffman_encode_consts[0]);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[1], coefs1, 1);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[2], coefs1, 2);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[3], coefs1, 3);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[4], coefs1, 4);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[5], coefs1, 5);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[6], coefs1, 6);
            coefs1 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[7], coefs1, 7);
            int16x8_t coefs2 = vld1q_dup_s16(input_addr + my_huffman_encode_consts[8]);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[9], coefs2, 1);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[10], coefs2, 2);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[11], coefs2, 3);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[12], coefs2, 4);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[13], coefs2, 5);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[14], coefs2, 6);
            coefs2 = vld1q_lane_s16(input_addr + my_huffman_encode_consts[15], coefs2, 7);

            /* Isolate sign of coefficients. */
            uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
            uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
            /* Compute absolute value of coefficients and apply point transform 4. */
            uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
            uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
            abs_coefs1 = vshlq_u16(abs_coefs1, vdupq_n_s16(-4));
            abs_coefs2 = vshlq_u16(abs_coefs2, vdupq_n_s16(-4));

            /* Compute diff values. */
            uint16x8_t diff1 = veorq_u16(abs_coefs1, sign_coefs1);
            uint16x8_t diff2 = veorq_u16(abs_coefs2, sign_coefs2);

            /* Store transformed coefficients and diff values. */
            vst1q_u16(values_ptr, abs_coefs1);
            vst1q_u16(values_ptr + 8, abs_coefs2);
            vst1q_u16(diff_values_ptr, diff1);
            vst1q_u16(diff_values_ptr + 8, diff2);
            values_ptr += 16;
            diff_values_ptr += 16;
            my_huffman_encode_consts += 16;
        }

        /* Construct zerobits bitmap.  A set bit means that the corresponding
        * coefficient != 0.
        */
        uint16x8_t row0 = vld1q_u16(values_ptr + 0 * 8);
        uint16x8_t row1 = vld1q_u16(values_ptr + 1 * 8);
        uint16x8_t row2 = vld1q_u16(values_ptr + 2 * 8);
        uint16x8_t row3 = vld1q_u16(values_ptr + 3 * 8);
        uint16x8_t row4 = vld1q_u16(values_ptr + 4 * 8);
        uint16x8_t row5 = vld1q_u16(values_ptr + 5 * 8);
        uint16x8_t row6 = vld1q_u16(values_ptr + 6 * 8);
        uint16x8_t row7 = vld1q_u16(values_ptr + 7 * 8);

        uint8x8_t row0_eq0 = vmovn_u16(vceqq_u16(row0, vdupq_n_u16(0)));
        uint8x8_t row1_eq0 = vmovn_u16(vceqq_u16(row1, vdupq_n_u16(0)));
        uint8x8_t row2_eq0 = vmovn_u16(vceqq_u16(row2, vdupq_n_u16(0)));
        uint8x8_t row3_eq0 = vmovn_u16(vceqq_u16(row3, vdupq_n_u16(0)));
        uint8x8_t row4_eq0 = vmovn_u16(vceqq_u16(row4, vdupq_n_u16(0)));
        uint8x8_t row5_eq0 = vmovn_u16(vceqq_u16(row5, vdupq_n_u16(0)));
        uint8x8_t row6_eq0 = vmovn_u16(vceqq_u16(row6, vdupq_n_u16(0)));
        uint8x8_t row7_eq0 = vmovn_u16(vceqq_u16(row7, vdupq_n_u16(0)));

        /* { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 } */
        const uint8x8_t bitmap_mask =
            vreinterpret_u8_u64(vmov_n_u64(0x8040201008040201));

        row0_eq0 = vand_u8(row0_eq0, bitmap_mask);
        row1_eq0 = vand_u8(row1_eq0, bitmap_mask);
        row2_eq0 = vand_u8(row2_eq0, bitmap_mask);
        row3_eq0 = vand_u8(row3_eq0, bitmap_mask);
        row4_eq0 = vand_u8(row4_eq0, bitmap_mask);
        row5_eq0 = vand_u8(row5_eq0, bitmap_mask);
        row6_eq0 = vand_u8(row6_eq0, bitmap_mask);
        row7_eq0 = vand_u8(row7_eq0, bitmap_mask);

        uint8x8_t bitmap_rows_01 = vpadd_u8(row0_eq0, row1_eq0);
        uint8x8_t bitmap_rows_23 = vpadd_u8(row2_eq0, row3_eq0);
        uint8x8_t bitmap_rows_45 = vpadd_u8(row4_eq0, row5_eq0);
        uint8x8_t bitmap_rows_67 = vpadd_u8(row6_eq0, row7_eq0);
        uint8x8_t bitmap_rows_0123 = vpadd_u8(bitmap_rows_01, bitmap_rows_23);
        uint8x8_t bitmap_rows_4567 = vpadd_u8(bitmap_rows_45, bitmap_rows_67);
        uint8x8_t bitmap_all = vpadd_u8(bitmap_rows_0123, bitmap_rows_4567);

        /* Move bitmap to two 32-bit scalar registers. */
        uint32_t bitmap0 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 0);
        uint32_t bitmap1 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 1);
        /* Store zerobits bitmap. */
        zero_addr[0] = ~bitmap0;
        zero_addr[1] = ~bitmap1;

        input_addr += 64;
        output_addr += 128;
        zero_addr += 2;
    }
}
