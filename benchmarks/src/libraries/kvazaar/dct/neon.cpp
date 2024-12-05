#include "dct.hpp"
#include "kvazaar.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

#define SHIFT_P1 6 // shift_1st
#define SHIFT_P2 9 // shift_2nd

void dct_8x8_neon(const int16_t *input, int16_t *output);

void dct_neon(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t *dct_input = (dct_input_t *)input;
    dct_output_t *dct_output = (dct_output_t *)output;

    int count = dct_config->count;
    int8_t *bitdepth = dct_config->bitdepth;
    int16_t *in = dct_input->input;
    int16_t *out = dct_output->output;
    int32_t shift_1st = kvz_g_convert_to_bit[8] + 1 + (bitdepth[0] - 8);
    assert(shift_1st == SHIFT_P1);
    int32_t shift_2nd = kvz_g_convert_to_bit[8] + 8;
    assert(shift_2nd == SHIFT_P2);

    for (int i = 0; i < count; i++) {
        dct_8x8_neon(in, out);
        in += 64;
        out += 64;
    }
}

void dct_8x8_neon(const int16_t *input, int16_t *output) {
// NOTE: we remove the right half matrix since it is not used anyway
#if defined(__clang__) || defined(_MSC_VER)
    const int32x4x4_t coeffs_l = vld1q_s32_x4(&kvz_g_dct_4_s32_2D[0][0]);
    const int32x4x4_t coeffs_h = vld1q_s32_x4(&kvz_g_dct_4_s32_2D[4][0]);
#else
    /* GCC does not currently support the intrinsic vld1q_<type>_x4(). */
    const int32x4_t coeff1 = vld1q_s16(&kvz_g_dct_4_s32_2D[0][0]);
    const int32x4_t coeff2 = vld1q_s16(&kvz_g_dct_4_s32_2D[1][0]);
    const int32x4_t coeff3 = vld1q_s16(&kvz_g_dct_4_s32_2D[2][0]);
    const int32x4_t coeff4 = vld1q_s16(&kvz_g_dct_4_s32_2D[3][0]);
    const int32x4_t coeff5 = vld1q_s16(&kvz_g_dct_4_s32_2D[4][0]);
    const int32x4_t coeff6 = vld1q_s16(&kvz_g_dct_4_s32_2D[5][0]);
    const int32x4_t coeff7 = vld1q_s16(&kvz_g_dct_4_s32_2D[6][0]);
    const int32x4_t coeff8 = vld1q_s16(&kvz_g_dct_4_s32_2D[7][0]);
    const int32x4x4_t coeffs_l = {coeff1, coeff2, coeff3, coeff4};
    const int32x4x4_t coeffs_h = {coeff5, coeff6, coeff7, coeff8};
#endif
    /* NOTE: load 8x8 samples into Q Regs, transpose the block such that we have a */
    /* column of samples per vector - allowing all rows to be processed at */
    /* once. */
    int16x8x4_t s_rows_0123 = vld4q_s16(input);
    int16x8x4_t s_rows_4567 = vld4q_s16(input + 4 * 8);
    int16x8x2_t cols_04 = vuzpq_s16(s_rows_0123.val[0], s_rows_4567.val[0]);
    int16x8x2_t cols_15 = vuzpq_s16(s_rows_0123.val[1], s_rows_4567.val[1]);
    int16x8x2_t cols_26 = vuzpq_s16(s_rows_0123.val[2], s_rows_4567.val[2]);
    int16x8x2_t cols_37 = vuzpq_s16(s_rows_0123.val[3], s_rows_4567.val[3]);
    int16x8_t col0 = cols_04.val[0];
    int16x8_t col1 = cols_15.val[0];
    int16x8_t col2 = cols_26.val[0];
    int16x8_t col3 = cols_37.val[0];
    int16x8_t col4 = cols_04.val[1];
    int16x8_t col5 = cols_15.val[1];
    int16x8_t col6 = cols_26.val[1];
    int16x8_t col7 = cols_37.val[1];
    /* Step 1: row-wise transform. */
    // NOTE: regex schema: int16x8_t ((\w|_){3,4}) = v((add|sub))q_s16\(((\w|_){3,4}), ((\w|_){3,4})\);
    int32x4_t e_0_l = vaddl_s16(vget_low_s16(col0), vget_low_s16(col7));
    int32x4_t e_0_h = vaddl_high_s16(col0, col7);
    int32x4_t o_0_l = vsubl_s16(vget_low_s16(col0), vget_low_s16(col7));
    int32x4_t o_0_h = vsubl_high_s16(col0, col7);
    int32x4_t e_1_l = vaddl_s16(vget_low_s16(col1), vget_low_s16(col6));
    int32x4_t e_1_h = vaddl_high_s16(col1, col6);
    int32x4_t o_1_l = vsubl_s16(vget_low_s16(col1), vget_low_s16(col6));
    int32x4_t o_1_h = vsubl_high_s16(col1, col6);
    int32x4_t e_2_l = vaddl_s16(vget_low_s16(col2), vget_low_s16(col5));
    int32x4_t e_2_h = vaddl_high_s16(col2, col5);
    int32x4_t o_2_l = vsubl_s16(vget_low_s16(col2), vget_low_s16(col5));
    int32x4_t o_2_h = vsubl_high_s16(col2, col5);
    int32x4_t e_3_l = vaddl_s16(vget_low_s16(col3), vget_low_s16(col4));
    int32x4_t e_3_h = vaddl_high_s16(col3, col4);
    int32x4_t o_3_l = vsubl_s16(vget_low_s16(col3), vget_low_s16(col4));
    int32x4_t o_3_h = vsubl_high_s16(col3, col4);
    int32x4_t ee_0_l = vaddq_s32(e_0_l, e_3_l);
    int32x4_t ee_0_h = vaddq_s32(e_0_h, e_3_h);
    int32x4_t eo_0_l = vsubq_s32(e_0_l, e_3_l);
    int32x4_t eo_0_h = vsubq_s32(e_0_h, e_3_h);
    int32x4_t ee_1_l = vaddq_s32(e_1_l, e_2_l);
    int32x4_t ee_1_h = vaddq_s32(e_1_h, e_2_h);
    int32x4_t eo_1_l = vsubq_s32(e_1_l, e_2_l);
    int32x4_t eo_1_h = vsubq_s32(e_1_h, e_2_h);
    // ee cols
    col0 = vcombine_s16(vrshrn_n_s32(vshlq_n_s32(vaddq_s32(ee_0_l, ee_1_l), 6), SHIFT_P1),
                        vrshrn_n_s32(vshlq_n_s32(vaddq_s32(ee_0_h, ee_1_h), 6), SHIFT_P1));
    col4 = vcombine_s32(vrshrn_n_s32(vshlq_n_s32(vsubq_s32(ee_0_l, ee_1_l), 6), SHIFT_P1),
                        vrshrn_n_s32(vshlq_n_s32(vsubq_s32(ee_0_h, ee_1_h), 6), SHIFT_P1));
    // eo cols
    int32x4_t col2_l = vmulq_laneq_s32(eo_0_l, coeffs_l.val[2], 0);
    int32x4_t col2_h = vmulq_laneq_s32(eo_0_h, coeffs_l.val[2], 0);
    col2_l = vmlaq_laneq_s32(col2_l, eo_1_l, coeffs_l.val[2], 1);
    col2_h = vmlaq_laneq_s32(col2_h, eo_1_h, coeffs_l.val[2], 1);
    col2 = vcombine_s16(vrshrn_n_s32(col2_l, SHIFT_P1), vrshrn_n_s32(col2_h, SHIFT_P1));
    int32x4_t col6_l = vmulq_laneq_s32(eo_0_l, coeffs_h.val[2], 0);
    int32x4_t col6_h = vmulq_laneq_s32(eo_0_h, coeffs_h.val[2], 0);
    col6_l = vmlaq_laneq_s32(col6_l, eo_1_l, coeffs_h.val[2], 1);
    col6_h = vmlaq_laneq_s32(col6_h, eo_1_h, coeffs_h.val[2], 1);
    col6 = vcombine_s16(vrshrn_n_s32(col6_l, SHIFT_P1), vrshrn_n_s32(col6_h, SHIFT_P1));
    // o cols
    int32x4_t col1_l = vmulq_laneq_s32(o_0_l, coeffs_l.val[1], 0);
    int32x4_t col1_h = vmulq_laneq_s32(o_0_h, coeffs_l.val[1], 0);
    col1_l = vmlaq_laneq_s32(col1_l, o_1_l, coeffs_l.val[1], 1);
    col1_h = vmlaq_laneq_s32(col1_h, o_1_h, coeffs_l.val[1], 1);
    col1_l = vmlaq_laneq_s32(col1_l, o_2_l, coeffs_l.val[1], 2);
    col1_h = vmlaq_laneq_s32(col1_h, o_2_h, coeffs_l.val[1], 2);
    col1_l = vmlaq_laneq_s32(col1_l, o_3_l, coeffs_l.val[1], 3);
    col1_h = vmlaq_laneq_s32(col1_h, o_3_h, coeffs_l.val[1], 3);
    col1 = vcombine_s16(vrshrn_n_s32(col1_l, SHIFT_P1), vrshrn_n_s32(col1_h, SHIFT_P1));
    int32x4_t col3_l = vmulq_laneq_s32(o_0_l, coeffs_l.val[3], 0);
    int32x4_t col3_h = vmulq_laneq_s32(o_0_h, coeffs_l.val[3], 0);
    col3_l = vmlaq_laneq_s32(col3_l, o_1_l, coeffs_l.val[3], 1);
    col3_h = vmlaq_laneq_s32(col3_h, o_1_h, coeffs_l.val[3], 1);
    col3_l = vmlaq_laneq_s32(col3_l, o_2_l, coeffs_l.val[3], 2);
    col3_h = vmlaq_laneq_s32(col3_h, o_2_h, coeffs_l.val[3], 2);
    col3_l = vmlaq_laneq_s32(col3_l, o_3_l, coeffs_l.val[3], 3);
    col3_h = vmlaq_laneq_s32(col3_h, o_3_h, coeffs_l.val[3], 3);
    col3 = vcombine_s16(vrshrn_n_s32(col3_l, SHIFT_P1), vrshrn_n_s32(col3_h, SHIFT_P1));
    int32x4_t col5_l = vmulq_laneq_s32(o_0_l, coeffs_h.val[1], 0);
    int32x4_t col5_h = vmulq_laneq_s32(o_0_h, coeffs_h.val[1], 0);
    col5_l = vmlaq_laneq_s32(col5_l, o_1_l, coeffs_h.val[1], 1);
    col5_h = vmlaq_laneq_s32(col5_h, o_1_h, coeffs_h.val[1], 1);
    col5_l = vmlaq_laneq_s32(col5_l, o_2_l, coeffs_h.val[1], 2);
    col5_h = vmlaq_laneq_s32(col5_h, o_2_h, coeffs_h.val[1], 2);
    col5_l = vmlaq_laneq_s32(col5_l, o_3_l, coeffs_h.val[1], 3);
    col5_h = vmlaq_laneq_s32(col5_h, o_3_h, coeffs_h.val[1], 3);
    col5 = vcombine_s16(vrshrn_n_s32(col5_l, SHIFT_P1), vrshrn_n_s32(col5_h, SHIFT_P1));
    int32x4_t col7_l = vmulq_laneq_s32(o_0_l, coeffs_h.val[3], 0);
    int32x4_t col7_h = vmulq_laneq_s32(o_0_h, coeffs_h.val[3], 0);
    col7_l = vmlaq_laneq_s32(col7_l, o_1_l, coeffs_h.val[3], 1);
    col7_h = vmlaq_laneq_s32(col7_h, o_1_h, coeffs_h.val[3], 1);
    col7_l = vmlaq_laneq_s32(col7_l, o_2_l, coeffs_h.val[3], 2);
    col7_h = vmlaq_laneq_s32(col7_h, o_2_h, coeffs_h.val[3], 2);
    col7_l = vmlaq_laneq_s32(col7_l, o_3_l, coeffs_h.val[3], 3);
    col7_h = vmlaq_laneq_s32(col7_h, o_3_h, coeffs_h.val[3], 3);
    col7 = vcombine_s16(vrshrn_n_s32(col7_l, SHIFT_P1), vrshrn_n_s32(col7_h, SHIFT_P1));

    /* Step 2: transpose to work on columns in step 3. */
    int16x8x2_t cols_01 = vtrnq_s16(col0, col1);
    int16x8x2_t cols_23 = vtrnq_s16(col2, col3);
    int16x8x2_t cols_45 = vtrnq_s16(col4, col5);
    int16x8x2_t cols_67 = vtrnq_s16(col6, col7);
    int32x4x2_t cols_0145_l = vtrnq_s32(vreinterpretq_s32_s16(cols_01.val[0]), vreinterpretq_s32_s16(cols_45.val[0]));
    int32x4x2_t cols_0145_h = vtrnq_s32(vreinterpretq_s32_s16(cols_01.val[1]), vreinterpretq_s32_s16(cols_45.val[1]));
    int32x4x2_t cols_2367_l = vtrnq_s32(vreinterpretq_s32_s16(cols_23.val[0]), vreinterpretq_s32_s16(cols_67.val[0]));
    int32x4x2_t cols_2367_h = vtrnq_s32(vreinterpretq_s32_s16(cols_23.val[1]), vreinterpretq_s32_s16(cols_67.val[1]));
    int32x4x2_t rows_04 = vzipq_s32(cols_0145_l.val[0], cols_2367_l.val[0]);
    int32x4x2_t rows_15 = vzipq_s32(cols_0145_h.val[0], cols_2367_h.val[0]);
    int32x4x2_t rows_26 = vzipq_s32(cols_0145_l.val[1], cols_2367_l.val[1]);
    int32x4x2_t rows_37 = vzipq_s32(cols_0145_h.val[1], cols_2367_h.val[1]);
    int16x8_t row0 = vreinterpretq_s16_s32(rows_04.val[0]);
    int16x8_t row1 = vreinterpretq_s16_s32(rows_15.val[0]);
    int16x8_t row2 = vreinterpretq_s16_s32(rows_26.val[0]);
    int16x8_t row3 = vreinterpretq_s16_s32(rows_37.val[0]);
    int16x8_t row4 = vreinterpretq_s16_s32(rows_04.val[1]);
    int16x8_t row5 = vreinterpretq_s16_s32(rows_15.val[1]);
    int16x8_t row6 = vreinterpretq_s16_s32(rows_26.val[1]);
    int16x8_t row7 = vreinterpretq_s16_s32(rows_37.val[1]);

    /* Step 3: col-wise transform. */
    e_0_l = vaddl_s16(vget_low_s16(row0), vget_low_s16(row7));
    e_0_h = vaddl_high_s16(row0, row7);
    o_0_l = vsubl_s16(vget_low_s16(row0), vget_low_s16(row7));
    o_0_h = vsubl_high_s16(row0, row7);
    e_1_l = vaddl_s16(vget_low_s16(row1), vget_low_s16(row6));
    e_1_h = vaddl_high_s16(row1, row6);
    o_1_l = vsubl_s16(vget_low_s16(row1), vget_low_s16(row6));
    o_1_h = vsubl_high_s16(row1, row6);
    e_2_l = vaddl_s16(vget_low_s16(row2), vget_low_s16(row5));
    e_2_h = vaddl_high_s16(row2, row5);
    o_2_l = vsubl_s16(vget_low_s16(row2), vget_low_s16(row5));
    o_2_h = vsubl_high_s16(row2, row5);
    e_3_l = vaddl_s16(vget_low_s16(row3), vget_low_s16(row4));
    e_3_h = vaddl_high_s16(row3, row4);
    o_3_l = vsubl_s16(vget_low_s16(row3), vget_low_s16(row4));
    o_3_h = vsubl_high_s16(row3, row4);
    ee_0_l = vaddq_s32(e_0_l, e_3_l);
    ee_0_h = vaddq_s32(e_0_h, e_3_h);
    eo_0_l = vsubq_s32(e_0_l, e_3_l);
    eo_0_h = vsubq_s32(e_0_h, e_3_h);
    ee_1_l = vaddq_s32(e_1_l, e_2_l);
    ee_1_h = vaddq_s32(e_1_h, e_2_h);
    eo_1_l = vsubq_s32(e_1_l, e_2_l);
    eo_1_h = vsubq_s32(e_1_h, e_2_h);
    // ee rows
    row0 = vcombine_s16(vrshrn_n_s32(vshlq_n_s32(vaddq_s32(ee_0_l, ee_1_l), 6), SHIFT_P2),
                        vrshrn_n_s32(vshlq_n_s32(vaddq_s32(ee_0_h, ee_1_h), 6), SHIFT_P2));
    row4 = vcombine_s32(vrshrn_n_s32(vshlq_n_s32(vsubq_s32(ee_0_l, ee_1_l), 6), SHIFT_P2),
                        vrshrn_n_s32(vshlq_n_s32(vsubq_s32(ee_0_h, ee_1_h), 6), SHIFT_P2));
    // eo rows
    int32x4_t row2_l = vmulq_laneq_s32(eo_0_l, coeffs_l.val[2], 0);
    int32x4_t row2_h = vmulq_laneq_s32(eo_0_h, coeffs_l.val[2], 0);
    row2_l = vmlaq_laneq_s32(row2_l, eo_1_l, coeffs_l.val[2], 1);
    row2_h = vmlaq_laneq_s32(row2_h, eo_1_h, coeffs_l.val[2], 1);
    row2 = vcombine_s16(vrshrn_n_s32(row2_l, SHIFT_P2), vrshrn_n_s32(row2_h, SHIFT_P2));
    int32x4_t row6_l = vmulq_laneq_s32(eo_0_l, coeffs_h.val[2], 0);
    int32x4_t row6_h = vmulq_laneq_s32(eo_0_h, coeffs_h.val[2], 0);
    row6_l = vmlaq_laneq_s32(row6_l, eo_1_l, coeffs_h.val[2], 1);
    row6_h = vmlaq_laneq_s32(row6_h, eo_1_h, coeffs_h.val[2], 1);
    row6 = vcombine_s16(vrshrn_n_s32(row6_l, SHIFT_P2), vrshrn_n_s32(row6_h, SHIFT_P2));
    // o rows
    int32x4_t row1_l = vmulq_laneq_s32(o_0_l, coeffs_l.val[1], 0);
    int32x4_t row1_h = vmulq_laneq_s32(o_0_h, coeffs_l.val[1], 0);
    row1_l = vmlaq_laneq_s32(row1_l, o_1_l, coeffs_l.val[1], 1);
    row1_h = vmlaq_laneq_s32(row1_h, o_1_h, coeffs_l.val[1], 1);
    row1_l = vmlaq_laneq_s32(row1_l, o_2_l, coeffs_l.val[1], 2);
    row1_h = vmlaq_laneq_s32(row1_h, o_2_h, coeffs_l.val[1], 2);
    row1_l = vmlaq_laneq_s32(row1_l, o_3_l, coeffs_l.val[1], 3);
    row1_h = vmlaq_laneq_s32(row1_h, o_3_h, coeffs_l.val[1], 3);
    row1 = vcombine_s16(vrshrn_n_s32(row1_l, SHIFT_P2), vrshrn_n_s32(row1_h, SHIFT_P2));
    int32x4_t row3_l = vmulq_laneq_s32(o_0_l, coeffs_l.val[3], 0);
    int32x4_t row3_h = vmulq_laneq_s32(o_0_h, coeffs_l.val[3], 0);
    row3_l = vmlaq_laneq_s32(row3_l, o_1_l, coeffs_l.val[3], 1);
    row3_h = vmlaq_laneq_s32(row3_h, o_1_h, coeffs_l.val[3], 1);
    row3_l = vmlaq_laneq_s32(row3_l, o_2_l, coeffs_l.val[3], 2);
    row3_h = vmlaq_laneq_s32(row3_h, o_2_h, coeffs_l.val[3], 2);
    row3_l = vmlaq_laneq_s32(row3_l, o_3_l, coeffs_l.val[3], 3);
    row3_h = vmlaq_laneq_s32(row3_h, o_3_h, coeffs_l.val[3], 3);
    row3 = vcombine_s16(vrshrn_n_s32(row3_l, SHIFT_P2), vrshrn_n_s32(row3_h, SHIFT_P2));
    int32x4_t row5_l = vmulq_laneq_s32(o_0_l, coeffs_h.val[1], 0);
    int32x4_t row5_h = vmulq_laneq_s32(o_0_h, coeffs_h.val[1], 0);
    row5_l = vmlaq_laneq_s32(row5_l, o_1_l, coeffs_h.val[1], 1);
    row5_h = vmlaq_laneq_s32(row5_h, o_1_h, coeffs_h.val[1], 1);
    row5_l = vmlaq_laneq_s32(row5_l, o_2_l, coeffs_h.val[1], 2);
    row5_h = vmlaq_laneq_s32(row5_h, o_2_h, coeffs_h.val[1], 2);
    row5_l = vmlaq_laneq_s32(row5_l, o_3_l, coeffs_h.val[1], 3);
    row5_h = vmlaq_laneq_s32(row5_h, o_3_h, coeffs_h.val[1], 3);
    row5 = vcombine_s16(vrshrn_n_s32(row5_l, SHIFT_P2), vrshrn_n_s32(row5_h, SHIFT_P2));
    int32x4_t row7_l = vmulq_laneq_s32(o_0_l, coeffs_h.val[3], 0);
    int32x4_t row7_h = vmulq_laneq_s32(o_0_h, coeffs_h.val[3], 0);
    row7_l = vmlaq_laneq_s32(row7_l, o_1_l, coeffs_h.val[3], 1);
    row7_h = vmlaq_laneq_s32(row7_h, o_1_h, coeffs_h.val[3], 1);
    row7_l = vmlaq_laneq_s32(row7_l, o_2_l, coeffs_h.val[3], 2);
    row7_h = vmlaq_laneq_s32(row7_h, o_2_h, coeffs_h.val[3], 2);
    row7_l = vmlaq_laneq_s32(row7_l, o_3_l, coeffs_h.val[3], 3);
    row7_h = vmlaq_laneq_s32(row7_h, o_3_h, coeffs_h.val[3], 3);
    row7 = vcombine_s16(vrshrn_n_s32(row7_l, SHIFT_P2), vrshrn_n_s32(row7_h, SHIFT_P2));

    /* Step 4: store the results. */
    vst1q_s16(output + 0 * 8, row0);
    vst1q_s16(output + 1 * 8, row1);
    vst1q_s16(output + 2 * 8, row2);
    vst1q_s16(output + 3 * 8, row3);
    vst1q_s16(output + 4 * 8, row4);
    vst1q_s16(output + 5 * 8, row5);
    vst1q_s16(output + 6 * 8, row6);
    vst1q_s16(output + 7 * 8, row7);
}