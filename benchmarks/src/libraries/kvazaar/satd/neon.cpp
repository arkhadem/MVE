#include "kvazaar.hpp"
#include "neon_kernels.hpp"
#include "satd.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

int32_t satd_8x8_subblock_neon(const uint8_t *piOrg, const uint8_t *piCur);

void satd_neon(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {

    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t *satd_input = (satd_input_t *)input;
    satd_output_t *satd_output = (satd_output_t *)output;

    int count = satd_config->count;
    uint8_t *piOrg = satd_input->piOrg;
    uint8_t *piCur = satd_input->piCur;
    int32_t *result = satd_output->result;

    for (int __i = 0; __i < count; __i++) {

        for (unsigned y = 0; y < 8; y += 8) {
            unsigned row = y * 8;
            for (unsigned x = 0; x < 8; x += 8) {
                *result = satd_8x8_subblock_neon(&piOrg[row + x], &piCur[row + x]);
                result += 1;
            }
        }
        piOrg += 64;
        piCur += 64;
    }
}

void satd_8x8_subblock_compute_neon(const uint8_t *piOrg, const uint8_t *piCur, int32x4x2_t *m2) {
    int32x4x2_t m1[8], m3[8];
    int16x8_t diff[8];

    for (int32_t k = 0; k < 64; k += 8) {
        int8x8_t v_piOrg = vld1_u8(piOrg);
        int8x8_t v_piCur = vld1_u8(piCur);
        diff[k / 8] = vsubl_u8(v_piOrg, v_piCur);
        piCur += 8;
        piOrg += 8;
    }
    // transpose
    int16x8x2_t rows_01 = vtrnq_s16(diff[0], diff[1]);
    int16x8x2_t rows_23 = vtrnq_s16(diff[2], diff[3]);
    int16x8x2_t rows_45 = vtrnq_s16(diff[4], diff[5]);
    int16x8x2_t rows_67 = vtrnq_s16(diff[6], diff[7]);
    int32x4x2_t rows_0145_l = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[0]), vreinterpretq_s32_s16(rows_45.val[0]));
    int32x4x2_t rows_0145_h = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[1]), vreinterpretq_s32_s16(rows_45.val[1]));
    int32x4x2_t rows_2367_l = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[0]), vreinterpretq_s32_s16(rows_67.val[0]));
    int32x4x2_t rows_2367_h = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[1]), vreinterpretq_s32_s16(rows_67.val[1]));
    int32x4x2_t cols_04 = vzipq_s32(rows_0145_l.val[0], rows_2367_l.val[0]);
    int32x4x2_t cols_15 = vzipq_s32(rows_0145_h.val[0], rows_2367_h.val[0]);
    int32x4x2_t cols_26 = vzipq_s32(rows_0145_l.val[1], rows_2367_l.val[1]);
    int32x4x2_t cols_37 = vzipq_s32(rows_0145_h.val[1], rows_2367_h.val[1]);
    diff[0] = vreinterpretq_s16_s32(cols_04.val[0]);
    diff[1] = vreinterpretq_s16_s32(cols_15.val[0]);
    diff[2] = vreinterpretq_s16_s32(cols_26.val[0]);
    diff[3] = vreinterpretq_s16_s32(cols_37.val[0]);
    diff[4] = vreinterpretq_s16_s32(cols_04.val[1]);
    diff[5] = vreinterpretq_s16_s32(cols_15.val[1]);
    diff[6] = vreinterpretq_s16_s32(cols_26.val[1]);
    diff[7] = vreinterpretq_s16_s32(cols_37.val[1]);

    // horizontal
    m2[0].val[0] = vaddl_s16(vget_low_s16(diff[0]), vget_low_s16(diff[4]));
    m2[0].val[1] = vaddl_high_s16(diff[0], diff[4]);
    m2[1].val[0] = vaddl_s16(vget_low_s16(diff[1]), vget_low_s16(diff[5]));
    m2[1].val[1] = vaddl_high_s16(diff[1], diff[5]);
    m2[2].val[0] = vaddl_s16(vget_low_s16(diff[2]), vget_low_s16(diff[6]));
    m2[2].val[1] = vaddl_high_s16(diff[2], diff[6]);
    m2[3].val[0] = vaddl_s16(vget_low_s16(diff[3]), vget_low_s16(diff[7]));
    m2[3].val[1] = vaddl_high_s16(diff[3], diff[7]);
    m2[4].val[0] = vsubl_s16(vget_low_s16(diff[0]), vget_low_s16(diff[4]));
    m2[4].val[1] = vsubl_high_s16(diff[0], diff[4]);
    m2[5].val[0] = vsubl_s16(vget_low_s16(diff[1]), vget_low_s16(diff[5]));
    m2[5].val[1] = vsubl_high_s16(diff[1], diff[5]);
    m2[6].val[0] = vsubl_s16(vget_low_s16(diff[2]), vget_low_s16(diff[6]));
    m2[6].val[1] = vsubl_high_s16(diff[2], diff[6]);
    m2[7].val[0] = vsubl_s16(vget_low_s16(diff[3]), vget_low_s16(diff[7]));
    m2[7].val[1] = vsubl_high_s16(diff[3], diff[7]);

#pragma unroll(2)
    for (int p = 0; p < 2; p++) {
        m1[0].val[p] = vaddq_s32(m2[0].val[p], m2[2].val[p]);
        m1[1].val[p] = vaddq_s32(m2[1].val[p], m2[3].val[p]);
        m1[2].val[p] = vsubq_s32(m2[0].val[p], m2[2].val[p]);
        m1[3].val[p] = vsubq_s32(m2[1].val[p], m2[3].val[p]);
        m1[4].val[p] = vaddq_s32(m2[4].val[p], m2[6].val[p]);
        m1[5].val[p] = vaddq_s32(m2[5].val[p], m2[7].val[p]);
        m1[6].val[p] = vsubq_s32(m2[4].val[p], m2[6].val[p]);
        m1[7].val[p] = vsubq_s32(m2[5].val[p], m2[7].val[p]);

        m2[0].val[p] = vaddq_s32(m1[0].val[p], m1[1].val[p]);
        m2[1].val[p] = vsubq_s32(m1[0].val[p], m1[1].val[p]);
        m2[2].val[p] = vaddq_s32(m1[2].val[p], m1[3].val[p]);
        m2[3].val[p] = vsubq_s32(m1[2].val[p], m1[3].val[p]);
        m2[4].val[p] = vaddq_s32(m1[4].val[p], m1[5].val[p]);
        m2[5].val[p] = vsubq_s32(m1[4].val[p], m1[5].val[p]);
        m2[6].val[p] = vaddq_s32(m1[6].val[p], m1[7].val[p]);
        m2[7].val[p] = vsubq_s32(m1[6].val[p], m1[7].val[p]);
    }

    // transpose
    int32x4x2_t cols_01_l = vtrnq_s32(m2[0].val[0], m2[1].val[0]);
    int32x4x2_t cols_01_h = vtrnq_s32(m2[0].val[1], m2[1].val[1]);
    int32x4x2_t cols_23_l = vtrnq_s32(m2[2].val[0], m2[3].val[0]);
    int32x4x2_t cols_23_h = vtrnq_s32(m2[2].val[1], m2[3].val[1]);
    int32x4x2_t cols_45_l = vtrnq_s32(m2[4].val[0], m2[5].val[0]);
    int32x4x2_t cols_45_h = vtrnq_s32(m2[4].val[1], m2[5].val[1]);
    int32x4x2_t cols_67_l = vtrnq_s32(m2[6].val[0], m2[7].val[0]);
    int32x4x2_t cols_67_h = vtrnq_s32(m2[6].val[1], m2[7].val[1]);
    m2[0].val[0] = vtrn1q_s64(cols_01_l.val[0], cols_23_l.val[0]);
    m2[0].val[1] = vtrn1q_s64(cols_45_l.val[0], cols_67_l.val[0]);
    m2[2].val[0] = vtrn2q_s64(cols_01_l.val[0], cols_23_l.val[0]);
    m2[2].val[1] = vtrn2q_s64(cols_45_l.val[0], cols_67_l.val[0]);
    m2[1].val[0] = vtrn1q_s64(cols_01_l.val[1], cols_23_l.val[1]);
    m2[1].val[1] = vtrn1q_s64(cols_45_l.val[1], cols_67_l.val[1]);
    m2[3].val[0] = vtrn2q_s64(cols_01_l.val[1], cols_23_l.val[1]);
    m2[3].val[1] = vtrn2q_s64(cols_45_l.val[1], cols_67_l.val[1]);
    m2[4].val[0] = vtrn1q_s64(cols_01_h.val[0], cols_23_h.val[0]);
    m2[4].val[1] = vtrn1q_s64(cols_45_h.val[0], cols_67_h.val[0]);
    m2[6].val[0] = vtrn2q_s64(cols_01_h.val[0], cols_23_h.val[0]);
    m2[6].val[1] = vtrn2q_s64(cols_45_h.val[0], cols_67_h.val[0]);
    m2[5].val[0] = vtrn1q_s64(cols_01_h.val[1], cols_23_h.val[1]);
    m2[5].val[1] = vtrn1q_s64(cols_45_h.val[1], cols_67_h.val[1]);
    m2[7].val[0] = vtrn2q_s64(cols_01_h.val[1], cols_23_h.val[1]);
    m2[7].val[1] = vtrn2q_s64(cols_45_h.val[1], cols_67_h.val[1]);

    // vertical

#pragma unroll(2)
    for (int p = 0; p < 2; p++) {
        m3[0].val[p] = vaddq_s32(m2[0].val[p], m2[4].val[p]);
        m3[1].val[p] = vaddq_s32(m2[1].val[p], m2[5].val[p]);
        m3[2].val[p] = vaddq_s32(m2[2].val[p], m2[6].val[p]);
        m3[3].val[p] = vaddq_s32(m2[3].val[p], m2[7].val[p]);
        m3[4].val[p] = vsubq_s32(m2[0].val[p], m2[4].val[p]);
        m3[5].val[p] = vsubq_s32(m2[1].val[p], m2[5].val[p]);
        m3[6].val[p] = vsubq_s32(m2[2].val[p], m2[6].val[p]);
        m3[7].val[p] = vsubq_s32(m2[3].val[p], m2[7].val[p]);

        m1[0].val[p] = vaddq_s32(m3[0].val[p], m3[2].val[p]);
        m1[1].val[p] = vaddq_s32(m3[1].val[p], m3[3].val[p]);
        m1[2].val[p] = vsubq_s32(m3[0].val[p], m3[2].val[p]);
        m1[3].val[p] = vsubq_s32(m3[1].val[p], m3[3].val[p]);
        m1[4].val[p] = vaddq_s32(m3[4].val[p], m3[6].val[p]);
        m1[5].val[p] = vaddq_s32(m3[5].val[p], m3[7].val[p]);
        m1[6].val[p] = vsubq_s32(m3[4].val[p], m3[6].val[p]);
        m1[7].val[p] = vsubq_s32(m3[5].val[p], m3[7].val[p]);

        m2[0].val[p] = vaddq_s32(m1[0].val[p], m1[1].val[p]);
        m2[1].val[p] = vsubq_s32(m1[0].val[p], m1[1].val[p]);
        m2[2].val[p] = vaddq_s32(m1[2].val[p], m1[3].val[p]);
        m2[3].val[p] = vsubq_s32(m1[2].val[p], m1[3].val[p]);
        m2[4].val[p] = vaddq_s32(m1[4].val[p], m1[5].val[p]);
        m2[5].val[p] = vsubq_s32(m1[4].val[p], m1[5].val[p]);
        m2[6].val[p] = vaddq_s32(m1[6].val[p], m1[7].val[p]);
        m2[7].val[p] = vsubq_s32(m1[6].val[p], m1[7].val[p]);
    }
}

int32_t satd_8x8_subblock_reduce_neon(const int32x4x2_t *m2) {
    int32x4_t buf0;
    buf0 = vaddq_s32(m2[0].val[0], m2[0].val[1]);
    return (vaddvq_s32(buf0) + 2) >> 2;
}

int32_t satd_8x8_subblock_neon(const uint8_t *piOrg, const uint8_t *piCur) {
    int32x4x2_t m2[8];
    satd_8x8_subblock_compute_neon(piOrg, piCur, m2);
    return satd_8x8_subblock_reduce_neon(m2);
}