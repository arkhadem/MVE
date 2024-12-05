#include "idct.hpp"
#include "kvazaar.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

#define SHIFT_P1 7  // shift_1st
#define SHIFT_P2 12 // shift_2nd

void idct_8x8_neon(const int16_t *input, int16_t *output);

void idct_neon(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {

    idct_config_t *idct_config = (idct_config_t *)config;
    idct_input_t *idct_input = (idct_input_t *)input;
    idct_output_t *idct_output = (idct_output_t *)output;

    int count = idct_config->count;
    int8_t *bitdepth = idct_config->bitdepth;
    int16_t *in = idct_input->input;
    int16_t *out = idct_output->output;
    int32_t shift_1st = 7;
    assert(shift_1st == SHIFT_P1);
    int32_t shift_2nd = 12 - (bitdepth[0] - 8);
    assert(shift_2nd == SHIFT_P2);

    for (int i = 0; i < count; i++) {
        idct_8x8_neon(in, out);
        in += 64;
        out += 64;
    }
}

void idct_8x8_neon(const int16_t *input, int16_t *output) {
    int32x4_t coeffs[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        coeffs[i] = vld1q_s32(&kvz_g_dct_4_s32_2D[i][0]);
    }
    int32x4x2_t rows[8];
    int16x4x2_t tmp;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        tmp = vld1_s16_x2(input + i * 8);
        rows[i].val[0] = vmovl_s16(tmp.val[0]);
        rows[i].val[1] = vmovl_s16(tmp.val[1]);
    }

    /* Step 1: col-wise transform. */
    int32x4x2_t o[4];
#pragma unroll
    for (int p = 0; p < 2; p++) {
        o[0].val[p] = vmulq_laneq_s32(rows[1].val[p], coeffs[1], 0);
        o[1].val[p] = vmulq_laneq_s32(rows[1].val[p], coeffs[1], 1);
        o[2].val[p] = vmulq_laneq_s32(rows[1].val[p], coeffs[1], 2);
        o[3].val[p] = vmulq_laneq_s32(rows[1].val[p], coeffs[1], 3);
    }
#pragma unroll
    for (int rc_idx = 3; rc_idx < 8; rc_idx += 2) {
        for (int p = 0; p < 2; p++) {
            o[0].val[p] = vmlaq_laneq_s32(o[0].val[p], rows[rc_idx].val[p], coeffs[rc_idx], 0);
            o[1].val[p] = vmlaq_laneq_s32(o[1].val[p], rows[rc_idx].val[p], coeffs[rc_idx], 1);
            o[2].val[p] = vmlaq_laneq_s32(o[2].val[p], rows[rc_idx].val[p], coeffs[rc_idx], 2);
            o[3].val[p] = vmlaq_laneq_s32(o[3].val[p], rows[rc_idx].val[p], coeffs[rc_idx], 3);
        }
    }
    int32x4x2_t ee[2], eo[2];
#pragma unroll
    for (int p = 0; p < 2; p++) {
        eo[0].val[p] = vmulq_laneq_s32(rows[2].val[p], coeffs[2], 0);
        eo[0].val[p] = vmlaq_laneq_s32(eo[0].val[p], rows[6].val[p], coeffs[6], 0);
        eo[1].val[p] = vmulq_laneq_s32(rows[2].val[p], coeffs[2], 1);
        eo[1].val[p] = vmlaq_laneq_s32(eo[1].val[p], rows[6].val[p], coeffs[6], 1);
    }
#pragma unroll
    for (int p = 0; p < 2; p++) {
        ee[0].val[p] = vmulq_laneq_s32(rows[0].val[p], coeffs[0], 0);
        ee[0].val[p] = vmlaq_laneq_s32(ee[0].val[p], rows[4].val[p], coeffs[4], 0);
        ee[1].val[p] = vmulq_laneq_s32(rows[0].val[p], coeffs[0], 1);
        ee[1].val[p] = vmlaq_laneq_s32(ee[1].val[p], rows[4].val[p], coeffs[4], 1);
    }
    int32x4x2_t e[4];
#pragma unroll
    for (int p = 0; p < 2; p++) {
        e[0].val[p] = vaddq_s32(ee[0].val[p], eo[0].val[p]);
        e[1].val[p] = vaddq_s32(ee[1].val[p], eo[1].val[p]);
        e[2].val[p] = vsubq_s32(ee[1].val[p], eo[1].val[p]);
        e[3].val[p] = vsubq_s32(ee[0].val[p], eo[0].val[p]);
    }
    // NOTE: r_rows stands for narrowed rows
    int16x8_t r_rows[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        r_rows[k] = vcombine_s16(vqmovn_s32(vrshrq_n_s32(vaddq_s32(e[k].val[0], o[k].val[0]), SHIFT_P1)),
                                 vqmovn_s32(vrshrq_n_s32(vaddq_s32(e[k].val[1], o[k].val[1]), SHIFT_P1)));
        r_rows[k + 4] = vcombine_s16(vqmovn_s32(vrshrq_n_s32(vsubq_s32(e[3 - k].val[0], o[3 - k].val[0]), SHIFT_P1)),
                                     vqmovn_s32(vrshrq_n_s32(vsubq_s32(e[3 - k].val[1], o[3 - k].val[1]), SHIFT_P1)));
    }

    /* Step 2: transpose to work on columns in step 3. */
    int16x8x2_t rows_01 = vtrnq_s16(r_rows[0], r_rows[1]);
    int16x8x2_t rows_23 = vtrnq_s16(r_rows[2], r_rows[3]);
    int16x8x2_t rows_45 = vtrnq_s16(r_rows[4], r_rows[5]);
    int16x8x2_t rows_67 = vtrnq_s16(r_rows[6], r_rows[7]);
    int32x4x2_t rows_0145_l = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[0]), vreinterpretq_s32_s16(rows_45.val[0]));
    int32x4x2_t rows_0145_h = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[1]), vreinterpretq_s32_s16(rows_45.val[1]));
    int32x4x2_t rows_2367_l = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[0]), vreinterpretq_s32_s16(rows_67.val[0]));
    int32x4x2_t rows_2367_h = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[1]), vreinterpretq_s32_s16(rows_67.val[1]));
    int32x4x2_t cols_04 = vzipq_s32(rows_0145_l.val[0], rows_2367_l.val[0]);
    int32x4x2_t cols_15 = vzipq_s32(rows_0145_h.val[0], rows_2367_h.val[0]);
    int32x4x2_t cols_26 = vzipq_s32(rows_0145_l.val[1], rows_2367_l.val[1]);
    int32x4x2_t cols_37 = vzipq_s32(rows_0145_h.val[1], rows_2367_h.val[1]);
    int16x8_t r_cols[8];
    r_cols[0] = vreinterpretq_s16_s32(cols_04.val[0]);
    r_cols[1] = vreinterpretq_s16_s32(cols_15.val[0]);
    r_cols[2] = vreinterpretq_s16_s32(cols_26.val[0]);
    r_cols[3] = vreinterpretq_s16_s32(cols_37.val[0]);
    r_cols[4] = vreinterpretq_s16_s32(cols_04.val[1]);
    r_cols[5] = vreinterpretq_s16_s32(cols_15.val[1]);
    r_cols[6] = vreinterpretq_s16_s32(cols_26.val[1]);
    r_cols[7] = vreinterpretq_s16_s32(cols_37.val[1]);
    int32x4x2_t cols[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        cols[i].val[0] = vmovl_s16(vget_low_s16(r_cols[i]));
        cols[i].val[1] = vmovl_high_s16(r_cols[i]);
    }

    /* Step 3: row-wise transform. */
#pragma unroll
    for (int p = 0; p < 2; p++) {
        o[0].val[p] = vmulq_laneq_s32(cols[1].val[p], coeffs[1], 0);
        o[1].val[p] = vmulq_laneq_s32(cols[1].val[p], coeffs[1], 1);
        o[2].val[p] = vmulq_laneq_s32(cols[1].val[p], coeffs[1], 2);
        o[3].val[p] = vmulq_laneq_s32(cols[1].val[p], coeffs[1], 3);
    }
#pragma unroll
    for (int rc_idx = 3; rc_idx < 8; rc_idx += 2) {
#pragma unroll
        for (int p = 0; p < 2; p++) {
            o[0].val[p] = vmlaq_laneq_s32(o[0].val[p], cols[rc_idx].val[p], coeffs[rc_idx], 0);
            o[1].val[p] = vmlaq_laneq_s32(o[1].val[p], cols[rc_idx].val[p], coeffs[rc_idx], 1);
            o[2].val[p] = vmlaq_laneq_s32(o[2].val[p], cols[rc_idx].val[p], coeffs[rc_idx], 2);
            o[3].val[p] = vmlaq_laneq_s32(o[3].val[p], cols[rc_idx].val[p], coeffs[rc_idx], 3);
        }
    }
#pragma unroll
    for (int p = 0; p < 2; p++) {
        eo[0].val[p] = vmulq_laneq_s32(cols[2].val[p], coeffs[2], 0);
        eo[0].val[p] = vmlaq_laneq_s32(eo[0].val[p], cols[6].val[p], coeffs[6], 0);
        eo[1].val[p] = vmulq_laneq_s32(cols[2].val[p], coeffs[2], 1);
        eo[1].val[p] = vmlaq_laneq_s32(eo[1].val[p], cols[6].val[p], coeffs[6], 1);
    }
#pragma unroll
    for (int p = 0; p < 2; p++) {
        ee[0].val[p] = vmulq_laneq_s32(cols[0].val[p], coeffs[0], 0);
        ee[0].val[p] = vmlaq_laneq_s32(ee[0].val[p], cols[4].val[p], coeffs[4], 0);
        ee[1].val[p] = vmulq_laneq_s32(cols[0].val[p], coeffs[0], 1);
        ee[1].val[p] = vmlaq_laneq_s32(ee[1].val[p], cols[4].val[p], coeffs[4], 1);
    }
#pragma unroll
    for (int p = 0; p < 2; p++) {
        e[0].val[p] = vaddq_s32(ee[0].val[p], eo[0].val[p]);
        e[1].val[p] = vaddq_s32(ee[1].val[p], eo[1].val[p]);
        e[2].val[p] = vsubq_s32(ee[1].val[p], eo[1].val[p]);
        e[3].val[p] = vsubq_s32(ee[0].val[p], eo[0].val[p]);
    }
    // NOTE: r_cols stands for narcoled cols
#pragma unroll
    for (int k = 0; k < 4; k++) {
        r_cols[k] = vcombine_s16(vqmovn_s32(vrshrq_n_s32(vaddq_s32(e[k].val[0], o[k].val[0]), SHIFT_P2)),
                                 vqmovn_s32(vrshrq_n_s32(vaddq_s32(e[k].val[1], o[k].val[1]), SHIFT_P2)));
        r_cols[k + 4] = vcombine_s16(vqmovn_s32(vrshrq_n_s32(vsubq_s32(e[3 - k].val[0], o[3 - k].val[0]), SHIFT_P2)),
                                     vqmovn_s32(vrshrq_n_s32(vsubq_s32(e[3 - k].val[1], o[3 - k].val[1]), SHIFT_P2)));
    }

    /* Step 4: transpose and store the results. */
    int16x8x4_t s_rows_0123, s_rows_4567;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int16x8x2_t tmp = vzipq_s16(r_cols[i], r_cols[i + 4]);
        s_rows_0123.val[i] = tmp.val[0];
        s_rows_4567.val[i] = tmp.val[1];
    }
    vst4q_s16(output, s_rows_0123);
    vst4q_s16(output + 4 * 8, s_rows_4567);
}