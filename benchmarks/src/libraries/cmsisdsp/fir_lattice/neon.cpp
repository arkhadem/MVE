#include "fir_lattice.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_lattice_neon(int LANE_NUM,
                      config_t *config,
                      input_t *input,
                      output_t *output) {

    fir_lattice_config_t *fir_lattice_config = (fir_lattice_config_t *)config;
    fir_lattice_input_t *fir_lattice_input = (fir_lattice_input_t *)input;
    fir_lattice_output_t *fir_lattice_output = (fir_lattice_output_t *)output;

    int sample_count = fir_lattice_config->sample_count;
    int coeff_count = fir_lattice_config->coeff_count;
    int32_t *src = fir_lattice_input->src;
    int32_t *coeff = fir_lattice_input->coeff;
    int32_t *dst = fir_lattice_output->dst;

    int G_size = coeff_count + 1;
    int32_t *G = new int32_t[G_size];
    memset(G, 0, (G_size) * sizeof(int32_t));

    int sample_idx = 0;

    while (sample_idx + 32 <= sample_count) {
        // Address for gm[n]
        int32_t *G_addr = G;

        // Address for km
        int32_t *coeff_addr = coeff;

        int32x4_t G_v0 = vld1q_s32(src + 0);
        int32x4_t G_v1 = vld1q_s32(src + 4);
        int32x4_t G_v2 = vld1q_s32(src + 8);
        int32x4_t G_v3 = vld1q_s32(src + 12);
        int32x4_t G_v4 = vld1q_s32(src + 16);
        int32x4_t G_v5 = vld1q_s32(src + 20);
        int32x4_t G_v6 = vld1q_s32(src + 24);
        int32x4_t G_v7 = vld1q_s32(src + 28);

        // Fval is kept in register
        int32x4_t F_v0 = G_v0;
        int32x4_t F_v1 = G_v1;
        int32x4_t F_v2 = G_v2;
        int32x4_t F_v3 = G_v3;
        int32x4_t F_v4 = G_v4;
        int32x4_t F_v5 = G_v5;
        int32x4_t F_v6 = G_v6;
        int32x4_t F_v7 = G_v7;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            int32x4_t G_v8 = vdupq_n_s16(0.0);

            G_v8 = vsetq_lane_s32(*G_addr, G_v8, 3);

            *G_addr = vgetq_lane_s32(G_v7, 3);
            G_addr += 1;

            int32x4_t prev_G_v0 = vextq_s32(G_v8, G_v0, 3);
            int32x4_t prev_G_v1 = vextq_s32(G_v0, G_v1, 3);
            int32x4_t prev_G_v2 = vextq_s32(G_v1, G_v2, 3);
            int32x4_t prev_G_v3 = vextq_s32(G_v2, G_v3, 3);
            int32x4_t prev_G_v4 = vextq_s32(G_v3, G_v4, 3);
            int32x4_t prev_G_v5 = vextq_s32(G_v4, G_v5, 3);
            int32x4_t prev_G_v6 = vextq_s32(G_v5, G_v6, 3);
            int32x4_t prev_G_v7 = vextq_s32(G_v6, G_v7, 3);

            int32x4_t C_v = vld1q_dup_s32(coeff_addr);

            G_v0 = vmlaq_s32(prev_G_v0, C_v, F_v0);
            G_v1 = vmlaq_s32(prev_G_v1, C_v, F_v1);
            G_v2 = vmlaq_s32(prev_G_v2, C_v, F_v2);
            G_v3 = vmlaq_s32(prev_G_v3, C_v, F_v3);
            G_v4 = vmlaq_s32(prev_G_v4, C_v, F_v4);
            G_v5 = vmlaq_s32(prev_G_v5, C_v, F_v5);
            G_v6 = vmlaq_s32(prev_G_v6, C_v, F_v6);
            G_v7 = vmlaq_s32(prev_G_v7, C_v, F_v7);

            F_v0 = vmlaq_s32(F_v0, C_v, prev_G_v0);
            F_v1 = vmlaq_s32(F_v1, C_v, prev_G_v1);
            F_v2 = vmlaq_s32(F_v2, C_v, prev_G_v2);
            F_v3 = vmlaq_s32(F_v3, C_v, prev_G_v3);
            F_v4 = vmlaq_s32(F_v4, C_v, prev_G_v4);
            F_v5 = vmlaq_s32(F_v5, C_v, prev_G_v5);
            F_v6 = vmlaq_s32(F_v6, C_v, prev_G_v6);
            F_v7 = vmlaq_s32(F_v7, C_v, prev_G_v7);

            coeff_addr += 1;
        }

        *G_addr = vgetq_lane_s32(G_v7, 3);

        vst1q_s32(dst + 0, F_v0);
        vst1q_s32(dst + 4, F_v1);
        vst1q_s32(dst + 8, F_v2);
        vst1q_s32(dst + 12, F_v3);
        vst1q_s32(dst + 16, F_v4);
        vst1q_s32(dst + 20, F_v5);
        vst1q_s32(dst + 24, F_v6);
        vst1q_s32(dst + 28, F_v7);

        src += 32;
        dst += 32;
        sample_idx += 32;
    }

    while (sample_idx + 16 <= sample_count) {
        // Address for gm[n]
        int32_t *G_addr = G;

        // Address for km
        int32_t *coeff_addr = coeff;

        int32x4_t G_v0 = vld1q_s32(src + 0);
        int32x4_t G_v1 = vld1q_s32(src + 4);
        int32x4_t G_v2 = vld1q_s32(src + 8);
        int32x4_t G_v3 = vld1q_s32(src + 12);

        // Fval is kept in register
        int32x4_t F_v0 = G_v0;
        int32x4_t F_v1 = G_v1;
        int32x4_t F_v2 = G_v2;
        int32x4_t F_v3 = G_v3;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            int32x4_t G_v4 = vdupq_n_s16(0.0);

            G_v4 = vsetq_lane_s32(*G_addr, G_v4, 3);

            *G_addr = vgetq_lane_s32(G_v3, 3);
            G_addr += 1;

            int32x4_t prev_G_v0 = vextq_s32(G_v4, G_v0, 3);
            int32x4_t prev_G_v1 = vextq_s32(G_v0, G_v1, 3);
            int32x4_t prev_G_v2 = vextq_s32(G_v1, G_v2, 3);
            int32x4_t prev_G_v3 = vextq_s32(G_v2, G_v3, 3);

            int32x4_t C_v = vld1q_dup_s32(coeff_addr);

            G_v0 = vmlaq_s32(prev_G_v0, C_v, F_v0);
            G_v1 = vmlaq_s32(prev_G_v1, C_v, F_v1);
            G_v2 = vmlaq_s32(prev_G_v2, C_v, F_v2);
            G_v3 = vmlaq_s32(prev_G_v3, C_v, F_v3);

            F_v0 = vmlaq_s32(F_v0, C_v, prev_G_v0);
            F_v1 = vmlaq_s32(F_v1, C_v, prev_G_v1);
            F_v2 = vmlaq_s32(F_v2, C_v, prev_G_v2);
            F_v3 = vmlaq_s32(F_v3, C_v, prev_G_v3);

            coeff_addr += 1;
        }

        *G_addr = vgetq_lane_s32(G_v3, 3);

        vst1q_s32(dst + 0, F_v0);
        vst1q_s32(dst + 4, F_v1);
        vst1q_s32(dst + 8, F_v2);
        vst1q_s32(dst + 12, F_v3);

        src += 16;
        dst += 16;
        sample_idx += 16;
    }

    while (sample_idx + 8 <= sample_count) {
        // Address for gm[n]
        int32_t *G_addr = G;

        // Address for km
        int32_t *coeff_addr = coeff;

        int32x4_t G_v0 = vld1q_s32(src + 0);
        int32x4_t G_v1 = vld1q_s32(src + 4);

        // Fval is kept in register
        int32x4_t F_v0 = G_v0;
        int32x4_t F_v1 = G_v1;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            int32x4_t G_v2 = vdupq_n_s16(0.0);

            G_v2 = vsetq_lane_s32(*G_addr, G_v2, 3);

            *G_addr = vgetq_lane_s32(G_v1, 3);
            G_addr += 1;

            int32x4_t prev_G_v0 = vextq_s32(G_v2, G_v0, 3);
            int32x4_t prev_G_v1 = vextq_s32(G_v0, G_v1, 3);

            int32x4_t C_v = vld1q_dup_s32(coeff_addr);

            G_v0 = vmlaq_s32(prev_G_v0, C_v, F_v0);
            G_v1 = vmlaq_s32(prev_G_v1, C_v, F_v1);

            F_v0 = vmlaq_s32(F_v0, C_v, prev_G_v0);
            F_v1 = vmlaq_s32(F_v1, C_v, prev_G_v1);

            coeff_addr += 1;
        }

        *G_addr = vgetq_lane_s32(G_v1, 3);

        vst1q_s32(dst + 0, F_v0);
        vst1q_s32(dst + 4, F_v1);

        src += 8;
        dst += 8;
        sample_idx += 8;
    }

    while (sample_idx + 4 <= sample_count) {
        // Address for gm[n]
        int32_t *G_addr = G;

        // Address for km
        int32_t *coeff_addr = coeff;

        int32x4_t G_v0 = vld1q_s32(src + 0);

        // Fval is kept in register
        int32x4_t F_v0 = G_v0;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            int32x4_t G_v1 = vdupq_n_s16(0.0);

            G_v1 = vsetq_lane_s32(*G_addr, G_v1, 3);

            *G_addr = vgetq_lane_s32(G_v0, 3);
            G_addr += 1;

            int32x4_t prev_G_v0 = vextq_s32(G_v1, G_v0, 3);

            int32x4_t C_v = vld1q_dup_s32(coeff_addr);

            G_v0 = vmlaq_s32(prev_G_v0, C_v, F_v0);

            F_v0 = vmlaq_s32(F_v0, C_v, prev_G_v0);

            coeff_addr += 1;
        }

        *G_addr = vgetq_lane_s32(G_v0, 3);

        vst1q_s32(dst + 0, F_v0);

        src += 4;
        dst += 4;
        sample_idx += 4;
    }

    while (sample_idx + 1 <= sample_count) {
        // Address for gm[n]
        int32_t *G_addr = G;

        // Address for km
        int32_t *coeff_addr = coeff;

        int32_t G_v = *src;

        // Fval is kept in register
        int32_t F_v = G_v;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            int32_t prev_G_v = *G_addr;

            *G_addr = G_v;
            G_addr += 1;

            int32_t C_v = *coeff_addr;

            G_v = prev_G_v + C_v * F_v;

            F_v += C_v * prev_G_v;

            coeff_addr += 1;
        }

        *G_addr = G_v;

        *dst = F_v;

        src += 1;
        dst += 1;
        sample_idx += 1;
    }
}
