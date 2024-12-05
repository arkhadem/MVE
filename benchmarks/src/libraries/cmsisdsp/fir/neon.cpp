#include "fir.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_neon(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    fir_config_t *fir_config = (fir_config_t *)config;
    fir_input_t *fir_input = (fir_input_t *)input;
    fir_output_t *fir_output = (fir_output_t *)output;

    int sample_count = fir_config->sample_count;
    int coeff_count = fir_config->coeff_count;
    int32_t *src = fir_input->src;
    int32_t *coeff = fir_input->coeff;
    int32_t *dst = fir_output->dst;

    int32_t *src_addr;
    int32_t *dst_addr;
    int32_t *coeff_addr;
    int32x4_t vcoeff;
    int sample_idx = 0;
    while (sample_idx + 32 <= sample_count) {
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vdupq_n_s32(0);
        int32x4_t vacc89AB = vdupq_n_s32(0);
        int32x4_t vaccCDEF = vdupq_n_s32(0);
        int32x4_t vaccGHIJ = vdupq_n_s32(0);
        int32x4_t vaccKLMN = vdupq_n_s32(0);
        int32x4_t vaccOPQR = vdupq_n_s32(0);
        int32x4_t vaccSTUV = vdupq_n_s32(0);
        src_addr = src + sample_idx;
        dst_addr = dst + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            int32x4_t vsrc0123 = vld1q_s32(src_addr);
            int32x4_t vsrc4567 = vld1q_s32(src_addr + 4);
            int32x4_t vsrc89AB = vld1q_s32(src_addr + 8);
            int32x4_t vsrcCDEF = vld1q_s32(src_addr + 12);
            int32x4_t vsrcGHIJ = vld1q_s32(src_addr + 16);
            int32x4_t vsrcKLMN = vld1q_s32(src_addr + 20);
            int32x4_t vsrcOPQR = vld1q_s32(src_addr + 24);
            int32x4_t vsrcSTUV = vld1q_s32(src_addr + 28);
            vcoeff = vld1q_dup_s32(coeff_addr);
            vacc0123 = vmlaq_s32(vacc0123, vsrc0123, vcoeff);
            vacc4567 = vmlaq_s32(vacc4567, vsrc4567, vcoeff);
            vacc89AB = vmlaq_s32(vacc89AB, vsrc89AB, vcoeff);
            vaccCDEF = vmlaq_s32(vaccCDEF, vsrcCDEF, vcoeff);
            vaccGHIJ = vmlaq_s32(vaccGHIJ, vsrcGHIJ, vcoeff);
            vaccKLMN = vmlaq_s32(vaccKLMN, vsrcKLMN, vcoeff);
            vaccOPQR = vmlaq_s32(vaccOPQR, vsrcOPQR, vcoeff);
            vaccSTUV = vmlaq_s32(vaccSTUV, vsrcSTUV, vcoeff);
            src_addr += 1;
            coeff_addr += 1;
        }
        dst_addr = dst + sample_idx;
        vst1q_s32(dst_addr, vacc0123);
        vst1q_s32(dst_addr + 4, vacc4567);
        vst1q_s32(dst_addr + 8, vacc89AB);
        vst1q_s32(dst_addr + 12, vaccCDEF);
        vst1q_s32(dst_addr + 16, vaccGHIJ);
        vst1q_s32(dst_addr + 20, vaccKLMN);
        vst1q_s32(dst_addr + 24, vaccOPQR);
        vst1q_s32(dst_addr + 28, vaccSTUV);
        sample_idx += 32;
    }
    while (sample_idx + 16 <= sample_count) {
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vdupq_n_s32(0);
        int32x4_t vacc89AB = vdupq_n_s32(0);
        int32x4_t vaccCDEF = vdupq_n_s32(0);
        src_addr = src + sample_idx;
        dst_addr = dst + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            int32x4_t vsrc0123 = vld1q_s32(src_addr);
            int32x4_t vsrc4567 = vld1q_s32(src_addr + 4);
            int32x4_t vsrc89AB = vld1q_s32(src_addr + 8);
            int32x4_t vsrcCDEF = vld1q_s32(src_addr + 12);
            vcoeff = vld1q_dup_s32(coeff_addr);
            vacc0123 = vmlaq_s32(vacc0123, vsrc0123, vcoeff);
            vacc4567 = vmlaq_s32(vacc4567, vsrc4567, vcoeff);
            vacc89AB = vmlaq_s32(vacc89AB, vsrc89AB, vcoeff);
            vaccCDEF = vmlaq_s32(vaccCDEF, vsrcCDEF, vcoeff);
            src_addr += 1;
            coeff_addr += 1;
        }
        dst_addr = dst + sample_idx;
        vst1q_s32(dst_addr, vacc0123);
        vst1q_s32(dst_addr + 4, vacc4567);
        vst1q_s32(dst_addr + 8, vacc89AB);
        vst1q_s32(dst_addr + 12, vaccCDEF);
        sample_idx += 16;
    }
    while (sample_idx + 8 <= sample_count) {
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vdupq_n_s32(0);
        src_addr = src + sample_idx;
        dst_addr = dst + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            int32x4_t vsrc0123 = vld1q_s32(src_addr);
            int32x4_t vsrc4567 = vld1q_s32(src_addr + 4);
            vcoeff = vld1q_dup_s32(coeff_addr);
            vacc0123 = vmlaq_s32(vacc0123, vsrc0123, vcoeff);
            vacc4567 = vmlaq_s32(vacc4567, vsrc4567, vcoeff);
            src_addr += 1;
            coeff_addr += 1;
        }
        dst_addr = dst + sample_idx;
        vst1q_s32(dst_addr, vacc0123);
        vst1q_s32(dst_addr + 4, vacc4567);
        sample_idx += 8;
    }
    while (sample_idx + 4 <= sample_count) {
        int32x4_t vacc0123 = vdupq_n_s32(0);
        src_addr = src + sample_idx;
        dst_addr = dst + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            int32x4_t vsrc0123 = vld1q_s32(src_addr);
            vcoeff = vld1q_dup_s32(coeff_addr);
            vacc0123 = vmlaq_s32(vacc0123, vsrc0123, vcoeff);
            src_addr += 1;
            coeff_addr += 1;
        }
        dst_addr = dst + sample_idx;
        vst1q_s32(dst_addr, vacc0123);
        sample_idx += 4;
    }
    while (sample_idx < sample_count) {
        int32_t acc = 0;
        src_addr = src + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            int32_t src_temp = *src_addr;
            int32_t coeff_temp = *coeff_addr;
            acc += src_temp * coeff_temp;
            src_addr += 1;
            coeff_addr += 1;
        }
        dst[sample_idx] = acc;
        sample_idx++;
    }
}
