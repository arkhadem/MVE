#include "kvazaar.hpp"
#include "neon_kernels.hpp"
#include "spmm.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

void spmm_neon(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {

    spmm_config_t *spmm_config = (spmm_config_t *)config;
    spmm_input_t *spmm_input = (spmm_input_t *)input;
    spmm_output_t *spmm_output = (spmm_output_t *)output;

    int M = spmm_config->M;
    int N = spmm_config->N;
    int K = spmm_config->K;
    int32_t min = spmm_config->min;
    int32_t max = spmm_config->max;
    int32_t *in = spmm_input->input;
    int32_t *bias = spmm_input->bias;
    int32_t *weights = spmm_input->weights;
    int32_t *IDX = spmm_input->IDX;
    uint32_t *NNZ = spmm_input->NNZ;
    int32_t *out = spmm_output->output;

    const int32x4_t vmin = vld1q_dup_s32(&min);
    const int32x4_t vmax = vld1q_dup_s32(&max);

    int m = 0;

    while (m + 32 <= M) {
        for (int n = 0; n < N; n++) {
            const int32_t *bias_addr = bias + n;
            int32x4_t vacc0123 = vld1q_dup_s32(bias_addr);
            int32x4_t vacc4567 = vacc0123;
            int32x4_t vacc89AB = vacc0123;
            int32x4_t vaccCDEF = vacc0123;
            int32x4_t vaccGHIJ = vacc0123;
            int32x4_t vaccKLMN = vacc0123;
            int32x4_t vaccOPQR = vacc0123;
            int32x4_t vaccSTUV = vacc0123;

            uint32_t max_k = NNZ[n + 1];
            for (uint32_t k = NNZ[n]; k < max_k; k++) {

                const int32_t *input_addr = (const int32_t *)(in + IDX[k] * M + m);
                const int32x4_t vi0123 = vld1q_s32(input_addr);
                const int32x4_t vi4567 = vld1q_s32(input_addr + 4);
                const int32x4_t vi89AB = vld1q_s32(input_addr + 8);
                const int32x4_t viCDEF = vld1q_s32(input_addr + 12);
                const int32x4_t viGHIJ = vld1q_s32(input_addr + 16);
                const int32x4_t viKLMN = vld1q_s32(input_addr + 20);
                const int32x4_t viOPQR = vld1q_s32(input_addr + 24);
                const int32x4_t viSTUV = vld1q_s32(input_addr + 28);

                const int32_t *weight_addr = (const int32_t *)(weights + k);
                const int32x4_t vw = vld1q_dup_s32(weight_addr);

                const int32x4_t vm0123 = vmulq_s32(vi0123, vw);
                const int32x4_t vm4567 = vmulq_s32(vi4567, vw);
                const int32x4_t vm89AB = vmulq_s32(vi89AB, vw);
                const int32x4_t vmCDEF = vmulq_s32(viCDEF, vw);
                const int32x4_t vmGHIJ = vmulq_s32(viGHIJ, vw);
                const int32x4_t vmKLMN = vmulq_s32(viKLMN, vw);
                const int32x4_t vmOPQR = vmulq_s32(viOPQR, vw);
                const int32x4_t vmSTUV = vmulq_s32(viSTUV, vw);

                vacc0123 = vaddq_s32(vacc0123, vm0123);
                vacc4567 = vaddq_s32(vacc4567, vm4567);
                vacc89AB = vaddq_s32(vacc89AB, vm89AB);
                vaccCDEF = vaddq_s32(vaccCDEF, vmCDEF);
                vaccGHIJ = vaddq_s32(vaccGHIJ, vmGHIJ);
                vaccKLMN = vaddq_s32(vaccKLMN, vmKLMN);
                vaccOPQR = vaddq_s32(vaccOPQR, vmOPQR);
                vaccSTUV = vaddq_s32(vaccSTUV, vmSTUV);
            }

            int32x4_t vout0123 = vminq_s32(vacc0123, vmax);
            int32x4_t vout4567 = vminq_s32(vacc4567, vmax);
            int32x4_t vout89AB = vminq_s32(vacc89AB, vmax);
            int32x4_t voutCDEF = vminq_s32(vaccCDEF, vmax);
            int32x4_t voutGHIJ = vminq_s32(vaccGHIJ, vmax);
            int32x4_t voutKLMN = vminq_s32(vaccKLMN, vmax);
            int32x4_t voutOPQR = vminq_s32(vaccOPQR, vmax);
            int32x4_t voutSTUV = vminq_s32(vaccSTUV, vmax);
            vout0123 = vmaxq_s32(vout0123, vmin);
            vout4567 = vmaxq_s32(vout4567, vmin);
            vout89AB = vmaxq_s32(vout89AB, vmin);
            voutCDEF = vmaxq_s32(voutCDEF, vmin);
            voutGHIJ = vmaxq_s32(voutGHIJ, vmin);
            voutKLMN = vmaxq_s32(voutKLMN, vmin);
            voutOPQR = vmaxq_s32(voutOPQR, vmin);
            voutSTUV = vmaxq_s32(voutSTUV, vmin);

            int32_t *output_addr = (int32_t *)(out + n * M + m);
            vst1q_s32(output_addr, vout0123);
            vst1q_s32(output_addr + 4, vout4567);
            vst1q_s32(output_addr + 8, vout89AB);
            vst1q_s32(output_addr + 12, voutCDEF);
            vst1q_s32(output_addr + 16, voutGHIJ);
            vst1q_s32(output_addr + 20, voutKLMN);
            vst1q_s32(output_addr + 24, voutOPQR);
            vst1q_s32(output_addr + 28, voutSTUV);
        }
        m += 32;
    }

    while (m + 16 <= M) {
        for (int n = 0; n < N; n++) {
            const int32_t *bias_addr = bias + n;
            int32x4_t vacc0123 = vld1q_dup_s32(bias_addr);
            int32x4_t vacc4567 = vacc0123;
            int32x4_t vacc89AB = vacc0123;
            int32x4_t vaccCDEF = vacc0123;

            uint32_t max_k = NNZ[n + 1];
            for (uint32_t k = NNZ[n]; k < max_k; k++) {

                const int32_t *input_addr = (const int32_t *)(in + IDX[k] * M + m);
                const int32x4_t vi0123 = vld1q_s32(input_addr);
                const int32x4_t vi4567 = vld1q_s32(input_addr + 4);
                const int32x4_t vi89AB = vld1q_s32(input_addr + 8);
                const int32x4_t viCDEF = vld1q_s32(input_addr + 12);

                const int32_t *weight_addr = (const int32_t *)(weights + k);
                const int32x4_t vw = vld1q_dup_s32(weight_addr);

                const int32x4_t vm0123 = vmulq_s32(vi0123, vw);
                const int32x4_t vm4567 = vmulq_s32(vi4567, vw);
                const int32x4_t vm89AB = vmulq_s32(vi89AB, vw);
                const int32x4_t vmCDEF = vmulq_s32(viCDEF, vw);

                vacc0123 = vaddq_s32(vacc0123, vm0123);
                vacc4567 = vaddq_s32(vacc4567, vm4567);
                vacc89AB = vaddq_s32(vacc89AB, vm89AB);
                vaccCDEF = vaddq_s32(vaccCDEF, vmCDEF);
            }

            int32x4_t vout0123 = vminq_s32(vacc0123, vmax);
            int32x4_t vout4567 = vminq_s32(vacc4567, vmax);
            int32x4_t vout89AB = vminq_s32(vacc89AB, vmax);
            int32x4_t voutCDEF = vminq_s32(vaccCDEF, vmax);
            vout0123 = vmaxq_s32(vout0123, vmin);
            vout4567 = vmaxq_s32(vout4567, vmin);
            vout89AB = vmaxq_s32(vout89AB, vmin);
            voutCDEF = vmaxq_s32(voutCDEF, vmin);

            int32_t *output_addr = (int32_t *)(out + n * M + m);
            vst1q_s32(output_addr, vout0123);
            vst1q_s32(output_addr + 4, vout4567);
            vst1q_s32(output_addr + 8, vout89AB);
            vst1q_s32(output_addr + 12, voutCDEF);
        }
        m += 16;
    }

    while (m + 8 <= M) {
        for (int n = 0; n < N; n++) {
            const int32_t *bias_addr = bias + n;
            int32x4_t vacc0123 = vld1q_dup_s32(bias_addr);
            int32x4_t vacc4567 = vacc0123;

            uint32_t max_k = NNZ[n + 1];
            for (uint32_t k = NNZ[n]; k < max_k; k++) {

                const int32_t *input_addr = (const int32_t *)(in + IDX[k] * M + m);
                const int32x4_t vi0123 = vld1q_s32(input_addr);
                const int32x4_t vi4567 = vld1q_s32(input_addr + 4);

                const int32_t *weight_addr = (const int32_t *)(weights + k);
                const int32x4_t vw = vld1q_dup_s32(weight_addr);

                const int32x4_t vm0123 = vmulq_s32(vi0123, vw);
                const int32x4_t vm4567 = vmulq_s32(vi4567, vw);

                vacc0123 = vaddq_s32(vacc0123, vm0123);
                vacc4567 = vaddq_s32(vacc4567, vm4567);
            }

            int32x4_t vout0123 = vminq_s32(vacc0123, vmax);
            int32x4_t vout4567 = vminq_s32(vacc4567, vmax);
            vout0123 = vmaxq_s32(vout0123, vmin);
            vout4567 = vmaxq_s32(vout4567, vmin);

            int32_t *output_addr = (int32_t *)(out + n * M + m);
            vst1q_s32(output_addr, vout0123);
            vst1q_s32(output_addr + 4, vout4567);
        }
        m += 8;
    }

    while (m + 4 <= M) {
        for (int n = 0; n < N; n++) {
            const int32_t *bias_addr = bias + n;
            int32x4_t vacc0123 = vld1q_dup_s32(bias_addr);

            uint32_t max_k = NNZ[n + 1];
            for (uint32_t k = NNZ[n]; k < max_k; k++) {

                const int32_t *input_addr = (const int32_t *)(in + IDX[k] * M + m);
                const int32x4_t vi0123 = vld1q_s32(input_addr);

                const int32_t *weight_addr = (const int32_t *)(weights + k);
                const int32x4_t vw = vld1q_dup_s32(weight_addr);

                const int32x4_t vm0123 = vmulq_s32(vi0123, vw);

                vacc0123 = vaddq_s32(vacc0123, vm0123);
            }

            int32x4_t vout0123 = vminq_s32(vacc0123, vmax);
            vout0123 = vmaxq_s32(vout0123, vmin);

            int32_t *output_addr = (int32_t *)(out + n * M + m);
            vst1q_s32(output_addr, vout0123);
        }
        m += 4;
    }

    for (int n = 0; n < N; n++) {
        for (int m_l = m; m_l < M; m_l++) {
            int32_t acc = bias[n];
            for (uint32_t k = NNZ[n]; k < NNZ[n + 1]; k++) {
                acc += (in[IDX[k] * M + m_l] * weights[k]);
            }
            acc = acc < max ? acc : max;
            acc = acc > min ? acc : min;
            out[n * M + m_l] = acc;
        }
    }
}