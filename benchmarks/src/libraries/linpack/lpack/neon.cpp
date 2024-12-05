#include "kvazaar.hpp"
#include "lpack.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

void lpack_neon(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {

    lpack_config_t *lpack_config = (lpack_config_t *)config;
    lpack_input_t *lpack_input = (lpack_input_t *)input;
    lpack_output_t *lpack_output = (lpack_output_t *)output;

    int n = lpack_config->n;
    int32_t *da = lpack_input->da;
    int32_t *dx = lpack_input->dx;
    int32_t *dyin = lpack_input->dyin;
    int32_t *dyout = lpack_output->dyout;
    int32x4_t da_v = vdupq_n_s32(da[0]);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t dyin_v = vld1q_s32(dyin + i);
        int32x4_t dx_v = vld1q_s32(dx + i);
        int32x4_t mul_result = vmulq_s32(dx_v, da_v);
        int32x4_t result = vaddq_s32(dyin_v, mul_result);
        vst1q_s32(dyout + i, result);
    }
    for (; i < n; i++) {
        dyout[i] = dyin[i] + da[0] * dx[i];
    }
}