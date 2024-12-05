#include "cstdint"
#include "fir.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_scalar(int LANE_NUM,
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

    int32_t acc;
    int32_t *src_addr;
    int32_t *coeff_addr;
    int32_t src_temp;
    int32_t coeff_temp;
    for (int sample_idx = 0; sample_idx < sample_count; sample_idx++) {
        acc = 0;
        src_addr = src + sample_idx;
        coeff_addr = coeff;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            src_temp = *src_addr;
            coeff_temp = *coeff_addr;
            acc += src_temp * coeff_temp;
            src_addr += 1;
            coeff_addr += 1;
        }
        dst[sample_idx] = acc;
    }
}