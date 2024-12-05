#include "cstdint"
#include "fir_sparse.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>
#include <cstring>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_sparse_scalar(int LANE_NUM,
                       config_t *config,
                       input_t *input,
                       output_t *output) {

    fir_sparse_config_t *fir_sparse_config = (fir_sparse_config_t *)config;
    fir_sparse_input_t *fir_sparse_input = (fir_sparse_input_t *)input;
    fir_sparse_output_t *fir_sparse_output = (fir_sparse_output_t *)output;

    int sample_count = fir_sparse_config->sample_count;
    int coeff_count = fir_sparse_config->effective_coeff_count;
    int32_t *src = fir_sparse_input->src;
    int32_t *coeff = fir_sparse_input->coeff;
    int32_t *delay = fir_sparse_input->delay;
    int32_t *dst = fir_sparse_output->dst;

    int32_t acc;
    int32_t *src_addr;
    int32_t *coeff_addr;
    int *delay_addr;
    int32_t src_temp;
    int32_t coeff_temp;
    int delay_temp;
    for (int sample_idx = 0; sample_idx < sample_count; sample_idx++) {
        acc = 0;
        coeff_addr = coeff;
        delay_addr = delay;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            delay_temp = *delay_addr;
            src_addr = src + delay_temp;
            src_temp = *src_addr;
            coeff_temp = *coeff_addr;
            acc += src_temp * coeff_temp;
            coeff_addr += 1;
            delay_addr += 1;
        }
        dst[sample_idx] = acc;
        src += 1;
    }
}