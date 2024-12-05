#include "fir_sparse.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_sparse_rvv(int LANE_NUM,
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

    // Dim0: samples
    _mve_set_dim_count(1);

    // Source is loaded consequetively
    __vidx_var src_dst_stride = {1, 0, 0, 0};

    // Same coefficient for all cells
    __vidx_var coeff_stride = {0, 0, 0, 0};

    int32_t *src_addr;
    int32_t *coeff_addr;
    int *delay_addr;
    int delay_temp;
    int sample_idx = 0;

    while (sample_idx < sample_count) {

        int curr_sample_per_iter = (sample_count - sample_idx) < LANE_NUM ? (sample_count - sample_idx) : LANE_NUM;

        _mve_set_dim_length(0, curr_sample_per_iter);

        __mdvdw acc_v = _mve_set1_dw(0);

        coeff_addr = coeff;
        delay_addr = delay;

        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            delay_temp = *delay_addr;
            src_addr = src + delay_temp;

            __mdvdw src_v = _mve_load_dw(src_addr, src_dst_stride);
            __mdvdw coeff_v = _mve_load_dw(coeff_addr, coeff_stride);
            __mdvdw mult_v = _mve_mul_dw(src_v, coeff_v);

            acc_v = _mve_add_dw(acc_v, mult_v);

            coeff_addr += 1;
            delay_addr += 1;
        }

        _mve_store_dw(dst, acc_v, src_dst_stride);

        sample_idx += curr_sample_per_iter;
        src += curr_sample_per_iter;
        dst += curr_sample_per_iter;
    }
}