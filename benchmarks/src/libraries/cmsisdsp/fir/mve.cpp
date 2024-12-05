#include "mve.hpp"
#include "cstdint"
#include "fir.hpp"
#include <cstdint>
#include <cstdio>

void fir_mve(int LANE_NUM,
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

    // Dim0: samples
    _mve_set_dim_count(1);

    // Source is loaded consequetively
    __vidx_var src_dst_stride = {1, 0, 0, 0};

    // Same coefficient for all cells
    __vidx_var coeff_stride = {0, 0, 0, 0};

    int32_t *src_addr;
    int32_t *dst_addr;
    int32_t *coeff_addr;
    int sample_idx = 0;

    while (sample_idx < sample_count) {

        int curr_sample_per_iter = (sample_count - sample_idx) < LANE_NUM ? (sample_count - sample_idx) : LANE_NUM;

        _mve_set_dim_length(0, curr_sample_per_iter);

        // r0
        __mdvdw acc_v = _mve_set1_dw(0);

        src_addr = src + sample_idx;
        coeff_addr = coeff;

        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            // r1
            __mdvdw src_v = _mve_load_dw(src_addr, src_dst_stride);

            // r2
            __mdvdw coeff_v = _mve_load_dw(coeff_addr, coeff_stride);

            // r3
            __mdvdw mult_v = _mve_mul_dw(src_v, coeff_v);
            // free src_v (r1) and coeff_v (r2)
            _mve_free_dw();
            _mve_free_dw();

            // r1
            acc_v = _mve_add_dw(acc_v, mult_v);
            // free mult_v (r3) and acc_v (r0 or r1)
            _mve_free_dw();
            _mve_free_dw();

            src_addr += 1;
            coeff_addr += 1;
        }

        dst_addr = dst + sample_idx;

        _mve_store_dw(dst_addr, acc_v, src_dst_stride);
        // free acc_v (r0 or r1)
        _mve_free_dw();

        sample_idx += curr_sample_per_iter;
    }
}