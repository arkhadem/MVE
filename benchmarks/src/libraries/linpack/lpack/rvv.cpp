#include "kvazaar.hpp"
#include "lpack.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

void lpack_rvv(int LANE_NUM,
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

    // Dim0: samples
    _mve_set_dim_count(1);

    // Source is loaded consequetively
    __vidx_var dx_dy_stride = {1, 0, 0, 0};

    // Same da for all cells
    __vidx_var da_stride = {0, 0, 0, 0};

    int32_t *dx_addr = dx;
    int32_t *dyin_addr = dyin;
    int32_t *dyout_addr = dyout;
    int sample_idx = 0;

    __mdvdw da_v = _mve_set1_dw(da[0]);

    while (sample_idx < n) {

        int curr_sample_per_iter = (n - sample_idx) < LANE_NUM ? (n - sample_idx) : LANE_NUM;

        _mve_set_dim_length(0, curr_sample_per_iter);

        dx_addr = dx + sample_idx;
        dyin_addr = dyin + sample_idx;
        dyin_addr = dyin + sample_idx;

        __mdvdw dx_v = _mve_load_dw(dx_addr, dx_dy_stride);

        __mdvdw mult_v = _mve_mul_dw(dx_v, da_v);
        // free dx_v
        _mve_free_dw();

        __mdvdw dyin_v = _mve_load_dw(dyin_addr, dx_dy_stride);

        __mdvdw new_dy = _mve_add_dw(dyin_v, mult_v);
        // free dyin_v and mult_v
        _mve_free_dw();
        _mve_free_dw();

        _mve_store_dw(dyout_addr, new_dy, dx_dy_stride);
        // free new_dy
        _mve_free_dw();

        sample_idx += curr_sample_per_iter;
        dx_addr += curr_sample_per_iter;
        dyin_addr += curr_sample_per_iter;
        dyout_addr += curr_sample_per_iter;
    }
}