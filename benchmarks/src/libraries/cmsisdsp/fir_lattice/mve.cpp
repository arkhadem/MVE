#include "mve.hpp"
#include "cstdint"
#include "fir_lattice.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

void fir_lattice_mve(int LANE_NUM,
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

    // Dim0: samples
    _mve_set_dim_count(1);

    // Source and dst is loaded consequetively
    __vidx_var unit_stride = {1, 0, 0, 0};

    // Same coefficient for all cells
    __vidx_var duplicate_stride = {0, 0, 0, 0};

    int G_size = LANE_NUM + 1;
    int32_t *G[2];
    G[0] = new int32_t[G_size];
    G[1] = new int32_t[G_size];
    memset(G[0], 0, G_size * sizeof(int32_t));
    memset(G[1], 0, G_size * sizeof(int32_t));
    int32_t *last_G_addr[2];

    int prev_G_size = coeff_count + 1;
    int32_t *prev_G;
    prev_G = new int32_t[prev_G_size];
    memset(prev_G, 0, prev_G_size * sizeof(int32_t));

    int32_t *coeff_addr;
    int sample_idx = 0;
    int G_idx;

    while (sample_idx < sample_count) {

        int curr_sample_per_iter = (sample_count - sample_idx) < LANE_NUM ? (sample_count - sample_idx) : LANE_NUM;
        _mve_set_dim_length(0, curr_sample_per_iter);
        last_G_addr[0] = G[0] + curr_sample_per_iter;
        last_G_addr[1] = G[1] + curr_sample_per_iter;

        coeff_addr = coeff;

        G[0][0] = prev_G[0];
        memcpy(G[0] + 1, src, curr_sample_per_iter * sizeof(int32_t));

        // r0
        __mdvdw fval_v = _mve_load_dw(src, unit_stride);

        int32_t *prev_G_addr = prev_G;

        *prev_G_addr = *(last_G_addr[0]);

        G_idx = 1;
        prev_G_addr += 1;

        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {

            G[G_idx][0] = *prev_G_addr;

            // gm-1[n-1]
            // r1
            __mdvdw prev_G_v = _mve_load_dw(G[1 - G_idx], unit_stride);

            // km
            // r2
            __mdvdw coeff_v = _mve_load_dw(coeff_addr, duplicate_stride);

            // gm[n] = gm-1[n-1] + km * fm-1[n]
            // r3
            __mdvdw f_mult_v = _mve_mul_dw(coeff_v, fval_v);

            // r4
            __mdvdw curr_G_v = _mve_add_dw(prev_G_v, f_mult_v);
            // free f_mult_v (r3)
            _mve_free_dw();

            // fm[n] = fm-1[n] + km * gm-1[n-1]
            // r3
            __mdvdw G_mult_v = _mve_mul_dw(coeff_v, prev_G_v);
            // free prev_G_v (r1) and coeff_v (r2)
            _mve_free_dw();
            _mve_free_dw();

            // r1
            fval_v = _mve_add_dw(fval_v, G_mult_v);
            // free G_mult_v (r3) and fval_v (r0 or r1)
            _mve_free_dw();
            _mve_free_dw();

            // store the results in G
            _mve_store_dw(G[G_idx] + 1, curr_G_v, unit_stride);
            // free curr_G_v (r4)
            _mve_free_dw();

            *prev_G_addr = *(last_G_addr[G_idx]);

            coeff_addr += 1;
            prev_G_addr += 1;
            G_idx = 1 - G_idx;
        }

        _mve_store_dw(dst, fval_v, unit_stride);
        // free fval_v (r0 or r1)
        _mve_free_dw();

        sample_idx += curr_sample_per_iter;
        src += curr_sample_per_iter;
        dst += curr_sample_per_iter;
    }
}