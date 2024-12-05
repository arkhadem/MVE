#include "cstdint"
#include "fir_lattice.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>
#include <cstring>

// Output stationary
// src[sample_count + coeff_count - 1]
// coeff[coeff_count]
// dst[sample_count]
void fir_lattice_scalar(int LANE_NUM,
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

    int32_t f_val;
    int32_t *G = new int32_t[coeff_count + 1];
    memset(G, 0, (coeff_count + 1) * sizeof(int32_t));
    int32_t *G_curr_addr;
    int32_t *coeff_addr;
    int32_t G_prev_temp;
    int32_t G_curr_temp;
    int32_t coeff_temp;
    for (int sample_idx = 0; sample_idx < sample_count; sample_idx++) {
        // f0[n] = src[n]
        f_val = src[sample_idx];

        // g0[n-1]
        G_prev_temp = G[0];

        // g0[n]
        G[0] = src[sample_idx];

        coeff_addr = coeff;

        // let's start from g1[n] now
        G_curr_addr = G + 1;
        for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
            coeff_temp = *coeff_addr;

            G_curr_temp = coeff_temp * f_val + G_prev_temp;
            f_val += coeff_temp * G_prev_temp;

            G_prev_temp = *G_curr_addr;
            *G_curr_addr = G_curr_temp;
            G_curr_addr += 1;
            coeff_addr += 1;
        }
        *dst = f_val;
        dst++;
    }
}