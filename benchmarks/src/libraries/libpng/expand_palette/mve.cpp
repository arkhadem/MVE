#include "mve.hpp"
#include "mve_kernels.hpp"
#include <cstdio>

#include "expand_palette.hpp"

void expand_palette_mve(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    expand_palette_config_t *expand_palette_config = (expand_palette_config_t *)config;
    expand_palette_input_t *expand_palette_input = (expand_palette_input_t *)input;
    expand_palette_output_t *expand_palette_output = (expand_palette_output_t *)output;
    // Dim1: pixels
    // Dim0: rows
    _mve_set_dim_count(2);

    // Dim1: Loading the next palette index
    // Dim2: loading next row (random)
    __vidx_var input_stride = {1, 0, 0, 0};
    uint8_t **input_addr = expand_palette_input->input_buf;

    // Dim0: Storing to every 4 byte
    // Dim1: Storing next row (randomly)
    _mve_set_store_stride(0, 4);
    __vidx_var output_stride = {3, 0, 0, 0};
    uint8_t **output_addr = expand_palette_output->output_buf;

    uint8_t *r_palette = expand_palette_input->r_palette;
    uint8_t *g_palette = expand_palette_input->g_palette;
    uint8_t *b_palette = expand_palette_input->b_palette;
    uint8_t *a_palette = expand_palette_input->a_palette;

    int num_cols = expand_palette_config->num_cols;
    int DIM_TILE = LANE_NUM / num_cols;
    _mve_set_dim_length(0, num_cols);
    _mve_set_dim_length(1, DIM_TILE);

    // printf("%d %d", DIM_TILE, num_cols);

    __mdvb key1_b = _mve_loadr_b((const __uint8_t **)input_addr, input_stride);
    input_addr += 8;
    __mdvb key2_b = _mve_loadr_b((const __uint8_t **)input_addr, input_stride);

    __mdvb r1_b = _mve_set1_b(0);
    __mdvb r2_b = _mve_set1_b(0);
    __mdvb g1_b = _mve_set1_b(0);
    __mdvb g2_b = _mve_set1_b(0);
    __mdvb b1_b = _mve_set1_b(0);
    __mdvb b2_b = _mve_set1_b(0);
    __mdvb a1_b = _mve_set1_b(0);
    __mdvb a2_b = _mve_set1_b(0);

#pragma unroll(256)
    for (int i = 0; i < 256; i++) {
        __mdvb key_b = _mve_set1_b(i);
        __mdvb r_b = _mve_set1_b(r_palette[i]);
        __mdvb g_b = _mve_set1_b(g_palette[i]);
        __mdvb b_b = _mve_set1_b(b_palette[i]);
        __mdvb a_b = _mve_set1_b(a_palette[i]);
        _mve_cmpeq_b(key1_b, key_b);
        r1_b = _mve_assign_b(r1_b, r_b);
        // free r1_b
        _mve_free_b();
        g1_b = _mve_assign_b(g1_b, g_b);
        // free g1_b
        _mve_free_b();
        b1_b = _mve_assign_b(b1_b, b_b);
        // free b1_b
        _mve_free_b();
        a1_b = _mve_assign_b(a1_b, a_b);
        // free a1_b
        _mve_free_b();
        _mve_set_mask();
        _mve_cmpeq_b(key2_b, key_b);
        // free key_b
        _mve_free_b();
        r2_b = _mve_assign_b(r2_b, r_b);
        // free r2_b and r_b
        _mve_free_b();
        _mve_free_b();
        g2_b = _mve_assign_b(g2_b, g_b);
        // free g2_b and g_b
        _mve_free_b();
        _mve_free_b();
        b2_b = _mve_assign_b(b2_b, b_b);
        // free b2_b and b_b
        _mve_free_b();
        _mve_free_b();
        a2_b = _mve_assign_b(a2_b, a_b);
        // free a2_b and a_b
        _mve_free_b();
        _mve_free_b();
        _mve_set_mask();
    }
    // free key1_b and key2_b
    _mve_free_b();
    _mve_free_b();

    _mve_storero_b(output_addr, 0, a1_b, output_stride);
    // free a1_b
    _mve_free_b();
    _mve_storero_b(output_addr, 1, r1_b, output_stride);
    // free r1_b
    _mve_free_b();
    _mve_storero_b(output_addr, 2, g1_b, output_stride);
    // free g1_b
    _mve_free_b();
    _mve_storero_b(output_addr, 3, b1_b, output_stride);
    // free b1_b
    _mve_free_b();
    output_addr += 8;
    _mve_storero_b(output_addr, 0, a2_b, output_stride);
    // free a2_b
    _mve_free_b();
    _mve_storero_b(output_addr, 1, r2_b, output_stride);
    // free r2_b
    _mve_free_b();
    _mve_storero_b(output_addr, 2, g2_b, output_stride);
    // free g2_b
    _mve_free_b();
    _mve_storero_b(output_addr, 3, b2_b, output_stride);
    // free b2_b
    _mve_free_b();
}