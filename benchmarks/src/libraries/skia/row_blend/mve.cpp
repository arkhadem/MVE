#include "mve.hpp"
#include "mve_kernels.hpp"

#include "row_blend.hpp"

void row_blend_mve(int LANE_NUM,
                   config_t *config,
                   input_t *input,
                   output_t *output) {
    row_blend_config_t *row_blend_config = (row_blend_config_t *)config;
    row_blend_input_t *row_blend_input = (row_blend_input_t *)input;
    row_blend_output_t *row_blend_output = (row_blend_output_t *)output;

    int num_cols = row_blend_config->num_cols;
    int num_rows = row_blend_config->num_rows;
    int num_bytes = (num_cols * num_rows) << 2;

    uint8_t *src = (uint8_t *)row_blend_input->src;
    uint8_t *dst = (uint8_t *)row_blend_output->dst;

    uint16_t src_scale = row_blend_config->alpha + 1;
    uint16_t dst_scale = 256 - src_scale;

    // load and store everything sequentially
    __vidx_var stride = {1, 0, 0, 0};

    // Dim0: 4 * columns * rows
    _mve_set_dim_count(1);

    __mdvw src_scale_w = _mve_set1_w(src_scale);
    __mdvw dst_scale_w = _mve_set1_w(dst_scale);

    int DIM0_TILE = num_bytes > LANE_NUM ? LANE_NUM : num_bytes;

    int byte = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    while (byte < num_bytes) {
        int remaining_bytes = num_bytes - byte;
        remaining_bytes = remaining_bytes > DIM0_TILE ? DIM0_TILE : remaining_bytes;
        if (remaining_bytes != DIM0_TILE) {
            _mve_set_dim_length(0, remaining_bytes);
        }

        __mdvb src_b = _mve_load_b(src, stride);

        __mdvw src_w = _mve_cvtu_btow(src_b);
        // free src_b
        _mve_free_b();

        __mdvw src_mul_w = _mve_mul_w(src_w, src_scale_w);
        // free src_w
        _mve_free_w();

        __mdvb dst_b = _mve_load_b(dst, stride);

        __mdvw dst_w = _mve_cvtu_btow(dst_b);
        // free dst_b
        _mve_free_b();

        __mdvw dst_mul_w = _mve_mul_w(dst_w, dst_scale_w);
        // free dst_w
        _mve_free_w();

        __mdvw rest_w = _mve_add_w(src_mul_w, dst_mul_w);
        // free src_mul_w and dst_mul_w
        _mve_free_w();
        _mve_free_w();

        __mdvw rest_shifted_w = _mve_shiru_w(rest_w, 8);
        // free rest_w
        _mve_free_w();

        __mdvb rest_shifted_b = _mve_cvt_wtob(rest_shifted_w);
        // free rest_shifted_w
        _mve_free_w();

        _mve_store_b(dst, rest_shifted_b, stride);
        // free rest_shifted_b
        _mve_free_b();

        byte += DIM0_TILE;
        src += DIM0_TILE;
        dst += DIM0_TILE;
    }

    // free src_scale_w and dst_scale_w
    _mve_free_w();
    _mve_free_w();
}