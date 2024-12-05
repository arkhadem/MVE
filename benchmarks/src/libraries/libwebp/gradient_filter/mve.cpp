#include "mve.hpp"
#include "mve_kernels.hpp"

#include "gradient_filter.hpp"

void gradient_filter_mve(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    gradient_filter_config_t *gradient_filter_config = (gradient_filter_config_t *)config;
    gradient_filter_input_t *gradient_filter_input = (gradient_filter_input_t *)input;
    gradient_filter_output_t *gradient_filter_output = (gradient_filter_output_t *)output;

    int stride = gradient_filter_config->stride;
    uint8_t *in = gradient_filter_input->in + stride + 1;
    uint8_t *preds = gradient_filter_input->in + 1;
    uint8_t *out = gradient_filter_output->out;

    // Dim0: cols
    // Dim0: rows
    _mve_set_dim_count(2);

    __mdvw min_w = _mve_set1_w(0);
    __mdvw max_w = _mve_set1_w(255);

    _mve_set_load_stride(1, stride);
    __vidx_var input_stride = {1, 3, 0, 0};

    _mve_set_store_stride(1, stride);
    __vidx_var output_stride = {1, 3, 0, 0};

    int num_rows = gradient_filter_config->num_rows;
    int num_cols = gradient_filter_config->num_cols;

    int DIM0_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM1_TILE = LANE_NUM / DIM0_TILE;

    int row = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM1_TILE ? DIM1_TILE : remaining_rows;
        if (remaining_rows != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_rows);
        }

        uint8_t *in_addr = in;
        uint8_t *out_addr = out;
        uint8_t *preds_addr = preds;

        int col = 0;
        _mve_set_dim_length(0, DIM0_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM0_TILE ? DIM0_TILE : remaining_cols;
            if (remaining_cols != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_cols);
            }

            __mdvb a_b = _mve_load_b(in_addr - 1, input_stride);
            __mdvw a_w = _mve_cvtu_btow(a_b);
            // free a_b
            _mve_free_b();
            __mdvb b_b = _mve_load_b(preds_addr, input_stride);
            __mdvw b_w = _mve_cvtu_btow(b_b);
            // free b_b
            _mve_free_b();
            __mdvb c_b = _mve_load_b(preds_addr - 1, input_stride);
            __mdvw c_w = _mve_cvtu_btow(c_b);
            // free c_b
            _mve_free_b();

            __mdvw ab_w = _mve_add_w(a_w, b_w);
            // free a_w and b_w
            _mve_free_w();
            _mve_free_w();
            __mdvw abc_w = _mve_sub_w(ab_w, c_w);
            // free ab_w and c_w
            _mve_free_w();
            _mve_free_w();

            __mdvw res_min_w = _mve_min_w(abc_w, max_w);
            // free abc_w
            _mve_free_w();

            __mdvw g_w = _mve_max_w(res_min_w, min_w);
            // free res_min_w
            _mve_free_w();

            __mdvb in_b = _mve_load_b(in_addr, input_stride);
            __mdvw in_w = _mve_cvtu_btow(in_b);
            // free in_b
            _mve_free_b();

            __mdvw sub_w = _mve_sub_w(in_w, g_w);
            // free in_w and g_w
            _mve_free_w();
            _mve_free_w();

            __mdvb sub_b = _mve_cvt_wtob(sub_w);
            // free sub_w
            _mve_free_w();

            _mve_store_b(out_addr, sub_b, output_stride);
            // free sub_b
            _mve_free_b();

            col += remaining_cols;
            in_addr += remaining_cols;
            out_addr += remaining_cols;
            preds_addr += remaining_cols;
        }

        row += remaining_rows;
        int remaining_stride = remaining_rows * stride;
        in += remaining_stride;
        out += remaining_stride;
        preds += remaining_stride;
    }

    // free min_w and max_w
    _mve_free_w();
    _mve_free_w();
}