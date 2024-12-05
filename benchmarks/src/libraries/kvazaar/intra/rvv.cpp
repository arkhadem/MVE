#include "intra.hpp"
#include "kvazaar.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

void intra_rvv(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {

    intra_config_t *intra_config = (intra_config_t *)config;
    intra_input_t *intra_input = (intra_input_t *)input;
    intra_output_t *intra_output = (intra_output_t *)output;

    int count = intra_config->count;
    const int_fast8_t log2_width = intra_config->log2_width;
    const int_fast8_t width = intra_config->width;
    kvz_pixel *ref_top = intra_input->ref_top;
    kvz_pixel *ref_left = intra_input->ref_left;
    kvz_pixel *dst = intra_output->dst;

    _mve_set_dim_count(3);

    int count_per_iter = LANE_NUM / width / width;
    int input_size_per_iter = count_per_iter * (2 * width + 1);
    int output_size_per_iter = count_per_iter * (width * width);

    // Second Dim: Columns (Y)
    _mve_set_dim_length(0, width);

    // First Dim: Input cells
    _mve_set_dim_length(1, count_per_iter);

    // First Dim: Rows (X)
    _mve_set_dim_length(2, width);

    // Load stride for different CUs
    _mve_set_load_stride(1, 2 * width + 1);

    // Stride = {0, 2w+1, 0}
    __vidx_var rtl_stride = {0, 3, 0, 0};

    // Stide = {1, 2w+1, 0}
    __vidx_var rl_stride = {1, 3, 0, 0};

    // Stide = {0, 2w+1, 1}
    __vidx_var rt_stride = {0, 3, 1, 0};

    // Store stride for different CUs
    _mve_set_store_stride(1, width * width);

    // Store stride for different CUs
    _mve_set_store_stride(0, width);

    // store stride = {w, w^2, 1}
    __vidx_var st_stride = {3, 3, 1, 0};

    __mdvb temp;

    while (count > 0) {
        // Third Dim: CUs
        _mve_set_dim_length(1, count > count_per_iter ? count_per_iter : count);
        count -= count_per_iter;

        __mdvb top_right_b_v = _mve_set1_b(0);
        __mdvb bottom_left_b_v = _mve_set1_b(0);
        __mdvb left_b_v = _mve_set1_b(0);
        __mdvb top_b_v = _mve_set1_b(0);
        __mdvw x_plus_one_v = _mve_set1_w(0);
        __mdvw y_plus_one_v = _mve_set1_w(0);
        __mdvw width_v = _mve_set1_w(width);
        for (int x = 0; x < 8; x++) {
            _mve_set_only_element(2, x);
            for (int y = 0; y < 8; y++) {
                _mve_set_only_element(0, y);
                top_right_b_v = _mve_assign_b(top_right_b_v, _mve_load_b(&ref_top[width + 1], rtl_stride));
                bottom_left_b_v = _mve_assign_b(bottom_left_b_v, _mve_load_b(&ref_left[width + 1], rtl_stride));
                left_b_v = _mve_assign_b(left_b_v, _mve_load_b((ref_left + 1), rl_stride));
                top_b_v = _mve_assign_b(top_b_v, _mve_load_b((ref_top + 1), rt_stride));
            }
        }
        _mve_set_all_elements(2);
        _mve_set_all_elements(0);

        for (int x = 0; x < 8; x++) {
            _mve_set_only_element(2, x);
            x_plus_one_v = _mve_assign_w(x_plus_one_v, _mve_set1_w(x + 1));
        }
        _mve_set_all_elements(2);

        for (int y = 0; y < 8; y++) {
            _mve_set_only_element(0, y);
            y_plus_one_v = _mve_assign_w(y_plus_one_v, _mve_set1_w(y + 1));
        }
        _mve_set_all_elements(0);

        __mdvw width_minus_x_plus_one_v = _mve_sub_w(width_v, x_plus_one_v);

        __mdvw width_minus_y_plus_one_v = _mve_sub_w(width_v, y_plus_one_v);

        __mdvw top_right_v = _mve_cvtu_btow(top_right_b_v);
        __mdvw bottom_left_v = _mve_cvtu_btow(bottom_left_b_v);
        __mdvw left_v = _mve_cvtu_btow(left_b_v);
        __mdvw top_v = _mve_cvtu_btow(top_b_v);

        __mdvw hor_iter_v = _mve_mul_w(top_right_v, x_plus_one_v);

        __mdvw hor_v = _mve_mul_w(width_minus_x_plus_one_v, left_v);

        hor_v = _mve_add_w(hor_v, hor_iter_v);

        __mdvw ver_v = _mve_mul_w(width_minus_y_plus_one_v, top_v);

        __mdvw ver_iter_v = _mve_mul_w(bottom_left_v, y_plus_one_v);

        ver_v = _mve_add_w(ver_v, ver_iter_v);

        __mdvw total_v = _mve_add_w(hor_v, ver_v);

        //      (w - (y+1)) * rt[x+1] + rl[w+1] * (y+1) + w
        total_v = _mve_add_w(total_v, width_v);

        total_v = _mve_shirs_w(total_v, log2_width + 1);

        temp = _mve_cvt_wtob(total_v);

        for (int x = 0; x < 8; x++) {
            _mve_set_only_element(2, x);
            _mve_store_b(dst, temp, st_stride);
        }
        _mve_set_all_elements(2);

        ref_top += input_size_per_iter;
        ref_left += input_size_per_iter;
        dst += output_size_per_iter;

        mve_flusher();
    }
}