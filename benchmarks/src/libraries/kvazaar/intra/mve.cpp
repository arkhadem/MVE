#include "mve.hpp"
#include "cstdint"
#include "intra.hpp"
#include "kvazaar.hpp"
#include <cstdint>
#include <cstdio>

void intra_mve(int LANE_NUM,
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

    // First Dim: Row
    _mve_set_dim_length(0, width);

    // Second Dim: Column
    _mve_set_dim_length(1, width);

    // Load stride for different CUs
    _mve_set_load_stride(2, 2 * width + 1);

    // Store stride for different CUs
    _mve_set_store_stride(2, width * width);

    // rt[w + 1] stride = {0, 0, 2w+1}
    __vidx_var rtl_stride = {0, 0, 3, 0};

    // rl[y+1] stride = {0, 1, 2w+1}
    __vidx_var rl_stride = {0, 1, 3, 0};

    // rt[x+1] stride = {1, 0, 2w+1}
    __vidx_var rt_stride = {1, 0, 3, 0};

    // (x+1) stride
    __vidx_var xp1_stride = {1, 0, 0, 0};

    // (y+1) stride
    __vidx_var yp1_stride = {0, 1, 0, 0};

    // store stride
    __vidx_var st_stride = {1, 2, 2, 0};

    int16_t xy_plus_one[32];

    int idx = 0;
    for (int xy = 0; xy < width; xy++) {
        xy_plus_one[idx] = (xy + 1);
        idx++;
    }

    int count_per_iter = LANE_NUM / width / width;
    int input_size_per_iter = count_per_iter * (2 * width + 1);
    int output_size_per_iter = count_per_iter * (width * width);

    _mve_set_dim_length(2, count_per_iter);

    // R0 = w
    __mdvw width_v = _mve_set1_w(width);

    //R1 = (x+1)
    __mdvw x_plus_one_v = _mve_load_w(xy_plus_one, xp1_stride);

    // R2 = (y+1)
    __mdvw y_plus_one_v = _mve_load_w(xy_plus_one, yp1_stride);

    // R3 = w - (x+1)
    __mdvw width_minus_x_plus_one_v = _mve_sub_w(width_v, x_plus_one_v);

    // R4 = w - (y+1)
    __mdvw width_minus_y_plus_one_v = _mve_sub_w(width_v, y_plus_one_v);

    __mdvb temp;

    while (count > 0) {
        // Third Dim: CUs
        _mve_set_dim_length(2, count > count_per_iter ? count_per_iter : count);
        count -= count_per_iter;

        // [R5:11]

        // R11-L
        temp = _mve_load_b(&ref_top[width + 1], rtl_stride);
        // R5 = rt[w+1]
        __mdvw top_right_v = _mve_cvtu_btow(temp);
        // free temp (R11-L)
        _mve_free_b();

        // R6 = rt[w+1] * (x+1)
        __mdvw hor_iter_v = _mve_mul_w(top_right_v, x_plus_one_v);
        // free top_right_v (R5)
        _mve_free_w();

        // [R5 - R7:15]

        // R11-L
        temp = _mve_load_b((ref_left + 1), rl_stride);
        // R5 = rl[y+1]
        __mdvw left_v = _mve_cvtu_btow(temp);
        // free temp (R11-L)
        _mve_free_b();

        // R7 = (w - (x+1)) * rl[y+1]
        __mdvw hor_v = _mve_mul_w(width_minus_x_plus_one_v, left_v);
        // free left_v (R5)
        _mve_free_w();

        // [R5 - R8:15]

        // R5 = (w - (x+1)) * rl[y+1] + rt[w+1] * (x+1)
        hor_v = _mve_add_w(hor_v, hor_iter_v);
        // free hor_iter_v (R6) and hor_v (R7)
        _mve_free_w();
        _mve_free_w();

        // [R6:15]

        // R11-L
        temp = _mve_load_b((ref_top + 1), rt_stride);
        // R6 = rt[x+1]
        __mdvw top_v = _mve_cvtu_btow(temp);
        // free temp (R11-L)
        _mve_free_b();

        // R7 = (w - (y+1)) * rt[x+1]
        __mdvw ver_v = _mve_mul_w(width_minus_y_plus_one_v, top_v);
        // free top_v (R6)
        _mve_free_w();

        // [R6 - R8:15]

        // R11-L
        temp = _mve_load_b(&ref_left[width + 1], rtl_stride);
        // R6 = rl[w+1]
        __mdvw bottom_left_v = _mve_cvtu_btow(temp);
        // free temp (R11-L)
        _mve_free_b();

        // R8 = rl[w+1] * (y+1)
        __mdvw ver_iter_v = _mve_mul_w(bottom_left_v, y_plus_one_v);
        // free bottom_left_v (R6)
        _mve_free_w();

        // [R6 - R9:15]

        // R6 = (w - (y+1)) * rt[x+1] + rl[w+1] * (y+1)
        ver_v = _mve_add_w(ver_v, ver_iter_v);
        // free ver_v (R7) and ver_iter_v (R8)
        _mve_free_w();
        _mve_free_w();

        // [R7:15]

        // R7 = (w - (x+1)) * rl[y+1] + rt[w+1] * (x+1) + (w - (y+1)) * rt[x+1] + rl[w+1] * (y+1)
        __mdvw total_v = _mve_add_w(hor_v, ver_v);
        // free hor_v (R5) and ver_v (R6)
        _mve_free_w();
        _mve_free_w();

        // [R5:6 - R8:15]

        // R5 = (w - (x+1)) * rl[y+1] + rt[w+1] * (x+1) +
        //      (w - (y+1)) * rt[x+1] + rl[w+1] * (y+1) + w
        total_v = _mve_add_w(total_v, width_v);
        // free total_v (R7)
        _mve_free_w();

        // [R6:15]

        // R6 = R3 >> lw
        total_v = _mve_shirs_w(total_v, log2_width + 1);
        // free total_v (R5)
        _mve_free_w();

        // [R5 - R7:15]

        // R11-L
        temp = _mve_cvt_wtob(total_v);
        // free total_v (R6)
        _mve_free_w();

        _mve_store_b(dst, temp, st_stride);
        // free temp (R11-L)
        _mve_free_b();

        // [R5:15]

        ref_top += input_size_per_iter;
        ref_left += input_size_per_iter;
        dst += output_size_per_iter;

#ifndef COMPARE
        mve_flusher();
#endif
    }

    // free width_v (R0), x_plus_one_v (R1), y_plus_one_v (R2), width_minus_x_plus_one_v (R3), width_minus_y_plus_one_v (R4)
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
}