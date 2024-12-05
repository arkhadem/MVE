#include "mve.hpp"
#include "cstdint"
#include "dct.hpp"
#include "kvazaar.hpp"
#include <cstdint>
#include <cstdio>

void dct_mve(int LANE_NUM,
             config_t *config,
             input_t *input,
             output_t *output) {

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t *dct_input = (dct_input_t *)input;
    dct_output_t *dct_output = (dct_output_t *)output;

    int count = dct_config->count;
    int8_t *bitdepth = dct_config->bitdepth;
    int16_t *in = dct_input->input;
    int16_t *out = dct_output->output;

    int16_t *tmp = (int16_t *)malloc(count * 64 * sizeof(int16_t));
    int8_t shift_1st = kvz_g_convert_to_bit[8] + 1 + (bitdepth[0] - 8);
    int8_t shift_2nd = kvz_g_convert_to_bit[8] + 8;

    _mve_set_dim_count(3);

    // Second Dim: Column
    _mve_set_dim_length(0, 8);

    // First Dim: Row
    _mve_set_dim_length(2, 8);

    _mve_set_load_stride(0, 8);
    _mve_set_load_stride(1, 64);
    _mve_set_load_stride(2, 8);

    _mve_set_store_stride(1, 64);

    _mve_set_store_stride(2, 8);

    int count_per_iter = LANE_NUM / 64;

    // LOADING SRC {8, 64, 0}
    __vidx_var src_stride = {3, 3, 0, 0};

    // LOADING PARAM {0, 0, 8}
    __vidx_var kvz_stride = {0, 0, 3, 0};

    // STORING DST {1, 64, 8}
    __vidx_var dst_stride = {1, 3, 3, 0};

    {
        const short *src = in;
        short *dst = tmp;
        int shift = shift_1st;
        int my_count = count;

        while (my_count > 0) {
            // Third Dim: other jobs
            _mve_set_dim_length(1, my_count > count_per_iter ? count_per_iter : my_count);
            my_count -= count_per_iter;

            // R1
            __mdvdw one_v = _mve_set1_dw(1);
            // R0
            __mdvdw result = _mve_shil_dw(one_v, shift - 1);
            // free one_v (R1)
            _mve_free_dw();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // [R1:5]
            __mdvw temp;

            // R1-L
            temp = _mve_load_w(src + 2, src_stride);
            // R2
            __mdvdw src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R1_L)
            _mve_free_w();
            // R1-L
            temp = _mve_load_w(src + 5, src_stride);
            // R3
            __mdvdw src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R1_L)
            _mve_free_w();
            // R4
            __mdvdw o_2 = _mve_sub_dw(src_2_v, src_5_v);
            // free src_2_v (R2) and src_5_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // R1-L
            temp = _mve_load_w(src + 3, src_stride);
            // R2
            __mdvdw src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1-L
            temp = _mve_load_w(src + 4, src_stride);
            // R3
            __mdvdw src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R5
            __mdvdw o_3 = _mve_sub_dw(src_3_v, src_4_v);
            // free src_3_v (R2) and src_4_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R1:3]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 2, kvz_stride);
            // R2
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R3
            __mdvdw mul1_v = _mve_mul_dw(o_2, kvz_2_v);
            // free o_2 (R4) and kvz_2_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R4]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 3, kvz_stride);
            // R2
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R4
            __mdvdw mul2_v = _mve_mul_dw(o_3, kvz_3_v);
            // free o_3 (R5) and kvz_3_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R5]

            // R1
            __mdvdw add_mul12 = _mve_add_dw(mul1_v, mul2_v);
            // free mul1_v (R3) and mul2_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R2-5]

            // R5
            __mdvdw add_res_mul12_v = _mve_add_dw(result, add_mul12);
            // free add_mul12 (R1)
            _mve_free_dw();

            // [R1-4]

            // r1
            result = _mve_assign_dw(result, add_res_mul12_v);
            // free add_res_mul12_v (R5) and result (R0)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2-5]

            _mve_set_all_elements(2);

            // R0-L
            temp = _mve_load_w(src + 0, src_stride);
            // R2
            __mdvdw src_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 7, src_stride);
            // R3
            __mdvdw src_7_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // R4
            __mdvdw ee_eo_o_0 = _mve_sub_dw(src_0_v, src_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R5
            __mdvdw add_07 = _mve_add_dw(src_0_v, src_7_v);
            // free src_0_v (R2) and src_7_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // e[0] for 0/2/4/6
            // R5
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_07);
            // free ee_eo_o_0 (R4) and add_07 (R5)
            _mve_free_dw();
            _mve_free_dw();

            _mve_set_all_elements(2);

            // [R0 R2 R3 R4]

            // R0-L
            temp = _mve_load_w(src + 1, src_stride);
            // R3
            __mdvdw src_1_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 6, src_stride);
            // R4
            __mdvdw src_6_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // R0
            __mdvdw ee_eo_o_1 = _mve_sub_dw(src_1_v, src_6_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw add_16 = _mve_add_dw(src_1_v, src_6_v);
            // free src_1_v (R3) and src_6_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // e[1] for 0/2/4/6
            // R4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_16);
            // free ee_eo_o_1 (R0) and add_16 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            // R0-L
            temp = _mve_load_w(src + 2, src_stride);
            // R2
            src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 5, src_stride);
            // R3
            src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw e_2 = _mve_add_dw(src_2_v, src_5_v);
            // free src_2_v (R2) and src_5_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            // R2
            __mdvdw add_12 = _mve_add_dw(ee_eo_o_1, e_2);
            // ee[1] = e[1] + e[2] for 0/4
            // R3
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_12);
            // free ee_eo_o_1 (R4) and add_12 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R4]

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw sub_12 = _mve_sub_dw(ee_eo_o_1, e_2);
            // free e_2 (R0)
            _mve_free_dw();
            // eo[1] = e[1] - e[2] for 2/6
            // R4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, sub_12);
            // free ee_eo_o_1 (R3) and sub_12 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R0-L
            temp = _mve_load_w(src + 3, src_stride);
            // R2
            src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 4, src_stride);
            // R3
            src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw e_3 = _mve_add_dw(src_3_v, src_4_v);
            // free src_3_v (R2) and src_4_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            // R2
            __mdvdw add_03 = _mve_add_dw(ee_eo_o_0, e_3);
            // ee[0] = e[0] + e[3] for 0/4
            // R3
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_03);
            // free ee_eo_o_0 (R5) and add_03 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R5]

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw sub_03 = _mve_sub_dw(ee_eo_o_0, e_3);
            // free e_3 (R0)
            _mve_free_dw();
            // eo[0] = e[0] - e[3] for 2/6
            // R5
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, sub_03);
            // free ee_eo_o_0 (R3) and sub_03 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            _mve_set_all_elements(2);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            // R2
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R3
            mul1_v = _mve_mul_dw(ee_eo_o_0, kvz_0_v);
            // free ee_eo_o_0 (R5) and kvz_0_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R5]

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            // R5
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R2
            mul2_v = _mve_mul_dw(ee_eo_o_1, kvz_1_v);
            // free ee_eo_o_1 (R4) and kvz_1_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4 R5]

            // R5
            add_12 = _mve_add_dw(mul1_v, mul2_v);
            // free mul1_v (R3) and mul2_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3 R4]

            // R2
            add_res_mul12_v = _mve_add_dw(result, add_12);
            // free add_12 (R5)
            _mve_free_dw();

            // [R0 R3 R4 R5]

            // R3
            result = _mve_assign_dw(result, add_res_mul12_v);
            // free result (R1) and add_res_mul12_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R1 R2 R4 R5]

            // R0
            result = _mve_shirs_dw(result, shift);
            // free result (R3)
            _mve_free_dw();

            // [R1 R2 R3 R4 R5]

            // R1-L
            temp = _mve_cvt_dwtow(result);
            // free result (R0)
            _mve_free_dw();

            _mve_store_w(dst, temp, dst_stride);
            // free temp (R1-L)
            _mve_free_w();

            // [R0 R1 R2 R3 R4 R5]

            dst += (LANE_NUM);
            src += (LANE_NUM);

            mve_flusher();
        }
    }

    {
        const short *src = tmp;
        short *dst = out;
        int shift = shift_2nd;
        int my_count = count;

        while (my_count > 0) {
            // Third Dim: other jobs
            _mve_set_dim_length(1, my_count > count_per_iter ? count_per_iter : my_count);
            my_count -= count_per_iter;

            // R1
            __mdvdw one_v = _mve_set1_dw(1);
            // R0
            __mdvdw result = _mve_shil_dw(one_v, shift - 1);
            // free one_v (R1)
            _mve_free_dw();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // [R1:5]
            __mdvw temp;

            // R1-L
            temp = _mve_load_w(src + 2, src_stride);
            // R2
            __mdvdw src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R1_L)
            _mve_free_w();
            // R1-L
            temp = _mve_load_w(src + 5, src_stride);
            // R3
            __mdvdw src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R1_L)
            _mve_free_w();
            // R4
            __mdvdw o_2 = _mve_sub_dw(src_2_v, src_5_v);
            // free src_2_v (R2) and src_5_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // R1-L
            temp = _mve_load_w(src + 3, src_stride);
            // R2
            __mdvdw src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1-L
            temp = _mve_load_w(src + 4, src_stride);
            // R3
            __mdvdw src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R5
            __mdvdw o_3 = _mve_sub_dw(src_3_v, src_4_v);
            // free src_3_v (R2) and src_4_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R1:3]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 2, kvz_stride);
            // R2
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R3
            __mdvdw mul1_v = _mve_mul_dw(o_2, kvz_2_v);
            // free o_2 (R4) and kvz_2_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R4]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 3, kvz_stride);
            // R2
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R4
            __mdvdw mul2_v = _mve_mul_dw(o_3, kvz_3_v);
            // free o_3 (R5) and kvz_3_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R5]

            // R1
            __mdvdw add_mul12 = _mve_add_dw(mul1_v, mul2_v);
            // free mul1_v (R3) and mul2_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R2-5]

            // R5
            __mdvdw add_res_mul12_v = _mve_add_dw(result, add_mul12);
            // free add_mul12 (R1)
            _mve_free_dw();

            // [R1-4]

            // r1
            result = _mve_assign_dw(result, add_res_mul12_v);
            // free add_res_mul12_v (R5) and result (R0)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2-5]

            _mve_set_all_elements(2);

            // R0-L
            temp = _mve_load_w(src + 0, src_stride);
            // R2
            __mdvdw src_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 7, src_stride);
            // R3
            __mdvdw src_7_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // R4
            __mdvdw ee_eo_o_0 = _mve_sub_dw(src_0_v, src_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R5
            __mdvdw add_07 = _mve_add_dw(src_0_v, src_7_v);
            // free src_0_v (R2) and src_7_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // e[0] for 0/2/4/6
            // R5
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_07);
            // free ee_eo_o_0 (R4) and add_07 (R5)
            _mve_free_dw();
            _mve_free_dw();

            _mve_set_all_elements(2);

            // [R0 R2 R3 R4]

            // R0-L
            temp = _mve_load_w(src + 1, src_stride);
            // R3
            __mdvdw src_1_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 6, src_stride);
            // R4
            __mdvdw src_6_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            // R0
            __mdvdw ee_eo_o_1 = _mve_sub_dw(src_1_v, src_6_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw add_16 = _mve_add_dw(src_1_v, src_6_v);
            // free src_1_v (R3) and src_6_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // e[1] for 0/2/4/6
            // R4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_16);
            // free ee_eo_o_1 (R0) and add_16 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            // R0-L
            temp = _mve_load_w(src + 2, src_stride);
            // R2
            src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 5, src_stride);
            // R3
            src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw e_2 = _mve_add_dw(src_2_v, src_5_v);
            // free src_2_v (R2) and src_5_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            // R2
            __mdvdw add_12 = _mve_add_dw(ee_eo_o_1, e_2);
            // ee[1] = e[1] + e[2] for 0/4
            // R3
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_12);
            // free ee_eo_o_1 (R4) and add_12 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R4]

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw sub_12 = _mve_sub_dw(ee_eo_o_1, e_2);
            // free e_2 (R0)
            _mve_free_dw();
            // eo[1] = e[1] - e[2] for 2/6
            // R4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, sub_12);
            // free ee_eo_o_1 (R3) and sub_12 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            // R0-L
            temp = _mve_load_w(src + 3, src_stride);
            // R2
            src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(src + 4, src_stride);
            // R3
            src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw e_3 = _mve_add_dw(src_3_v, src_4_v);
            // free src_3_v (R2) and src_4_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            // R2
            __mdvdw add_03 = _mve_add_dw(ee_eo_o_0, e_3);
            // ee[0] = e[0] + e[3] for 0/4
            // R3
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_03);
            // free ee_eo_o_0 (R5) and add_03 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R5]

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            // R2
            __mdvdw sub_03 = _mve_sub_dw(ee_eo_o_0, e_3);
            // free e_3 (R0)
            _mve_free_dw();
            // eo[0] = e[0] - e[3] for 2/6
            // R5
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, sub_03);
            // free ee_eo_o_0 (R3) and sub_03 (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3]

            _mve_set_all_elements(2);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            // R2
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R3
            mul1_v = _mve_mul_dw(ee_eo_o_0, kvz_0_v);
            // free ee_eo_o_0 (R5) and kvz_0_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R5]

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            // R5
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R2
            mul2_v = _mve_mul_dw(ee_eo_o_1, kvz_1_v);
            // free ee_eo_o_1 (R4) and kvz_1_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4 R5]

            // R5
            add_12 = _mve_add_dw(mul1_v, mul2_v);
            // free mul1_v (R3) and mul2_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R3 R4]

            // R2
            add_res_mul12_v = _mve_add_dw(result, add_12);
            // free add_12 (R5)
            _mve_free_dw();

            // [R0 R3 R4 R5]

            // R3
            result = _mve_assign_dw(result, add_res_mul12_v);
            // free result (R1) and add_res_mul12_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R1 R2 R4 R5]

            // R0
            result = _mve_shirs_dw(result, shift);
            // free result (R3)
            _mve_free_dw();

            // [R1 R2 R3 R4 R5]

            // R1-L
            temp = _mve_cvt_dwtow(result);
            // free result (R0)
            _mve_free_dw();

            _mve_store_w(dst, temp, dst_stride);
            // free temp (R1-L)
            _mve_free_w();

            // [R0 R1 R2 R3 R4 R5]

            dst += (LANE_NUM);
            src += (LANE_NUM);

            mve_flusher();
        }
    }
}