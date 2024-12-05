#include "dct.hpp"
#include "kvazaar.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

void dct_rvv(int LANE_NUM,
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

            __mdvdw one_v = _mve_set1_dw(1);
            __mdvdw result = _mve_shil_dw(one_v, shift - 1);

            __mdvw kvz_2_w_v = _mve_set1_w(0);
            __mdvw kvz_3_w_v = _mve_set1_w(0);
            __mdvw src_2_w_v = _mve_set1_w(0);
            __mdvw src_5_w_v = _mve_set1_w(0);
            __mdvw src_3_w_v = _mve_set1_w(0);
            __mdvw src_4_w_v = _mve_set1_w(0);

            for (int element_idx = 1; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 2, src_stride));
                src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 5, src_stride));
                src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 3, src_stride));
                src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 4, src_stride));
                kvz_2_w_v = _mve_assign_w(kvz_2_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 2, kvz_stride));
                kvz_3_w_v = _mve_assign_w(kvz_3_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 3, kvz_stride));
            }

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw src_2_v = _mve_cvts_wtodw(src_2_w_v);
            __mdvdw src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw o_2 = _mve_sub_dw(src_2_v, src_5_v);
            __mdvdw kvz_2_v = _mve_cvts_wtodw(kvz_2_w_v);
            __mdvdw mul1_v = _mve_mul_dw(o_2, kvz_2_v);

            __mdvdw src_3_v = _mve_cvts_wtodw(src_3_w_v);
            __mdvdw src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw o_3 = _mve_sub_dw(src_3_v, src_4_v);
            __mdvdw kvz_3_v = _mve_cvts_wtodw(kvz_3_w_v);
            __mdvdw mul2_v = _mve_mul_dw(o_3, kvz_3_v);

            __mdvdw add_mul12 = _mve_add_dw(mul1_v, mul2_v);

            __mdvdw add_res_mul12_v = _mve_add_dw(result, add_mul12);

            result = _mve_assign_dw(result, add_res_mul12_v);

            _mve_set_all_elements(2);

            __mdvw src_0_w_v = _mve_set1_w(0);
            __mdvw src_7_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                src_0_w_v = _mve_assign_w(src_0_w_v, _mve_load_w(src + 0, src_stride));
                src_7_w_v = _mve_assign_w(src_7_w_v, _mve_load_w(src + 7, src_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw src_0_v = _mve_cvts_wtodw(src_0_w_v);
            __mdvdw src_7_v = _mve_cvts_wtodw(src_7_w_v);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw ee_eo_o_0 = _mve_sub_dw(src_0_v, src_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            __mdvdw add_07 = _mve_add_dw(src_0_v, src_7_v);

            // e[0] for 0/2/4/6
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_07);

            _mve_set_all_elements(2);

            __mdvw src_1_w_v = _mve_set1_w(0);
            __mdvw src_6_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                src_1_w_v = _mve_assign_w(src_1_w_v, _mve_load_w(src + 1, src_stride));
                src_6_w_v = _mve_assign_w(src_6_w_v, _mve_load_w(src + 6, src_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw src_1_v = _mve_cvts_wtodw(src_1_w_v);
            __mdvdw src_6_v = _mve_cvts_wtodw(src_6_w_v);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw ee_eo_o_1 = _mve_sub_dw(src_1_v, src_6_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            __mdvdw add_16 = _mve_add_dw(src_1_v, src_6_v);

            // e[1] for 0/2/4/6
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_16);

            src_2_w_v = _mve_set1_w(0);
            src_5_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 2, src_stride));
                src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 5, src_stride));
            }

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_2_v = _mve_cvts_wtodw(src_2_w_v);
            src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw e_2 = _mve_add_dw(src_2_v, src_5_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            __mdvdw add_12 = _mve_add_dw(ee_eo_o_1, e_2);
            // ee[1] = e[1] + e[2] for 0/4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_12);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            __mdvdw sub_12 = _mve_sub_dw(ee_eo_o_1, e_2);
            // eo[1] = e[1] - e[2] for 2/6
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, sub_12);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_3_w_v = _mve_set1_w(0);
            src_4_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 3, src_stride));
                src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 4, src_stride));
            }

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_3_v = _mve_cvts_wtodw(src_3_w_v);
            src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw e_3 = _mve_add_dw(src_3_v, src_4_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            __mdvdw add_03 = _mve_add_dw(ee_eo_o_0, e_3);
            // ee[0] = e[0] + e[3] for 0/4
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_03);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            __mdvdw sub_03 = _mve_sub_dw(ee_eo_o_0, e_3);
            // eo[0] = e[0] - e[3] for 2/6
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, sub_03);

            _mve_set_all_elements(2);

            __mdvw kvz_0_w_v = _mve_set1_w(0);
            __mdvw kvz_1_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                kvz_0_w_v = _mve_assign_w(kvz_0_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride));
                kvz_1_w_v = _mve_assign_w(kvz_1_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw kvz_0_v = _mve_cvts_wtodw(kvz_0_w_v);
            mul1_v = _mve_mul_dw(ee_eo_o_0, kvz_0_v);

            __mdvdw kvz_1_v = _mve_cvts_wtodw(kvz_1_w_v);
            mul2_v = _mve_mul_dw(ee_eo_o_1, kvz_1_v);

            add_12 = _mve_add_dw(mul1_v, mul2_v);

            result = _mve_add_dw(result, add_12);

            result = _mve_shirs_dw(result, shift);

            __mdvw result_w = _mve_cvt_dwtow(result);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    _mve_store_w(dst, result_w, dst_stride);
                }
            }
            _mve_set_all_elements(0);
            _mve_set_all_elements(2);

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

            __mdvdw one_v = _mve_set1_dw(1);
            __mdvdw result = _mve_shil_dw(one_v, shift - 1);

            __mdvw kvz_2_w_v = _mve_set1_w(0);
            __mdvw kvz_3_w_v = _mve_set1_w(0);
            __mdvw src_2_w_v = _mve_set1_w(0);
            __mdvw src_5_w_v = _mve_set1_w(0);
            __mdvw src_3_w_v = _mve_set1_w(0);
            __mdvw src_4_w_v = _mve_set1_w(0);

            for (int element_idx = 1; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 2, src_stride));
                src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 5, src_stride));
                src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 3, src_stride));
                src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 4, src_stride));
                kvz_2_w_v = _mve_assign_w(kvz_2_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 2, kvz_stride));
                kvz_3_w_v = _mve_assign_w(kvz_3_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 3, kvz_stride));
            }

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw src_2_v = _mve_cvts_wtodw(src_2_w_v);
            __mdvdw src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw o_2 = _mve_sub_dw(src_2_v, src_5_v);
            __mdvdw kvz_2_v = _mve_cvts_wtodw(kvz_2_w_v);
            __mdvdw mul1_v = _mve_mul_dw(o_2, kvz_2_v);

            __mdvdw src_3_v = _mve_cvts_wtodw(src_3_w_v);
            __mdvdw src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw o_3 = _mve_sub_dw(src_3_v, src_4_v);
            __mdvdw kvz_3_v = _mve_cvts_wtodw(kvz_3_w_v);
            __mdvdw mul2_v = _mve_mul_dw(o_3, kvz_3_v);

            __mdvdw add_mul12 = _mve_add_dw(mul1_v, mul2_v);

            __mdvdw add_res_mul12_v = _mve_add_dw(result, add_mul12);

            result = _mve_assign_dw(result, add_res_mul12_v);

            _mve_set_all_elements(2);

            __mdvw src_0_w_v = _mve_set1_w(0);
            __mdvw src_7_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                src_0_w_v = _mve_assign_w(src_0_w_v, _mve_load_w(src + 0, src_stride));
                src_7_w_v = _mve_assign_w(src_7_w_v, _mve_load_w(src + 7, src_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw src_0_v = _mve_cvts_wtodw(src_0_w_v);
            __mdvdw src_7_v = _mve_cvts_wtodw(src_7_w_v);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw ee_eo_o_0 = _mve_sub_dw(src_0_v, src_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            __mdvdw add_07 = _mve_add_dw(src_0_v, src_7_v);

            // e[0] for 0/2/4/6
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_07);

            _mve_set_all_elements(2);

            __mdvw src_1_w_v = _mve_set1_w(0);
            __mdvw src_6_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                src_1_w_v = _mve_assign_w(src_1_w_v, _mve_load_w(src + 1, src_stride));
                src_6_w_v = _mve_assign_w(src_6_w_v, _mve_load_w(src + 6, src_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw src_1_v = _mve_cvts_wtodw(src_1_w_v);
            __mdvdw src_6_v = _mve_cvts_wtodw(src_6_w_v);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 7);

            __mdvdw ee_eo_o_1 = _mve_sub_dw(src_1_v, src_6_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            __mdvdw add_16 = _mve_add_dw(src_1_v, src_6_v);

            // e[1] for 0/2/4/6
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_16);

            src_2_w_v = _mve_set1_w(0);
            src_5_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 2, src_stride));
                src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 5, src_stride));
            }

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_2_v = _mve_cvts_wtodw(src_2_w_v);
            src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw e_2 = _mve_add_dw(src_2_v, src_5_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            __mdvdw add_12 = _mve_add_dw(ee_eo_o_1, e_2);
            // ee[1] = e[1] + e[2] for 0/4
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, add_12);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            __mdvdw sub_12 = _mve_sub_dw(ee_eo_o_1, e_2);
            // eo[1] = e[1] - e[2] for 2/6
            ee_eo_o_1 = _mve_assign_dw(ee_eo_o_1, sub_12);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_3_w_v = _mve_set1_w(0);
            src_4_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 2) {
                _mve_set_only_element(2, element_idx);
                src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 3, src_stride));
                src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 4, src_stride));
            }

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 6);

            src_3_v = _mve_cvts_wtodw(src_3_w_v);
            src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw e_3 = _mve_add_dw(src_3_v, src_4_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 4);

            __mdvdw add_03 = _mve_add_dw(ee_eo_o_0, e_3);
            // ee[0] = e[0] + e[3] for 0/4
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, add_03);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 6);

            __mdvdw sub_03 = _mve_sub_dw(ee_eo_o_0, e_3);
            // eo[0] = e[0] - e[3] for 2/6
            ee_eo_o_0 = _mve_assign_dw(ee_eo_o_0, sub_03);

            _mve_set_all_elements(2);

            __mdvw kvz_0_w_v = _mve_set1_w(0);
            __mdvw kvz_1_w_v = _mve_set1_w(0);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                kvz_0_w_v = _mve_assign_w(kvz_0_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride));
                kvz_1_w_v = _mve_assign_w(kvz_1_w_v, _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride));
            }

            _mve_set_all_elements(2);

            __mdvdw kvz_0_v = _mve_cvts_wtodw(kvz_0_w_v);
            mul1_v = _mve_mul_dw(ee_eo_o_0, kvz_0_v);

            __mdvdw kvz_1_v = _mve_cvts_wtodw(kvz_1_w_v);
            mul2_v = _mve_mul_dw(ee_eo_o_1, kvz_1_v);

            add_12 = _mve_add_dw(mul1_v, mul2_v);

            result = _mve_add_dw(result, add_12);

            result = _mve_shirs_dw(result, shift);

            __mdvw result_w = _mve_cvt_dwtow(result);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    _mve_store_w(dst, result_w, dst_stride);
                }
            }
            _mve_set_all_elements(0);
            _mve_set_all_elements(2);

            dst += (LANE_NUM);
            src += (LANE_NUM);

            mve_flusher();
        }
    }
}