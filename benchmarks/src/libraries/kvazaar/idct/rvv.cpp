#include "idct.hpp"
#include "kvazaar.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

void idct_rvv(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    idct_config_t *idct_config = (idct_config_t *)config;
    idct_input_t *idct_input = (idct_input_t *)input;
    idct_output_t *idct_output = (idct_output_t *)output;

    int count = idct_config->count;
    int8_t *bitdepth = idct_config->bitdepth;
    int16_t *in = idct_input->input;
    int16_t *out = idct_output->output;

    int16_t *tmp = (int16_t *)malloc(count * 64 * sizeof(int16_t));
    int32_t shift_1st = 7;
    int32_t shift_2nd = 12 - (bitdepth[0] - 8);

    _mve_set_dim_count(3);

    // Second Dim: Column
    _mve_set_dim_length(0, 8);

    // First Dim: Row
    _mve_set_dim_length(2, 8);

    _mve_set_load_stride(1, 64);
    _mve_set_store_stride(0, 8);
    _mve_set_store_stride(1, 64);

    int count_per_iter = LANE_NUM / 64;

    // __vidx_var shift_stride = {0, 0, 1, 0};

    // LOADING SRC {1, 64, 0}
    __vidx_var src_stride = {1, 3, 0, 0};

    // LOADING PARAMETERS {0, 0, 0}
    __vidx_var kvz_stride = {0, 0, 0, 0};

    // STORING DST {8, 64, 1}
    __vidx_var dst_stride = {3, 3, 1, 0};

    __mdvw temp, temp2;

    {
        const short *src = in;
        short *dst = tmp;
        int shift = shift_1st;
        int my_count = count;

        while (my_count > 0) {
            // Third Dim: other jobs
            _mve_set_dim_length(1, my_count > count_per_iter ? count_per_iter : my_count);
            my_count -= count_per_iter;

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 7);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 16, kvz_stride);
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 32, kvz_stride);
            __mdvdw kvz_4_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 48, kvz_stride);
            __mdvdw kvz_6_v = _mve_cvts_wtodw(temp);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            __mdvdw kvz_1_temp_v = _mve_cvts_wtodw(temp);
            kvz_0_v = _mve_assign_dw(kvz_0_v, kvz_1_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 17, kvz_stride);
            __mdvdw kvz_17_temp_v = _mve_cvts_wtodw(temp);
            kvz_2_v = _mve_assign_dw(kvz_2_v, kvz_17_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 33, kvz_stride);
            __mdvdw kvz_33_temp_v = _mve_cvts_wtodw(temp);
            kvz_4_v = _mve_assign_dw(kvz_4_v, kvz_33_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 49, kvz_stride);
            __mdvdw kvz_49_temp_v = _mve_cvts_wtodw(temp);
            kvz_6_v = _mve_assign_dw(kvz_6_v, kvz_49_temp_v);

            __mdvw src_0_w_v = _mve_set1_w(0);
            __mdvw src_2_w_v = _mve_set1_w(0);
            __mdvw src_4_w_v = _mve_set1_w(0);
            __mdvw src_6_w_v = _mve_set1_w(0);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    src_0_w_v = _mve_assign_w(src_0_w_v, _mve_load_w(src + 0, src_stride));
                    src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 16, src_stride));
                    src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 32, src_stride));
                    src_6_w_v = _mve_assign_w(src_6_w_v, _mve_load_w(src + 48, src_stride));
                }
            }
            _mve_set_all_elements(2);
            _mve_set_all_elements(0);

            __mdvdw src_0_v = _mve_cvts_wtodw(src_0_w_v);
            __mdvdw mul_0_v = _mve_mul_dw(kvz_0_v, src_0_v);

            __mdvdw src_2_v = _mve_cvts_wtodw(src_2_w_v);
            __mdvdw mul_2_v = _mve_mul_dw(kvz_2_v, src_2_v);

            __mdvdw src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw mul_4_v = _mve_mul_dw(kvz_4_v, src_4_v);

            __mdvdw src_6_v = _mve_cvts_wtodw(src_6_w_v);
            __mdvdw mul_6_v = _mve_mul_dw(kvz_6_v, src_6_v);

            __mdvdw eo_v = _mve_add_dw(mul_2_v, mul_6_v);
            __mdvdw ee_v = _mve_add_dw(mul_0_v, mul_4_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            __mdvdw e_v = _mve_add_dw(ee_v, eo_v);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 5);

            __mdvdw ee_sub_eo_v = _mve_sub_dw(ee_v, eo_v);
            e_v = _mve_assign_dw(e_v, ee_sub_eo_v);

            // kvz_1_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 8, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 9, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 10, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 11, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp2);

            // kvz_3_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 24, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 25, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 26, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 27, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp2);

            // kvz_5_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 40, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 41, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 42, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 43, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_5_v = _mve_cvts_wtodw(temp2);

            // kvz_7_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 56, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 57, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 58, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 59, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_7_v = _mve_cvts_wtodw(temp2);

            // Swap with [R4 and R5]

            __mdvw src_1_w_v = _mve_set1_w(0);
            __mdvw src_3_w_v = _mve_set1_w(0);
            __mdvw src_5_w_v = _mve_set1_w(0);
            __mdvw src_7_w_v = _mve_set1_w(0);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    src_1_w_v = _mve_assign_w(src_1_w_v, _mve_load_w(src + 8, src_stride));
                    src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 24, src_stride));
                    src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 40, src_stride));
                    src_7_w_v = _mve_assign_w(src_7_w_v, _mve_load_w(src + 56, src_stride));
                }
            }
            _mve_set_all_elements(2);
            _mve_set_all_elements(0);

            __mdvdw src_1_v = _mve_cvts_wtodw(src_1_w_v);
            __mdvdw mul1_v = _mve_mul_dw(kvz_1_v, src_1_v);

            __mdvdw src_3_v = _mve_cvts_wtodw(src_3_w_v);
            __mdvdw mul3_v = _mve_mul_dw(kvz_3_v, src_3_v);
            __mdvdw mul1_3_v = _mve_add_dw(mul1_v, mul3_v);

            __mdvdw src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw mul5_v = _mve_mul_dw(kvz_5_v, src_5_v);

            __mdvdw src_7_v = _mve_cvts_wtodw(src_7_w_v);
            __mdvdw mul7_v = _mve_mul_dw(kvz_7_v, src_7_v);
            __mdvdw mul5_7_v = _mve_add_dw(mul5_v, mul7_v);

            __mdvdw o_v = _mve_add_dw(mul1_3_v, mul5_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 3);

            __mdvdw result = _mve_add_dw(e_v, o_v);

            _mve_set_only_element(2, 4);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            __mdvdw e_sub_o_v = _mve_sub_dw(e_v, o_v);

            result = _mve_assign_dw(result, e_sub_o_v);

            _mve_set_all_elements(2);

            __mdvdw one_v = _mve_set1_dw(1);
            __mdvdw add_v = _mve_shil_dw(one_v, shift - 1);
            result = _mve_add_dw(result, add_v);

            result = _mve_shirs_dw(result, shift);

            __mdvdw min_v = _mve_set1_dw(-32768);
            __mdvdw max_v = _mve_set1_dw(32767);

            result = _mve_min_dw(result, max_v);
            result = _mve_max_dw(result, min_v);

            temp = _mve_cvt_dwtow(result);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                _mve_store_w(dst, temp, dst_stride);
            }
            _mve_set_all_elements(2);

            dst += LANE_NUM;
            src += LANE_NUM;

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

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 7);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 16, kvz_stride);
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 32, kvz_stride);
            __mdvdw kvz_4_v = _mve_cvts_wtodw(temp);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 48, kvz_stride);
            __mdvdw kvz_6_v = _mve_cvts_wtodw(temp);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            __mdvdw kvz_1_temp_v = _mve_cvts_wtodw(temp);
            kvz_0_v = _mve_assign_dw(kvz_0_v, kvz_1_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 17, kvz_stride);
            __mdvdw kvz_17_temp_v = _mve_cvts_wtodw(temp);
            kvz_2_v = _mve_assign_dw(kvz_2_v, kvz_17_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 33, kvz_stride);
            __mdvdw kvz_33_temp_v = _mve_cvts_wtodw(temp);
            kvz_4_v = _mve_assign_dw(kvz_4_v, kvz_33_temp_v);

            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 49, kvz_stride);
            __mdvdw kvz_49_temp_v = _mve_cvts_wtodw(temp);
            kvz_6_v = _mve_assign_dw(kvz_6_v, kvz_49_temp_v);

            __mdvw src_0_w_v = _mve_set1_w(0);
            __mdvw src_2_w_v = _mve_set1_w(0);
            __mdvw src_4_w_v = _mve_set1_w(0);
            __mdvw src_6_w_v = _mve_set1_w(0);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    src_0_w_v = _mve_assign_w(src_0_w_v, _mve_load_w(src + 0, src_stride));
                    src_2_w_v = _mve_assign_w(src_2_w_v, _mve_load_w(src + 16, src_stride));
                    src_4_w_v = _mve_assign_w(src_4_w_v, _mve_load_w(src + 32, src_stride));
                    src_6_w_v = _mve_assign_w(src_6_w_v, _mve_load_w(src + 48, src_stride));
                }
            }
            _mve_set_all_elements(2);
            _mve_set_all_elements(0);

            __mdvdw src_0_v = _mve_cvts_wtodw(src_0_w_v);
            __mdvdw mul_0_v = _mve_mul_dw(kvz_0_v, src_0_v);

            __mdvdw src_2_v = _mve_cvts_wtodw(src_2_w_v);
            __mdvdw mul_2_v = _mve_mul_dw(kvz_2_v, src_2_v);

            __mdvdw src_4_v = _mve_cvts_wtodw(src_4_w_v);
            __mdvdw mul_4_v = _mve_mul_dw(kvz_4_v, src_4_v);

            __mdvdw src_6_v = _mve_cvts_wtodw(src_6_w_v);
            __mdvdw mul_6_v = _mve_mul_dw(kvz_6_v, src_6_v);

            __mdvdw eo_v = _mve_add_dw(mul_2_v, mul_6_v);
            __mdvdw ee_v = _mve_add_dw(mul_0_v, mul_4_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            __mdvdw e_v = _mve_add_dw(ee_v, eo_v);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 5);

            __mdvdw ee_sub_eo_v = _mve_sub_dw(ee_v, eo_v);
            e_v = _mve_assign_dw(e_v, ee_sub_eo_v);

            // kvz_1_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 8, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 9, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 10, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 11, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp2);

            // kvz_3_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 24, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 25, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 26, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 27, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp2);

            // kvz_5_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 40, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 41, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 42, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 43, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_5_v = _mve_cvts_wtodw(temp2);

            // kvz_7_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 56, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 57, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 58, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 59, kvz_stride);
            temp2 = _mve_assign_w(temp2, temp);

            _mve_set_all_elements(2);
            __mdvdw kvz_7_v = _mve_cvts_wtodw(temp2);

            // Swap with [R4 and R5]

            __mdvw src_1_w_v = _mve_set1_w(0);
            __mdvw src_3_w_v = _mve_set1_w(0);
            __mdvw src_5_w_v = _mve_set1_w(0);
            __mdvw src_7_w_v = _mve_set1_w(0);

            for (int i = 0; i < 8; i++) {
                _mve_set_only_element(2, i);
                for (int j = 0; j < 8; j++) {
                    _mve_set_only_element(0, j);
                    src_1_w_v = _mve_assign_w(src_1_w_v, _mve_load_w(src + 8, src_stride));
                    src_3_w_v = _mve_assign_w(src_3_w_v, _mve_load_w(src + 24, src_stride));
                    src_5_w_v = _mve_assign_w(src_5_w_v, _mve_load_w(src + 40, src_stride));
                    src_7_w_v = _mve_assign_w(src_7_w_v, _mve_load_w(src + 56, src_stride));
                }
            }
            _mve_set_all_elements(2);
            _mve_set_all_elements(0);

            __mdvdw src_1_v = _mve_cvts_wtodw(src_1_w_v);
            __mdvdw mul1_v = _mve_mul_dw(kvz_1_v, src_1_v);

            __mdvdw src_3_v = _mve_cvts_wtodw(src_3_w_v);
            __mdvdw mul3_v = _mve_mul_dw(kvz_3_v, src_3_v);
            __mdvdw mul1_3_v = _mve_add_dw(mul1_v, mul3_v);

            __mdvdw src_5_v = _mve_cvts_wtodw(src_5_w_v);
            __mdvdw mul5_v = _mve_mul_dw(kvz_5_v, src_5_v);

            __mdvdw src_7_v = _mve_cvts_wtodw(src_7_w_v);
            __mdvdw mul7_v = _mve_mul_dw(kvz_7_v, src_7_v);
            __mdvdw mul5_7_v = _mve_add_dw(mul5_v, mul7_v);

            __mdvdw o_v = _mve_add_dw(mul1_3_v, mul5_7_v);

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 3);

            __mdvdw result = _mve_add_dw(e_v, o_v);

            _mve_set_only_element(2, 4);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            __mdvdw e_sub_o_v = _mve_sub_dw(e_v, o_v);

            result = _mve_assign_dw(result, e_sub_o_v);

            _mve_set_all_elements(2);

            __mdvdw one_v = _mve_set1_dw(1);
            __mdvdw add_v = _mve_shil_dw(one_v, shift - 1);
            result = _mve_add_dw(result, add_v);

            result = _mve_shirs_dw(result, shift);

            __mdvdw min_v = _mve_set1_dw(-32768);
            __mdvdw max_v = _mve_set1_dw(32767);

            result = _mve_min_dw(result, max_v);
            result = _mve_max_dw(result, min_v);

            temp = _mve_cvt_dwtow(result);

            for (int element_idx = 0; element_idx < 8; element_idx += 1) {
                _mve_set_only_element(2, element_idx);
                _mve_store_w(dst, temp, dst_stride);
            }
            _mve_set_all_elements(2);

            dst += LANE_NUM;
            src += LANE_NUM;

            mve_flusher();
        }
    }
}