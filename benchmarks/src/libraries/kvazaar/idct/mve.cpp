#include "mve.hpp"
#include "cstdint"
#include "idct.hpp"
#include "kvazaar.hpp"
#include <cstdint>
#include <cstdio>

void idct_mve(int LANE_NUM,
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

            // [R0:5]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 7);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            // R1
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 16, kvz_stride);
            // R2
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 32, kvz_stride);
            // R3
            __mdvdw kvz_4_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 48, kvz_stride);
            // R4
            __mdvdw kvz_6_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            // [R0 R5]

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            // R5
            __mdvdw kvz_1_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            kvz_0_v = _mve_assign_dw(kvz_0_v, kvz_1_temp_v);
            // free kvz_0_v (R1) and kvz_1_temp_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R5]

            // R5-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 17, kvz_stride);
            // R1
            __mdvdw kvz_17_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R5-L)
            _mve_free_w();
            // R5
            kvz_2_v = _mve_assign_dw(kvz_2_v, kvz_17_temp_v);
            // free kvz_2_v (R2) and kvz_17_temp_v (R1)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 33, kvz_stride);
            // R2
            __mdvdw kvz_33_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1
            kvz_4_v = _mve_assign_dw(kvz_4_v, kvz_33_temp_v);
            // free kvz_4_v (R3) and kvz_33_temp_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            // R2-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 49, kvz_stride);
            // R3
            __mdvdw kvz_49_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R2-L)
            _mve_free_w();
            // R2
            kvz_6_v = _mve_assign_dw(kvz_6_v, kvz_49_temp_v);
            // free kvz_6_v (R4) and kvz_49_temp_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R3 R4]

            _mve_set_all_elements(2);

            // R3-L
            temp = _mve_load_w(src + 0, src_stride);
            // R4
            __mdvdw src_0_v = _mve_cvts_wtodw(temp);
            // free temp (R3-L)
            _mve_free_w();
            // R3
            __mdvdw mul_0_v = _mve_mul_dw(kvz_0_v, src_0_v);
            // free kvz_0_v (R0) and src_0_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4]

            // R0-L
            temp = _mve_load_w(src + 16, src_stride);
            // R4
            __mdvdw src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw mul_2_v = _mve_mul_dw(kvz_2_v, src_2_v);
            // free kvz_2_v (R5) and src_2_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R4 R5]

            // R4-L
            temp = _mve_load_w(src + 32, src_stride);
            // R5
            __mdvdw src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R4-L)
            _mve_free_w();
            // R4
            __mdvdw mul_4_v = _mve_mul_dw(kvz_4_v, src_4_v);
            // free kvz_4_v (R1) and src_4_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R5]

            // R1-L
            temp = _mve_load_w(src + 48, src_stride);
            // R5
            __mdvdw src_6_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1
            __mdvdw mul_6_v = _mve_mul_dw(kvz_6_v, src_6_v);
            // free kvz_6_v (R2) and src_6_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 and R5]

            // R2
            __mdvdw eo_v = _mve_add_dw(mul_2_v, mul_6_v);
            // free mul_2_v (R0) and mul_6_v (R1)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            __mdvdw ee_v = _mve_add_dw(mul_0_v, mul_4_v);
            // free mul_0_v (R3) and mul_4_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R3 R4 R5]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            // R0
            __mdvdw e_v = _mve_add_dw(ee_v, eo_v);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 5);

            // R3
            __mdvdw ee_sub_eo_v = _mve_sub_dw(ee_v, eo_v);
            // free ee_v (R1) and eo_v (R2)
            _mve_free_dw();
            _mve_free_dw();
            // R22
            e_v = _mve_assign_dw(e_v, ee_sub_eo_v);
            // free e_v (R0) and ee_sub_eo_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R0:5]
            // kvz_1_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 8, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 9, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 10, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 11, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R5
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:4]
            // kvz_3_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 24, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 25, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 26, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 27, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R4
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:3]
            // kvz_5_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 40, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 41, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 42, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 43, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R3
            __mdvdw kvz_5_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:2]
            // kvz_7_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 56, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 57, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 58, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 59, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R2
            __mdvdw kvz_7_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0 and R1]
            // Swap with [R4 and R5]
            // [R4 and R5]

            // R4-L
            temp = _mve_load_w(src + 8, src_stride);
            // R5
            __mdvdw src_1_v = _mve_cvts_wtodw(temp);
            // free temp (R4-L)
            _mve_free_w();
            // R4
            __mdvdw mul1_v = _mve_mul_dw(kvz_1_v, src_1_v);
            // free kvz_1_v (R0) and src_1_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R5]

            // R0-L
            temp = _mve_load_w(src + 24, src_stride);
            // R5
            __mdvdw src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw mul3_v = _mve_mul_dw(kvz_3_v, src_3_v);
            // free kvz_3_v (R1) and src_3_v (R5)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            __mdvdw mul1_3_v = _mve_add_dw(mul1_v, mul3_v);
            // free mul1_v (R4) and mul3_v (R0)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4 R5]

            // R0-L
            temp = _mve_load_w(src + 40, src_stride);
            // R4
            __mdvdw src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R5
            __mdvdw mul5_v = _mve_mul_dw(kvz_5_v, src_5_v);
            // free kvz_5_v (R2) and src_5_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R4]

            // R0-L
            temp = _mve_load_w(src + 56, src_stride);
            // R2
            __mdvdw src_7_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R4
            __mdvdw mul7_v = _mve_mul_dw(kvz_7_v, src_7_v);
            // free kvz_7_v (R3) and src_7_v (R2)
            _mve_free_dw();
            _mve_free_dw();
            // R2
            __mdvdw mul5_7_v = _mve_add_dw(mul5_v, mul7_v);
            // free mul5_v (R5) and mul7_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R3 R4 R5]

            // R5
            __mdvdw o_v = _mve_add_dw(mul1_3_v, mul5_7_v);
            // free mul1_3_v (R1) and mul5_7_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0:4]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 3);

            // R0
            __mdvdw result = _mve_add_dw(e_v, o_v);

            _mve_set_only_element(2, 4);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            // R3
            __mdvdw e_sub_o_v = _mve_sub_dw(e_v, o_v);
            // free e_v (R22) and o_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R4 R5]

            // R5
            result = _mve_assign_dw(result, e_sub_o_v);
            // free result (r0) and e_sub_o_v (r3)
            _mve_free_dw();
            _mve_free_dw();

            // [R0-4]

            _mve_set_all_elements(2);

            // R0
            __mdvdw one_v = _mve_set1_dw(1);
            // R1
            __mdvdw add_v = _mve_shil_dw(one_v, shift - 1);
            // free one_v (R0)
            _mve_free_dw();
            // R0
            result = _mve_add_dw(result, add_v);
            // free result (r5) and add_v (R1)
            _mve_free_dw();
            _mve_free_dw();

            // [R1-5]

            // R1
            result = _mve_shirs_dw(result, shift);
            // free result (R0)
            _mve_free_dw();

            // [R0 R2-5]

            // R2
            __mdvdw min_v = _mve_set1_dw(-32768);
            // R3
            __mdvdw max_v = _mve_set1_dw(32767);

            // [R0 R4 R5]

            // R0
            result = _mve_min_dw(result, max_v);
            // free result (R1) and max_v (R3)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            result = _mve_max_dw(result, min_v);
            // free result (R0) and min_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // R0-L
            temp = _mve_cvt_dwtow(result);
            // free result (R1)
            _mve_free_dw();

            _mve_store_w(dst, temp, dst_stride);
            // free temp (R0-L)
            _mve_free_w();

            // [R0-5]

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

            // [R0:5]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 7);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 0, kvz_stride);
            // R1
            __mdvdw kvz_0_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 16, kvz_stride);
            // R2
            __mdvdw kvz_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 32, kvz_stride);
            // R3
            __mdvdw kvz_4_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 48, kvz_stride);
            // R4
            __mdvdw kvz_6_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();

            // [R0 R5]

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);

            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 1, kvz_stride);
            // R5
            __mdvdw kvz_1_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            kvz_0_v = _mve_assign_dw(kvz_0_v, kvz_1_temp_v);
            // free kvz_0_v (R1) and kvz_1_temp_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R5]

            // R5-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 17, kvz_stride);
            // R1
            __mdvdw kvz_17_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R5-L)
            _mve_free_w();
            // R5
            kvz_2_v = _mve_assign_dw(kvz_2_v, kvz_17_temp_v);
            // free kvz_2_v (R2) and kvz_17_temp_v (R1)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2]

            // R1-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 33, kvz_stride);
            // R2
            __mdvdw kvz_33_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1
            kvz_4_v = _mve_assign_dw(kvz_4_v, kvz_33_temp_v);
            // free kvz_4_v (R3) and kvz_33_temp_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 R3]

            // R2-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 49, kvz_stride);
            // R3
            __mdvdw kvz_49_temp_v = _mve_cvts_wtodw(temp);
            // free temp (R2-L)
            _mve_free_w();
            // R2
            kvz_6_v = _mve_assign_dw(kvz_6_v, kvz_49_temp_v);
            // free kvz_6_v (R4) and kvz_49_temp_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R3 R4]

            _mve_set_all_elements(2);

            // R3-L
            temp = _mve_load_w(src + 0, src_stride);
            // R4
            __mdvdw src_0_v = _mve_cvts_wtodw(temp);
            // free temp (R3-L)
            _mve_free_w();
            // R3
            __mdvdw mul_0_v = _mve_mul_dw(kvz_0_v, src_0_v);
            // free kvz_0_v (R0) and src_0_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4]

            // R0-L
            temp = _mve_load_w(src + 16, src_stride);
            // R4
            __mdvdw src_2_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw mul_2_v = _mve_mul_dw(kvz_2_v, src_2_v);
            // free kvz_2_v (R5) and src_2_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R4 R5]

            // R4-L
            temp = _mve_load_w(src + 32, src_stride);
            // R5
            __mdvdw src_4_v = _mve_cvts_wtodw(temp);
            // free temp (R4-L)
            _mve_free_w();
            // R4
            __mdvdw mul_4_v = _mve_mul_dw(kvz_4_v, src_4_v);
            // free kvz_4_v (R1) and src_4_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R5]

            // R1-L
            temp = _mve_load_w(src + 48, src_stride);
            // R5
            __mdvdw src_6_v = _mve_cvts_wtodw(temp);
            // free temp (R1-L)
            _mve_free_w();
            // R1
            __mdvdw mul_6_v = _mve_mul_dw(kvz_6_v, src_6_v);
            // free kvz_6_v (R2) and src_6_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R2 and R5]

            // R2
            __mdvdw eo_v = _mve_add_dw(mul_2_v, mul_6_v);
            // free mul_2_v (R0) and mul_6_v (R1)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            __mdvdw ee_v = _mve_add_dw(mul_0_v, mul_4_v);
            // free mul_0_v (R3) and mul_4_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R3 R4 R5]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            // R0
            __mdvdw e_v = _mve_add_dw(ee_v, eo_v);

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 3);
            _mve_set_active_element(2, 4);
            _mve_set_active_element(2, 5);

            // R3
            __mdvdw ee_sub_eo_v = _mve_sub_dw(ee_v, eo_v);
            // free ee_v (R1) and eo_v (R2)
            _mve_free_dw();
            _mve_free_dw();
            // R22
            e_v = _mve_assign_dw(e_v, ee_sub_eo_v);
            // free e_v (R0) and ee_sub_eo_v (R3)
            _mve_free_dw();
            _mve_free_dw();

            // [R0:5]
            // kvz_1_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 8, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 9, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 10, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 11, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R5
            __mdvdw kvz_1_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:4]
            // kvz_3_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 24, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 25, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 26, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 27, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R4
            __mdvdw kvz_3_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:3]
            // kvz_5_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 40, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 41, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 42, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 43, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R3
            __mdvdw kvz_5_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0:2]
            // kvz_7_v starts from here

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 7);
            // R1-L
            temp2 = _mve_load_w(kvz_g_dct_8_s16_1D + 56, kvz_stride);

            _mve_set_only_element(2, 1);
            _mve_set_active_element(2, 6);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 57, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 2);
            _mve_set_active_element(2, 5);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 58, kvz_stride);
            // R1-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R2-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_only_element(2, 3);
            _mve_set_active_element(2, 4);
            // R0-L
            temp = _mve_load_w(kvz_g_dct_8_s16_1D + 59, kvz_stride);
            // R2-L
            temp2 = _mve_assign_w(temp2, temp);
            // free temp2 (R1-L) and temp (R0-L)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(2);
            // R2
            __mdvdw kvz_7_v = _mve_cvts_wtodw(temp2);
            // free temp2 (R2-L)
            _mve_free_w();

            // [R0 and R1]
            // Swap with [R4 and R5]
            // [R4 and R5]

            // R4-L
            temp = _mve_load_w(src + 8, src_stride);
            // R5
            __mdvdw src_1_v = _mve_cvts_wtodw(temp);
            // free temp (R4-L)
            _mve_free_w();
            // R4
            __mdvdw mul1_v = _mve_mul_dw(kvz_1_v, src_1_v);
            // free kvz_1_v (R0) and src_1_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R5]

            // R0-L
            temp = _mve_load_w(src + 24, src_stride);
            // R5
            __mdvdw src_3_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R0
            __mdvdw mul3_v = _mve_mul_dw(kvz_3_v, src_3_v);
            // free kvz_3_v (R1) and src_3_v (R5)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            __mdvdw mul1_3_v = _mve_add_dw(mul1_v, mul3_v);
            // free mul1_v (R4) and mul3_v (R0)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R4 R5]

            // R0-L
            temp = _mve_load_w(src + 40, src_stride);
            // R4
            __mdvdw src_5_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R5
            __mdvdw mul5_v = _mve_mul_dw(kvz_5_v, src_5_v);
            // free kvz_5_v (R2) and src_5_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R2 R4]

            // R0-L
            temp = _mve_load_w(src + 56, src_stride);
            // R2
            __mdvdw src_7_v = _mve_cvts_wtodw(temp);
            // free temp (R0-L)
            _mve_free_w();
            // R4
            __mdvdw mul7_v = _mve_mul_dw(kvz_7_v, src_7_v);
            // free kvz_7_v (R3) and src_7_v (R2)
            _mve_free_dw();
            _mve_free_dw();
            // R2
            __mdvdw mul5_7_v = _mve_add_dw(mul5_v, mul7_v);
            // free mul5_v (R5) and mul7_v (R4)
            _mve_free_dw();
            _mve_free_dw();

            // [R0 R3 R4 R5]

            // R5
            __mdvdw o_v = _mve_add_dw(mul1_3_v, mul5_7_v);
            // free mul1_3_v (R1) and mul5_7_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // [R0:4]

            _mve_set_only_element(2, 0);
            _mve_set_active_element(2, 1);
            _mve_set_active_element(2, 2);
            _mve_set_active_element(2, 3);

            // R0
            __mdvdw result = _mve_add_dw(e_v, o_v);

            _mve_set_only_element(2, 4);
            _mve_set_active_element(2, 5);
            _mve_set_active_element(2, 6);
            _mve_set_active_element(2, 7);

            // R3
            __mdvdw e_sub_o_v = _mve_sub_dw(e_v, o_v);
            // free e_v (R22) and o_v (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R1 R2 R4 R5]

            // R5
            result = _mve_assign_dw(result, e_sub_o_v);
            // free result (r0) and e_sub_o_v (r3)
            _mve_free_dw();
            _mve_free_dw();

            // [R0-4]

            _mve_set_all_elements(2);

            // R0
            __mdvdw one_v = _mve_set1_dw(1);
            // R1
            __mdvdw add_v = _mve_shil_dw(one_v, shift - 1);
            // free one_v (R0)
            _mve_free_dw();
            // R0
            result = _mve_add_dw(result, add_v);
            // free result (r5) and add_v (R1)
            _mve_free_dw();
            _mve_free_dw();

            // [R1-5]

            // R1
            result = _mve_shirs_dw(result, shift);
            // free result (R0)
            _mve_free_dw();

            // [R0 R2-5]

            // R2
            __mdvdw min_v = _mve_set1_dw(-32768);
            // R3
            __mdvdw max_v = _mve_set1_dw(32767);

            // [R0 R4 R5]

            // R0
            result = _mve_min_dw(result, max_v);
            // free result (R1) and max_v (R3)
            _mve_free_dw();
            _mve_free_dw();
            // R1
            result = _mve_max_dw(result, min_v);
            // free result (R0) and min_v (R2)
            _mve_free_dw();
            _mve_free_dw();

            // R0-L
            temp = _mve_cvt_dwtow(result);
            // free result (R1)
            _mve_free_dw();

            _mve_store_w(dst, temp, dst_stride);
            // free temp (R0-L)
            _mve_free_w();

            // [R0-5]

            dst += LANE_NUM;
            src += LANE_NUM;

            mve_flusher();
        }
    }
}