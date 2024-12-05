#include "mve.hpp"
#include "mve_kernels.hpp"
#include <cstdio>

#include "sharp_filter_row.hpp"

void sharp_filter_row_mve(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    sharp_filter_row_config_t *sharp_filter_row_config = (sharp_filter_row_config_t *)config;
    sharp_filter_row_input_t *sharp_filter_row_input = (sharp_filter_row_input_t *)input;
    sharp_filter_row_output_t *sharp_filter_row_output = (sharp_filter_row_output_t *)output;

    const int max_y = (1 << sharp_filter_row_config->bit_depth) - 1;

    // Dim0: group of 2 pixels within a row
    // Dim1: different groups within a row
    // Dim2: different rows
    _mve_set_dim_count(3);

    int num_rows = sharp_filter_row_config->num_rows;
    int num_cols = sharp_filter_row_config->num_cols;

    // both pixels in the same group load the same A or B
    // next group loads from the next pixel
    // next row is num_cols + 1 pixels apart (within the code)
    int AB_stride_val = num_cols + 1;
    // _mve_set_load_stride(2, AB_stride_val);
    __vidx_var AB_stride = {0, 1, 3, 0};

    // Loading from consecuetive cells in a row (DIM0 and DIM1)
    // next row is 2 * num_cols pixels apart (within the code)
    int best_y_stride_val = 2 * num_cols;
    // _mve_set_load_stride(2, best_y_stride_val);
    __vidx_var best_y_stride = {2, 2, 3, 0};

    // Storing in the adjacent cells (in a row)
    // next row is 2 * num_cols pixels apart
    int output_stride_val = 2 * num_cols;
    _mve_set_store_stride(2, output_stride_val);
    __vidx_var output_stride = {2, 2, 0, 0};

    _mve_set_dim_length(0, 2);
    LANE_NUM /= 2;

    int DIM1_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM2_TILE = LANE_NUM / DIM1_TILE;

    int AB_increase = DIM2_TILE * (num_cols + 1);
    int bout_increase = DIM2_TILE * (num_cols * 2);

    int16_t *A = sharp_filter_row_input->A;
    int16_t *B = sharp_filter_row_input->B;
    uint16_t *best_y = sharp_filter_row_input->best_y;
    uint16_t *out = sharp_filter_row_output->out;

    int row = 0;
    _mve_set_dim_length(1, DIM1_TILE);
    _mve_set_dim_length(2, DIM2_TILE);

    // R11
    __mdvw min_w = _mve_set1_w(0);
    // R10
    __mdvw one_w = _mve_set1_w(1);
    // R9
    __mdvw max_w = _mve_set1_w(max_y);

    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM2_TILE ? DIM2_TILE : remaining_rows;
        if (remaining_rows != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_rows);
        }

        int16_t *A_addr = A;
        int16_t *B_addr = B;
        uint16_t *best_y_addr = best_y;
        uint16_t *out_addr = out;

        int col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
            }

            _mve_set_load_stride(2, AB_stride_val);

            // R0
            __mdvw A_first_w = _mve_load_w(A_addr, AB_stride);
            // R1
            __mdvw A_first_copy_w = _mve_copy_w(A_first_w);
            // R2
            __mdvw A_second_w = _mve_load_w(A_addr + 1, AB_stride);

            // R3
            __mdvw B_first_w = _mve_load_w(B_addr, AB_stride);
            // R4
            __mdvw B_first_copy_w = _mve_copy_w(B_first_w);
            // R5
            __mdvw B_second_w = _mve_load_w(B_addr + 1, AB_stride);

            _mve_set_only_element(0, 1);

            // R6
            A_first_w = _mve_assign_w(A_first_w, A_second_w);
            // free A_first_w (R0)
            _mve_free_w();

            // R7
            A_second_w = _mve_assign_w(A_second_w, A_first_copy_w);
            // free A_second_w (R2) and A_first_copy_w (R1)
            _mve_free_w();
            _mve_free_w();

            // R8
            B_first_w = _mve_assign_w(B_first_w, B_second_w);
            // free B_first_w (R3)
            _mve_free_w();

            // R0
            B_second_w = _mve_assign_w(B_second_w, B_first_copy_w);
            // free B_second_w (R5) and B_first_copy_w (R4)
            _mve_free_w();
            _mve_free_w();

            _mve_set_all_elements(0);

            // R1 = A1 + B0
            __mdvw A_second_B_first_w = _mve_add_w(A_second_w, B_first_w);
            // free A_second_w (R7) and B_first_w (R8)
            _mve_free_w();
            _mve_free_w();

            // R2 = B1 + A0
            __mdvw B_second_A_first_w = _mve_add_w(B_second_w, A_first_w);
            // free B_second_w (R0)
            _mve_free_w();

            // R3 = A0 + A1 + B0 + B1
            __mdvw A_B_sum_w = _mve_add_w(A_second_B_first_w, B_second_A_first_w);
            // free B_second_A_first_w (R2)
            _mve_free_w();

            // R4 = 2 * (A1 + B0)
            __mdvw A_second_B_first_2_w = _mve_shil_w(A_second_B_first_w, 1);
            // free A_second_B_first_w (R1)
            _mve_free_w();

            // R5 = A0 + A1 * 3 + B0 * 3 + B1
            __mdvw AB_all_w = _mve_add_w(A_B_sum_w, A_second_B_first_2_w);
            // free A_B_sum_w (R3) and A_second_B_first_2_w (R4)
            _mve_free_w();
            _mve_free_w();

            // R7 = (A0 + A1 * 3 + B0 * 3 + B1) >> 3
            __mdvw AB_all_shifted_w = _mve_shirs_w(AB_all_w, 1);
            // free AB_all_w (R5)
            _mve_free_w();

            // R8 = A0 + 1
            __mdvw A_first_1_t = _mve_add_w(A_first_w, one_w);
            // free A_first_w (R6)
            _mve_free_w();

            // R0 = (A0 * 9 + A1 * 3 + B0 * 3 + B1 + 8) >> 3
            __mdvw all_w = _mve_add_w(AB_all_shifted_w, A_first_1_t);
            // free AB_all_shifted_w (R7) and A_first_1_t (R8)
            _mve_free_w();
            _mve_free_w();

            // R1 = (A0 * 9 + A1 * 3 + B0 * 3 + B1 + 8) >> 4
            __mdvw result_w = _mve_shirs_w(all_w, 1);
            // free all_w (R0)
            _mve_free_w();

            _mve_set_load_stride(2, best_y_stride_val);

            // R2
            __mdvw best_y_w = _mve_load_w((const __int16_t *)best_y_addr, best_y_stride);

            // R3
            __mdvw result_best_y_w = _mve_add_w(result_w, best_y_w);
            // free result_w (R1) and best_y_w (R2)
            _mve_free_w();
            _mve_free_w();

            // R4
            __mdvw result_min_w = _mve_min_w(result_best_y_w, max_w);
            // free result_best_y_w (R3)
            _mve_free_w();

            // R5
            __mdvw result_min_max_w = _mve_min_w(result_min_w, min_w);
            // free result_min_w (R4)
            _mve_free_w();

            _mve_store_w((__int16_t *)out_addr, result_min_max_w, output_stride);
            // free result_min_max_w (R5)
            _mve_free_w();

            col += remaining_cols;
            A_addr += DIM1_TILE;
            B_addr += DIM1_TILE;
            best_y_addr += DIM1_TILE << 1;
            out_addr += DIM1_TILE << 1;
        }

        row += remaining_rows;
        A += AB_increase;
        B += AB_increase;
        best_y += bout_increase;
        out += bout_increase;
    }

    // free min_w (R11)
    _mve_free_w();
    // free one_w (R10)
    _mve_free_w();
    // free max_w (R9)
    _mve_free_w();
}