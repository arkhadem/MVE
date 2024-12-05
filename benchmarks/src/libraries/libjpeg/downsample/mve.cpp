#include "mve.hpp"
#include "mve_kernels.hpp"

#include "downsample.hpp"

/* Downsample pixel values of a single component.
 * This version handles the standard case of 2:1 horizontal and 2:1 vertical,
 * without smoothing.
 */

void downsample_mve(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    downsample_config_t *downsample_config = (downsample_config_t *)config;
    downsample_input_t *downsample_input = (downsample_input_t *)input;
    downsample_output_t *downsample_output = (downsample_output_t *)output;

    // Dim0: a 2-pixel group within 1 row
    // Dim1: 2-pixel groups within 1 row
    // Dim0: rows
    _mve_set_dim_count(3);

    // Loading every 2 adjacent pixels within a group
    // Loading every 4 adjacent pixels across groups
    // Loading randomely across rows
    _mve_set_load_stride(0, 2);
    _mve_set_load_stride(1, 4);
    __vidx_var input_stride = {3, 3, 0, 0};
    uint8_t *input1_addr[256];
    uint8_t *input2_addr[256];

    // Constant load
    const int16_t bias[2] = {1, 2};
    __vidx_var bias_stride = {1, 0, 0, 0};
    // R5H
    __mdvw bias_w = _mve_set1_w(0);

    // Storing pixels sequentially
    __vidx_var output_stride = {1, 0, 0, 0};
    uint8_t *output_addr[256];

    int num_rows = downsample_config->num_rows;
    int num_cols = downsample_config->num_cols;

    // Because we have groups of 2 columns with 1 or 2 bias
    LANE_NUM /= 2;
    _mve_set_dim_length(0, 2);

    int DIM1_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM2_TILE = LANE_NUM / DIM1_TILE;

    JDIMENSION row = 0;
    _mve_set_dim_length(2, DIM2_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM2_TILE ? DIM2_TILE : remaining_rows;
        if (remaining_rows != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_rows);
        }

#pragma unroll
        for (int r_row = 0; r_row < remaining_rows; r_row++) {
            input1_addr[r_row] = downsample_input->input_buf[2 * row];
            input2_addr[r_row] = downsample_input->input_buf[2 * row + 1];
            output_addr[r_row] = downsample_output->output_buf[row];
            row++;
        }

        JDIMENSION col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        // R5L
        bias_w = _mve_load_w(bias, bias_stride);
        // free bias_w (R5H)
        _mve_free_w();
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
                // R5H
                bias_w = _mve_load_w(bias, bias_stride);
                // free bias_w (R5L)
                _mve_free_w();
            }

            int offset = col << 1;

            // R0
            __mdvb in11_b = _mve_loadro_b((const uint8_t **)input1_addr, offset, input_stride);
            // R1
            __mdvw in11_w = _mve_cvtu_btow(in11_b);
            // free in11_b (R0)
            _mve_free_b();

            // R0
            __mdvb in12_b = _mve_loadro_b((const uint8_t **)input1_addr, offset + 1, input_stride);
            // R2
            __mdvw in12_w = _mve_cvtu_btow(in12_b);
            // free in12_b (R0)
            _mve_free_b();

            // R3
            __mdvw in1_w = _mve_add_w(in11_w, in12_w);
            // free in11_w (R1) and in12_w (R2)
            _mve_free_w();
            _mve_free_w();

            // R0
            __mdvb in21_b = _mve_loadro_b((const uint8_t **)input2_addr, offset, input_stride);
            // R1
            __mdvw in21_w = _mve_cvtu_btow(in21_b);
            // free in21_b (R0)
            _mve_free_b();

            // R0
            __mdvb in22_b = _mve_loadro_b((const uint8_t **)input2_addr, offset + 1, input_stride);
            // R2
            __mdvw in22_w = _mve_cvtu_btow(in22_b);
            // free in22_b (R0)
            _mve_free_b();

            // R4
            __mdvw in2_w = _mve_add_w(in21_w, in22_w);
            // free in21_w (R1) and in22_w (R2)
            _mve_free_w();
            _mve_free_w();

            // R0
            __mdvw in_w = _mve_add_w(in1_w, in2_w);
            // free in1_w (R3) and in2_w (R4)
            _mve_free_w();
            _mve_free_w();

            // R1
            __mdvw biased_in_w = _mve_add_w(bias_w, in_w);
            // free in_w (R0)
            _mve_free_w();

            // R2
            __mdvw output_w = _mve_shirs_w(biased_in_w, 2);
            // free biased_in_w (R1)
            _mve_free_w();

            // R3
            __mdvb output_b = _mve_cvt_wtob(output_w);
            // free output_w (R2)
            _mve_free_w();

            _mve_storero_b(output_addr, col, output_b, output_stride);
            // free output_b (R3)
            _mve_free_b();

            col += DIM1_TILE;
        }
    }

    // free bias_w (R5H)
    _mve_free_w();
}