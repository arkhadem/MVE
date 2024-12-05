#include "mve.hpp"
#include "mve_kernels.hpp"

#include "read_sub.hpp"

void read_sub_mve(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    read_sub_config_t *read_sub_config = (read_sub_config_t *)config;
    read_sub_input_t *read_sub_input = (read_sub_input_t *)input;
    read_sub_output_t *read_sub_output = (read_sub_output_t *)output;

    // Dim0: a 4-byte group within 1 row
    // Dim1: 4-pixel groups within 1 row
    // Dim0: rows
    _mve_set_dim_count(3);

    // Loading every 4 adjacent pixels within a group
    // Loading the same group
    // Loading randomely across rows
    __vidx_var input_stride = {1, 0, 0, 0};
    uint8_t **input_addr;

    // Storing pixels sequentially
    __vidx_var output_stride = {2, 2, 0, 0};
    uint8_t **output_addr;

    int num_rows = read_sub_config->num_rows;
    int num_cols = read_sub_config->num_cols;

    // Because we have groups of 4 columns
    LANE_NUM /= 4;
    num_cols /= 4;
    _mve_set_dim_length(0, 4);

    uint8_t **input_buf = read_sub_input->input_buf;
    uint8_t **output_buf = read_sub_output->output_buf;

    int DIM2_TILE = num_rows > LANE_NUM ? LANE_NUM : num_rows;
    int DIM1_TILE = LANE_NUM / DIM2_TILE;

    int row = 0;
    _mve_set_dim_length(2, DIM2_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM2_TILE ? DIM2_TILE : remaining_rows;
        if (remaining_rows != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_rows);
        }

        // R0
        __mdvb val = _mve_set1_b(0);

        input_addr = (uint8_t **)input_buf + row;
        output_addr = (uint8_t **)output_buf + row;

        int col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
            }

            int input_offset = col << 2;
#pragma unroll
            for (int c = 0; c < remaining_cols; c++) {
                // R1
                __mdvb temp = _mve_loadro_b((const __uint8_t **)input_addr, input_offset, input_stride);
                // R2
                val = _mve_add_b(val, temp);
                // free val (R0 or R1) and temp (R1)
                _mve_free_b();
                _mve_free_b();
                _mve_unset_only_element(1, c);
                input_offset += 4;
            }

            _mve_set_all_elements(1);

            _mve_storero_b(output_addr, col << 2, val, output_stride);
            // free val (R2)
            _mve_free_b();

            // R0
            val = _mve_loadro_b((const __uint8_t **)output_addr, (col + remaining_cols - 1) << 2, input_stride);

            col += remaining_cols;
        }

        // free val (R0)
        _mve_free_b();

        row += remaining_rows;
    }
}