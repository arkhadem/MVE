#include "mve.hpp"
#include "mve_kernels.hpp"

#include "read_up.hpp"

void read_up_mve(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {
    read_up_config_t *read_up_config = (read_up_config_t *)config;
    read_up_input_t *read_up_input = (read_up_input_t *)input;
    read_up_output_t *read_up_output = (read_up_output_t *)output;

    // Dim0: cols
    // Dim0: rows
    _mve_set_dim_count(2);

    __vidx_var input_stride = {1, 0, 0, 0};
    uint8_t **input_addr;
    uint8_t **prev_input_addr;

    __vidx_var output_stride = {1, 0, 0, 0};
    uint8_t **output_addr;

    int num_rows = read_up_config->num_rows;
    int num_cols = read_up_config->num_cols;

    uint8_t **input_buf = read_up_input->input_buf;
    uint8_t **prev_input_buf = read_up_input->prev_input_buf;
    uint8_t **output_buf = read_up_output->output_buf;

    int DIM0_TILE = read_up_config->num_cols > LANE_NUM ? LANE_NUM : read_up_config->num_cols;
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

        input_addr = (uint8_t **)input_buf + row;
        prev_input_addr = (uint8_t **)prev_input_buf + row;
        output_addr = (uint8_t **)output_buf + row;

        int col = 0;
        _mve_set_dim_length(0, DIM0_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM0_TILE ? DIM0_TILE : remaining_cols;
            if (remaining_cols != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_cols);
            }

            // R0
            __mdvb curr_b = _mve_loadro_b((const __uint8_t **)input_addr, col, input_stride);
            // R1
            __mdvb prev_b = _mve_loadro_b((const __uint8_t **)prev_input_addr, col, input_stride);

            // R2
            __mdvb add_b = _mve_add_b(curr_b, prev_b);
            _mve_free_b();
            _mve_free_b();

            _mve_storero_b(output_addr, col, add_b, output_stride);
            _mve_free_b();

            col += remaining_cols;
        }

        row += remaining_rows;
    }
}