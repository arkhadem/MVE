#include "mve.hpp"
#include "mve_kernels.hpp"

#include "vertical_filter.hpp"

void vertical_filter_mve(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    vertical_filter_config_t *vertical_filter_config = (vertical_filter_config_t *)config;
    vertical_filter_input_t *vertical_filter_input = (vertical_filter_input_t *)input;
    vertical_filter_output_t *vertical_filter_output = (vertical_filter_output_t *)output;

    int stride = vertical_filter_config->stride;
    uint8_t *out = vertical_filter_output->out;
    uint8_t *preds = vertical_filter_input->in;
    uint8_t *in = preds + stride;
    int num_rows = vertical_filter_config->num_rows;
    int num_cols = vertical_filter_config->num_cols;

    // Dim0: cols
    // Dim0: rows
    _mve_set_dim_count(2);

    _mve_set_load_stride(1, stride);
    __vidx_var input_stride = {1, 3, 0, 0};

    _mve_set_store_stride(1, stride);
    __vidx_var output_stride = {1, 3, 0, 0};

    int DIM0_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM1_TILE = LANE_NUM / DIM0_TILE;

    int remaining_stride = DIM1_TILE * stride;

    int row = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM1_TILE ? DIM1_TILE : remaining_rows;
        if (remaining_rows != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_rows);
        }

        uint8_t *in_addr = in;
        uint8_t *out_addr = out;
        uint8_t *preds_addr = preds;

        int col = 0;
        _mve_set_dim_length(0, DIM0_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM0_TILE ? DIM0_TILE : remaining_cols;
            if (remaining_cols != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_cols);
            }

            // R0
            __mdvb curr_b = _mve_load_b(in_addr, input_stride);
            // R1
            __mdvb prev_b = _mve_load_b(preds_addr, input_stride);

            // R2
            __mdvb sub_b = _mve_add_b(curr_b, prev_b);
            _mve_free_b();
            _mve_free_b();

            _mve_store_b(out_addr, sub_b, output_stride);
            _mve_free_b();

            col += DIM0_TILE;
            in_addr += DIM0_TILE;
            out_addr += DIM0_TILE;
            preds_addr += DIM0_TILE;
        }

        row += DIM1_TILE;
        in += remaining_stride;
        out += remaining_stride;
        preds += remaining_stride;
    }
}