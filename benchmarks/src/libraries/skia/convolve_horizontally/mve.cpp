#include "mve.hpp"
#include "mve_kernels.hpp"

#include "convolve_horizontally.hpp"

void convolve_horizontally_mve(int LANE_NUM,
                               config_t *config,
                               input_t *input,
                               output_t *output) {
    convolve_horizontally_config_t *convolve_horizontally_config = (convolve_horizontally_config_t *)config;
    convolve_horizontally_input_t *convolve_horizontally_input = (convolve_horizontally_input_t *)input;
    convolve_horizontally_output_t *convolve_horizontally_output = (convolve_horizontally_output_t *)output;

    // Loop over each pixel on this row in the output image.

    int num_cols = convolve_horizontally_config->num_cols;
    int num_rows = convolve_horizontally_config->num_rows;
    int filter_length = convolve_horizontally_config->filter_length;

    uint8_t *src_data = convolve_horizontally_input->src_data;
    int16_t *filter_values = convolve_horizontally_input->filter_values;
    uint8_t *out_row = convolve_horizontally_output->out_row;

    // Dim0: a pixel: group of 4 colors (RGBA)
    // Dim1: column pixels
    // Dim2: row pixels
    _mve_set_dim_count(3);

    __mdvw min_w = _mve_set1_w(0);
    __mdvw max_w = _mve_set1_w(255);

    // input is loaded and stored sequentially in a row
    // loaded with (col + coeff) * 4 size in the next row
    int colors_per_input_row = (num_cols + filter_length) << 2;
    _mve_set_load_stride(2, colors_per_input_row);
    // stored with col * 4 size in the next row
    int colors_per_output_row = num_cols << 2;
    _mve_set_store_stride(2, colors_per_output_row);
    __vidx_var src_dst_stride = {2, 2, 3, 0};

    // Same coefficient for all cells
    __vidx_var coeff_stride = {0, 0, 0, 0};

    // a pixel: group of 4 colors
    _mve_set_dim_length(0, 4);

    LANE_NUM >>= 2;

    int DIM1_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM2_TILE = LANE_NUM / DIM1_TILE;

    int src_blk = (DIM2_TILE * (num_cols + filter_length)) << 2;
    int dst_blk = (DIM2_TILE * num_cols) << 2;

    int row = 0;
    _mve_set_dim_length(2, DIM2_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM2_TILE ? DIM2_TILE : remaining_rows;
        if (remaining_rows != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_rows);
        }

        // start address of current row
        uint8_t *src_addr = src_data;
        uint8_t *dst_addr = out_row;

        int col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
            }

            int16_t *coeff_addr = filter_values;
            uint8_t *my_src_addr = src_addr;

            __mdvdw acc_dw = _mve_set1_dw(0);

            for (int coeff_idx = 0; coeff_idx < filter_length; coeff_idx++) {
                __mdvb src_b = _mve_load_b(my_src_addr, src_dst_stride);
                __mdvdw src_dw = _mve_cvtu_btodw(src_b);
                // free src_b
                _mve_free_b();

                __mdvw coeff_w = _mve_load_w(coeff_addr, coeff_stride);
                __mdvdw coeff_dw = _mve_cvts_wtodw(coeff_w);
                // free coeff_w
                _mve_free_w();

                __mdvdw mult_dw = _mve_mul_dw(src_dw, coeff_dw);
                // free src_dw and coeff_dw (r4)
                _mve_free_dw();
                _mve_free_dw();

                acc_dw = _mve_add_dw(acc_dw, mult_dw);
                // free mult_dw and acc_dw (r2 or r3)
                _mve_free_dw();
                _mve_free_dw();

                my_src_addr += 4;
                coeff_addr += 1;
            }

            __mdvdw shifted_dw = _mve_shirs_dw(acc_dw, 2);
            // free acc_dw
            _mve_free_dw();

            __mdvw shifted_w = _mve_cvt_dwtow(shifted_dw);
            // free shifted_dw
            _mve_free_dw();

            __mdvw acc_min_w = _mve_min_w(shifted_w, max_w);
            // free shifted_w
            _mve_free_w();

            __mdvw acc_max_min_w = _mve_max_w(acc_min_w, min_w);
            // free acc_min_w
            _mve_free_w();

            __mdvb acc_b = _mve_cvt_wtob(acc_max_min_w);
            // free acc_max_min_w
            _mve_free_w();

            _mve_store_b(dst_addr, acc_b, src_dst_stride);
            // free acc_b
            _mve_free_b();

            // processed DIM1_TILE pixels (*4 colors)
            src_addr += DIM1_TILE << 2;
            dst_addr += DIM1_TILE << 2;
            col += DIM1_TILE;
        }

        // processed DIM2_TILE rows * (num_cols + filter_length) pixels * 4 colors
        src_data += src_blk;
        // processed DIM2_TILE rows * num_cols pixels * 4 colors
        out_row += dst_blk;
        row += DIM2_TILE;
    }

    // free min_w and max_w
    _mve_free_w();
    _mve_free_w();
}