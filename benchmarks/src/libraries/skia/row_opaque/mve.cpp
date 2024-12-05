#include "mve.hpp"
#include "mve_kernels.hpp"

#include "row_opaque.hpp"

void row_opaque_mve(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    row_opaque_config_t *row_opaque_config = (row_opaque_config_t *)config;
    row_opaque_input_t *row_opaque_input = (row_opaque_input_t *)input;
    row_opaque_output_t *row_opaque_output = (row_opaque_output_t *)output;

    int num_cols = row_opaque_config->num_cols;
    int num_rows = row_opaque_config->num_rows;

    uint32_t *color = row_opaque_input->color;
    uint32_t *opaque_dst = row_opaque_output->opaque_dst;
    uint16_t *src = row_opaque_input->src;
    uint32_t *dst = row_opaque_output->dst;

    // Dim0: a pixel (RGBA)
    // Dim1: different pixels within a row
    // Dim2: different rows
    _mve_set_dim_count(3);

    // R11
    __mdvw zero_w = _mve_set1_w(0);
    // R10
    __mdvw FFFF_w = _mve_set1_w(0xFFFF);
    // R9H
    __mdvb FF_b = _mve_set1_b(0xFF);

    // Reading sequentially in a pixel
    // Reading same value across pixels within a row
    // Reading next 4 values in the next row
    _mve_set_load_stride(2, 4);
    __vidx_var row_stride = {1, 0, 3, 0};

    // Loading same value for the bytes of a pixel
    // Loading sequential values for sequential pixels
    // Loading next row
    __vidx_var pixel_stride = {0, 1, 2, 0};

    // Reading and writing completely sequentially
    __vidx_var byte_stride = {2, 2, 2, 0};

    uint8_t maskbits_shifts[4] = {0, 11, 5, 0};
    uint16_t maskbits_vals[4] = {0xFFFF, 0xF800, 0x07E0, 0x001F};
    __vidx_var maskbits_stride = {1, 0, 0, 0};

    _mve_set_dim_length(0, 4);
    LANE_NUM /= 4;

    int DIM1_TILE = num_cols > LANE_NUM ? LANE_NUM : num_cols;
    int DIM2_TILE = LANE_NUM / DIM1_TILE;
    int pixel_per_row_chunk = DIM2_TILE * num_cols;

    int row = 0;
    _mve_set_dim_length(2, DIM2_TILE);
    while (row < num_rows) {
        int remaining_rows = num_rows - row;
        remaining_rows = remaining_rows > DIM2_TILE ? DIM2_TILE : remaining_rows;
        if (remaining_rows != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_rows);
        }

        uint16_t *my_src = src;
        uint32_t *my_dst = dst;

        int col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        // R9L
        __mdvb color_b = _mve_load_b((const unsigned char *)color, row_stride);
        // R8
        __mdvw color_w = _mve_cvtu_btow(color_b);
        // free color_b (R9L)
        _mve_free_b();
        // R9L
        __mdvb opaque_b = _mve_load_b((const unsigned char *)opaque_dst, row_stride);
        // R7
        __mdvw maskbits_vals_w = _mve_load_w((const __int16_t *)maskbits_vals, maskbits_stride);
        // R6H
        __mdvb maskbits_shifts_b = _mve_load_b((const unsigned char *)maskbits_shifts, maskbits_stride);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
                // R9L
                color_b = _mve_load_b((const unsigned char *)color, row_stride);
                // R8
                color_w = _mve_cvtu_btow(color_b);
                // free color_b (R9L) and color_w (R8)
                _mve_free_b();
                _mve_free_w();
                // R9L
                opaque_b = _mve_load_b((const unsigned char *)opaque_dst, row_stride);
                // free opaque_b (R9L)
                _mve_free_b();
                // R7
                maskbits_vals_w = _mve_load_w((const __int16_t *)maskbits_vals, maskbits_stride);
                // free maskbits_vals_w (R7)
                _mve_free_w();
                // R6H
                maskbits_shifts_b = _mve_load_b((const unsigned char *)maskbits_shifts, maskbits_stride);
                // free maskbits_shifts_b (R6H)
                _mve_free_b();
            }

            // [R0, R1, R2, R3, R4, R5, R6L]

            // R6L
            __mdvb dst_b = _mve_load_b((const unsigned char *)my_src, byte_stride);
            // R5
            __mdvw mask_w = _mve_load_w((const __int16_t *)my_dst, pixel_stride);

            _mve_set_only_element(0, 0);

            // [R0, R1, R2, R3, R4]

            /***************** Find dst_b for A ******************/

            _mve_cmpneq_w(mask_w, zero_w);
            // R4H
            dst_b = _mve_assign_b(dst_b, FF_b);
            // free dst_b (R6L)
            _mve_free_b();

            _mve_unset_only_element(0, 0);

            // [R0, R1, R2, R3, R4L, R6L]

            /***************** Find dst_b for RGB ******************/

            // R3
            __mdvw mask_ARGB_w = _mve_and_w(mask_w, maskbits_vals_w);
            // R2
            mask_ARGB_w = _mve_shrru_w(mask_ARGB_w, maskbits_shifts_b);
            // free mask_ARGB_w (R3)
            _mve_free_w();
            // R3
            __mdvw mask_ARGB_shr4_w = _mve_shiru_w(mask_ARGB_w, 4);
            // R1
            mask_ARGB_w = _mve_add_w(mask_ARGB_w, mask_ARGB_shr4_w);
            // free mask_ARGB_w (R2) and mask_ARGB_shr4_w (R3)
            _mve_free_w();
            _mve_free_w();
            // R3
            __mdvw dst_w = _mve_cvtu_btow(dst_b);
            // R2
            dst_w = _mve_sub_w(color_w, dst_w);
            // free dst_w (R3)
            _mve_free_w();
            // R3
            dst_w = _mve_mul_w(dst_w, mask_ARGB_w);
            // free mask_ARGB_w (R1) and dst_w (R2)
            _mve_free_w();
            _mve_free_w();
            // R2
            dst_w = _mve_shiru_w(dst_w, 5);
            // free dst_w (R3)
            _mve_free_w();
            // R6L
            __mdvb temp_dst_b = _mve_cvt_wtob(dst_w);
            // free dst_w (R2)
            _mve_free_w();
            // R4L
            dst_b = _mve_assign_b(dst_b, temp_dst_b);
            // free dst_b (R4H) and temp_dst_b (R6L)
            _mve_free_b();
            _mve_free_b();

            _mve_set_all_elements(0);

            // [R0, R1, R2, R3, R4H, R6L]

            /***************** dst = (mask == FFFF) ? opaque : dst ******************/

            _mve_cmpeq_w(mask_w, FFFF_w);
            // free mask_w (R5)
            _mve_free_w();
            // R6L
            dst_b = _mve_assign_b(dst_b, opaque_b);
            // free dst_b (R4L)
            _mve_free_b();

            _mve_set_all_elements(0);

            // [R0, R1, R2, R3, R4, R5]

            /***************** storing dst ******************/

            _mve_store_b((__uint8_t *)my_dst, dst_b, byte_stride);
            // free dst_b (R6L)
            _mve_free_b();

            // [R0, R1, R2, R3, R4, R5, R6L]

            col += DIM1_TILE;
            my_src += DIM1_TILE;
            my_dst += DIM1_TILE;
        }

        // free color_w (R8)
        _mve_free_w();
        // free opaque_b (R9L)
        _mve_free_b();
        // free maskbits_vals_w (R7)
        _mve_free_w();
        // free maskbits_shifts_b (R6H)
        _mve_free_b();

        row += DIM2_TILE;
        color += DIM2_TILE;
        opaque_dst += DIM2_TILE;
        src += pixel_per_row_chunk;
        dst += pixel_per_row_chunk;
    }

    // free zero_w (R11)
    _mve_free_w();
    // free FFFF_w (R10)
    _mve_free_w();
    // free FF_b (R9H)
    _mve_free_b();
}