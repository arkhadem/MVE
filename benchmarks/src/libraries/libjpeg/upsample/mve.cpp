
/* RGB -> Grayscale conversion is defined by the following equation:
 *    Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *
 * Avoid floating point arithmetic by using shifted integer constants:
 *    0.29899597 = 19595 * 2^-16
 *    0.58700561 = 38470 * 2^-16
 *    0.11399841 =  7471 * 2^-16
 * These constants are defined in jcgray-neon.c
 *
 * This is the same computation as the RGB -> Y portion of RGB -> YCbCr.
 */

#include "mve.hpp"
#include "mve_kernels.hpp"
#include <cstdio>

#include "libjpeg.hpp"
#include "upsample.hpp"

void upsample_mve(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    upsample_config_t *upsample_config = (upsample_config_t *)config;
    upsample_input_t *upsample_input = (upsample_input_t *)input;
    upsample_output_t *upsample_output = (upsample_output_t *)output;

    // Dim0: group of 2 subsequent pixels in a row
    // Dim1: different groups in a row
    // Dim2: columns
    _mve_set_dim_count(3);

    // Loading Y: loading sequentially within a row(2, 2), random across columns (0)
    __vidx_var Y_load_stride = {2, 2, 0, 0};
    // Loading CB and CR:   loading same element within a group (0)
    //                      next element in the next group (1)
    //                      random across columns (0)
    __vidx_var CB_CR_load_stride = {0, 1, 0, 0};

    uint8_t **Y_input_addr;
    uint8_t *CB_input_addr[256];
    uint8_t *CR_input_addr[256];

    // Storing RGB- with RGB_PIXELSIZE stride
    // Next group stores in the next 2 RGB_PIXELSIZE pixels
    // Columns store in different columns randomely
    _mve_set_store_stride(0, RGB_PIXELSIZE);
    _mve_set_store_stride(1, 2 * RGB_PIXELSIZE);
    __vidx_var output_stride = {3, 3, 0, 0};

    uint8_t **output_addr;

    // R5L
    __mdvw F_1_402_const = _mve_set1_w((uint16_t)F_1_402);
    // R5H
    __mdvw F_0_714_const = _mve_set1_w((uint16_t)F_0_714);
    // R4H
    __mdvw F_128_const = _mve_set1_w((uint16_t)128);

    int num_rows = upsample_config->num_rows;
    int num_cols = upsample_config->num_cols;

    // First dimension is a group of 2 output pixels
    _mve_set_dim_length(0, 2);
    LANE_NUM /= 2;
    // Because the second and third dimensions fall on each other
    num_rows *= 2;

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

        Y_input_addr = (unsigned char **)upsample_input->input_buf[0] + row;
        output_addr = (unsigned char **)upsample_output->output_buf + row;

#pragma unroll
        for (int r_row = 0; r_row < remaining_rows; r_row++) {
            CB_input_addr[r_row] = upsample_input->input_buf[1][row / 2];
            CR_input_addr[r_row] = upsample_input->input_buf[2][row / 2];
            row++;
        }

        JDIMENSION col = 0;
        _mve_set_dim_length(1, DIM1_TILE);
        while (col < num_cols) {
            int remaining_cols = num_cols - col;
            remaining_cols = remaining_cols > DIM1_TILE ? DIM1_TILE : remaining_cols;
            if (remaining_cols != DIM1_TILE) {
                _mve_set_dim_length(1, remaining_cols);
            }

            int Y_offset = col << 1;
            int RGB_offset = Y_offset << RGB_PIXELSIZE_LOG;

            // [R0 R1 R2 R3 R4L]

            // R1
            __mdvb Y_b = _mve_loadro_b((const uint8_t **)Y_input_addr, Y_offset, Y_load_stride);

            // R0
            __mdvdw Y_dw = _mve_cvtu_btodw(Y_b);
            // free Y_b (R1)
            _mve_free_b();

            // [R1 R2 R3 R4L]

            // R1
            __mdvb CR_b = _mve_loadro_b((const uint8_t **)CR_input_addr, col, CB_CR_load_stride);

            // R3L
            __mdvw CR_w = _mve_cvtu_btow(CR_b);
            // free CR_b (R1)
            _mve_free_b();

            // R1L
            __mdvw CR_128_w = _mve_sub_w(CR_w, F_128_const);
            // free CR_w (R3L)
            _mve_free_w();

            // [R1H R2 R3 R4L]

            // R1H
            __mdvw F_1_402_CR_128_w = _mve_mul_w(F_1_402_const, CR_128_w);

            // R2
            __mdvdw F_1_402_CR_128_dw = _mve_cvtu_wtodw(F_1_402_CR_128_w);
            // free F_1_402_CR_128_w (R1H)
            _mve_free_w();

            // R2 (It's R3 indeed!)
            __mdvdw F_1_402_CR_128_shifted_dw = _mve_shirs_dw(F_1_402_CR_128_dw, 15);
            // free F_1_402_CR_128_dw (R2)
            _mve_free_dw();

            // R3
            __mdvdw R_dw = _mve_add_dw(Y_dw, F_1_402_CR_128_shifted_dw);
            // free F_1_402_CR_128_shifted_dw (R2)
            _mve_free_dw();

            // R2
            __mdvb R_b = _mve_cvt_dwtob(R_dw);
            // free R_dw (R3)
            _mve_free_dw();

            _mve_storero_b(output_addr, RGB_offset, R_b, output_stride);
            // free R_b (R2)
            _mve_free_b();

            // [R1H R2 R3 R4L]

            // R1H
            __mdvw F_0_714_CR_128_w = _mve_mul_w(F_0_714_const, CR_128_w);
            // free CR_128_w (R1L)
            _mve_free_w();

            // R3
            __mdvdw F_0_714_CR_128_dw = _mve_cvtu_wtodw(F_0_714_CR_128_w);
            // free F_0_714_CR_128_w (R1H)
            _mve_free_w();

            // [R1 R2 R4L]

            // R2
            __mdvb CB_b = _mve_loadro_b((const uint8_t **)CB_input_addr, col, CB_CR_load_stride);

            // R1
            __mdvw CB_w = _mve_cvtu_btow(CB_b);
            // free CB_b (R2)
            _mve_free_b();

            // R4L
            __mdvw CB_128_w = _mve_sub_w(CB_w, F_128_const);
            // free CB_w (R1)
            _mve_free_w();

            // [R1 R2]

            // R1L
            __mdvw F_0_344_const = _mve_set1_w((uint16_t)F_0_344);

            // R2
            __mdvw F_0_344_CB_128_w = _mve_mul_w(F_0_344_const, CB_128_w);
            // free F_0_344_const (R1L)
            _mve_free_w();

            // R1
            __mdvdw F_0_344_CB_128_dw = _mve_cvtu_wtodw(F_0_344_CB_128_w);
            // free F_0_344_CB_128_w (R2)
            _mve_free_w();

            // R2
            __mdvdw G1_dw = _mve_add_dw(F_0_344_CB_128_dw, F_0_714_CR_128_dw);
            // free F_0_344_CB_128_dw (R1) and F_0_714_CR_128_dw (R3)
            _mve_free_dw();
            _mve_free_dw();

            // R1
            __mdvdw G1_shiftd_dw = _mve_shirs_dw(G1_dw, 14);
            // free G1_dw (R2)
            _mve_free_dw();

            // R3
            __mdvdw G2_dw = _mve_sub_dw(Y_dw, G1_shiftd_dw);
            // free G1_shiftd_dw (R1)
            _mve_free_dw();

            // R2
            __mdvb G_b = _mve_cvt_dwtob(G2_dw);
            // free G2_dw (R3)
            _mve_free_dw();

            _mve_storero_b(output_addr, RGB_offset + 1, G_b, output_stride);
            // free G_b (R2)
            _mve_free_b();

            // [R1 R2 R3]

            // R1L
            __mdvw F_1_772_const = _mve_set1_w((uint16_t)F_1_772);

            // R2
            __mdvw F_1_772_CB_128_w = _mve_mul_w(F_1_772_const, CB_128_w);
            // free F_1_772_const (R1L) and CB_128_w (R4L)
            _mve_free_w();
            _mve_free_w();

            // R1
            __mdvdw F_1_772_CB_128_dw = _mve_cvtu_wtodw(F_1_772_CB_128_w);
            // free F_1_772_CB_128_w (R2)
            _mve_free_w();

            // R3
            __mdvdw F_1_772_CB_128_shifted_dw = _mve_shirs_dw(F_1_772_CB_128_dw, 15);
            // free F_1_772_CB_128_dw (R1)
            _mve_free_dw();

            // R2
            __mdvdw B_dw = _mve_add_dw(Y_dw, F_1_772_CB_128_shifted_dw);
            // free F_1_772_CB_128_shifted_dw (R3) and Y_dw (R0)
            _mve_free_dw();
            _mve_free_dw();

            // R0L
            __mdvb B_b = _mve_cvt_dwtob(B_dw);
            // free B_dw (R2)
            _mve_free_dw();

            _mve_storero_b(output_addr, RGB_offset + 2, B_b, output_stride);
            // free B_b (R0L)
            _mve_free_b();

            // [R0 R1 R2 R3 R4L]
            col += DIM1_TILE;
        }
    }

    // free F_1_402_const (R5L), F_0_714_const (R5H), F_128_const (R4H)
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
}