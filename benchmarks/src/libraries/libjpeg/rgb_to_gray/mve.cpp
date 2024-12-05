
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

#include "rgb_to_gray.hpp"

void rgb_to_gray_mve(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    rgb_to_gray_config_t *rgb_to_gray_config = (rgb_to_gray_config_t *)config;
    rgb_to_gray_input_t *rgb_to_gray_input = (rgb_to_gray_input_t *)input;
    rgb_to_gray_output_t *rgb_to_gray_output = (rgb_to_gray_output_t *)output;
    // Dim0: columns
    // Dim1: rows
    _mve_set_dim_count(2);

    // Loading RGB- with stride of RGB_PIXELSIZE
    _mve_set_load_stride(0, RGB_PIXELSIZE);
    __vidx_var input_stride = {3, 0, 0, 0};
    uint8_t **input_addr;

    // Storing Y value sequentially
    __vidx_var output_stride = {1, 0, 0, 0};
    uint8_t **output_addr;

    // R0
    __mdvw R_const = _mve_set1_w((uint16_t)F_0_298);
    // R1
    __mdvw G_const = _mve_set1_w((uint16_t)F_0_587);
    // R2
    __mdvw B_const = _mve_set1_w((uint16_t)F_0_113);

    int DIM0_TILE = rgb_to_gray_config->num_cols > LANE_NUM ? LANE_NUM : rgb_to_gray_config->num_cols;
    int DIM1_TILE = LANE_NUM / DIM0_TILE;

    JDIMENSION row = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);
    while (row < rgb_to_gray_config->num_rows) {
        int remaining_rows = rgb_to_gray_config->num_rows - row;
        remaining_rows = remaining_rows > DIM1_TILE ? DIM1_TILE : remaining_rows;
        if (remaining_rows != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_rows);
        }

        input_addr = (unsigned char **)rgb_to_gray_input->input_buf + row;
        output_addr = (unsigned char **)rgb_to_gray_output->output_buf + row;

        JDIMENSION col = 0;
        _mve_set_dim_length(0, DIM0_TILE);
        while (col < rgb_to_gray_config->num_cols) {
            int remaining_cols = rgb_to_gray_config->num_cols - col;
            remaining_cols = remaining_cols > DIM0_TILE ? DIM0_TILE : remaining_cols;
            if (remaining_cols != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_cols);
            }

            // [R3 R4 R5]

            int offset = col << RGB_PIXELSIZE_LOG;

            // R3
            __mdvb R_b = _mve_loadro_b((const uint8_t **)input_addr, offset, input_stride);

            // R4
            __mdvw R_w = _mve_cvtu_btow(R_b);
            // free R_b (R3)
            _mve_free_b();

            // R3
            __mdvw R_mul_w = _mve_mul_w(R_w, R_const);
            // free R_w (R4)
            _mve_free_w();

            // R4
            __mdvdw acc_dw_1 = _mve_cvtu_wtodw(R_mul_w);
            // free R_mul_w (R3)
            _mve_free_w();

            // [R3 R5]

            // R3
            __mdvb G_b = _mve_loadro_b((const uint8_t **)input_addr, offset + 1, input_stride);

            // R5
            __mdvw G_w = _mve_cvtu_btow(G_b);
            // free G_b (R3)
            _mve_free_b();

            // R3
            __mdvw G_mul_w = _mve_mul_w(G_w, G_const);
            // free G_w (R5)
            _mve_free_w();

            // R5
            __mdvdw G_mul_dw = _mve_cvtu_wtodw(G_mul_w);
            // free G_mul_w (R3)
            _mve_free_w();

            // R3
            __mdvdw acc_dw_2 = _mve_add_dw(acc_dw_1, G_mul_dw);
            // free acc_dw_1 (R4) and G_mul_dw (R5)
            _mve_free_dw();
            _mve_free_dw();

            // [R4 R5]

            // R4
            __mdvb B_b = _mve_loadro_b((const uint8_t **)input_addr, offset + 2, input_stride);
            // R5
            __mdvw B_w = _mve_cvtu_btow(B_b);
            // free B_b (R4)
            _mve_free_b();

            // R4
            __mdvw B_mul_w = _mve_mul_w(B_w, B_const);
            // free B_w (R5)
            _mve_free_w();

            // R5
            __mdvdw B_mul_dw = _mve_cvtu_wtodw(B_mul_w);
            // free B_mul_w (R4)
            _mve_free_w();

            // R4
            __mdvdw acc_dw_3 = _mve_add_dw(acc_dw_2, B_mul_dw);
            // free acc_dw_2 (R3) and B_mul_dw (R5)
            _mve_free_dw();
            _mve_free_dw();

            // R3
            __mdvdw acc_dw = _mve_shiru_dw(acc_dw_3, 16);
            // free acc_dw_3 (R4)
            _mve_free_dw();

            // R4
            __mdvb acc_b = _mve_cvt_dwtob(acc_dw);
            // free acc_dw (R3)
            _mve_free_dw();

            _mve_storero_b(output_addr, col, acc_b, output_stride);
            // free acc_b (R4)
            _mve_free_b();

            // [R3 R4 R5]
            col += DIM0_TILE;
        }
        row += DIM1_TILE;
    }

    // free R_const (R0), G_const (R1), B_const (R2)
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
}