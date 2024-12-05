/* YCbCr -> RGB conversion is defined by the following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.3441467 = 11277 * 2^-15
 *    0.7141418 = 23401 * 2^-15
 *    1.4020386 = 22971 * 2^-14
 *    1.7720337 = 29033 * 2^-14
 * These constants are defined in jdcolor-neon.c.
 *
 * To ensure correct results, rounding is used when descaling.
 */

#include "mve.hpp"
#include "mve_kernels.hpp"

#include "libjpeg.hpp"
#include "ycbcr_to_rgb.hpp"

void ycbcr_to_rgb_mve(int LANE_NUM,
                      config_t *config,
                      input_t *input,
                      output_t *output) {
    ycbcr_to_rgb_config_t *ycbcr_to_rgb_config = (ycbcr_to_rgb_config_t *)config;
    ycbcr_to_rgb_input_t *ycbcr_to_rgb_input = (ycbcr_to_rgb_input_t *)input;
    ycbcr_to_rgb_output_t *ycbcr_to_rgb_output = (ycbcr_to_rgb_output_t *)output;
    // Dim0: samples
    _mve_set_dim_count(2);

    // Loading Y, CB, and CR from corresponding arrays sequentially
    __vidx_var input_stride = {1, 0, 0, 0};
    uint8_t **Y_input_addr;
    uint8_t **CB_input_addr;
    uint8_t **CR_input_addr;

    // Storing RGB- with RGB_PIXELSIZE stride
    _mve_set_store_stride(0, RGB_PIXELSIZE);
    __vidx_var output_stride = {3, 0, 0, 0};
    uint8_t **output_addr;

    // R5L
    __mdvw F_1_402_const = _mve_set1_w((uint16_t)F_1_402);
    // R5H
    __mdvw F_0_714_const = _mve_set1_w((uint16_t)F_0_714);
    // R4H
    __mdvw F_128_const = _mve_set1_w((uint16_t)128);

    int DIM0_TILE = ycbcr_to_rgb_config->num_cols > LANE_NUM ? LANE_NUM : ycbcr_to_rgb_config->num_cols;
    int DIM1_TILE = LANE_NUM / DIM0_TILE;

    JDIMENSION row = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);
    while (row < ycbcr_to_rgb_config->num_rows) {
        int remaining_rows = ycbcr_to_rgb_config->num_rows - row;
        remaining_rows = remaining_rows > DIM1_TILE ? DIM1_TILE : remaining_rows;
        if (remaining_rows != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_rows);
        }

        Y_input_addr = (unsigned char **)ycbcr_to_rgb_input->input_buf[0] + row;
        CB_input_addr = (unsigned char **)ycbcr_to_rgb_input->input_buf[1] + row;
        CR_input_addr = (unsigned char **)ycbcr_to_rgb_input->input_buf[2] + row;
        output_addr = (unsigned char **)ycbcr_to_rgb_output->output_buf + row;

        JDIMENSION col = 0;
        _mve_set_dim_length(0, DIM0_TILE);
        while (col < ycbcr_to_rgb_config->num_cols) {
            int remaining_cols = ycbcr_to_rgb_config->num_cols - col;
            remaining_cols = remaining_cols > DIM0_TILE ? DIM0_TILE : remaining_cols;
            if (remaining_cols != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_cols);
            }

            int offset = col << RGB_PIXELSIZE_LOG;

            // [R0 R1 R2 R3 R4L]

            // R1
            __mdvb Y_b = _mve_loadro_b((const uint8_t **)Y_input_addr, col, input_stride);

            // R0
            __mdvdw Y_dw = _mve_cvtu_btodw(Y_b);
            // free Y_b (R1)
            _mve_free_b();

            // [R1 R2 R3 R4L]

            // R1
            __mdvb CR_b = _mve_loadro_b((const uint8_t **)CR_input_addr, col, input_stride);

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

            _mve_storero_b(output_addr, offset, R_b, output_stride);
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
            __mdvb CB_b = _mve_loadro_b((const uint8_t **)CB_input_addr, col, input_stride);

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

            _mve_storero_b(output_addr, offset + 1, G_b, output_stride);
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

            _mve_storero_b(output_addr, offset + 2, B_b, output_stride);
            // free B_b (R0L)
            _mve_free_b();

            // [R0 R1 R2 R3 R4L]
            col += DIM0_TILE;
        }
        row += DIM1_TILE;
    }

    // free F_1_402_const (R5L), F_0_714_const (R5H), F_128_const (R4H)
    _mve_free_w();
    _mve_free_w();
    _mve_free_w();
}