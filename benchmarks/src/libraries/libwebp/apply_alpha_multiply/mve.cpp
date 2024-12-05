#include "mve.hpp"
#include "mve_kernels.hpp"

#include "apply_alpha_multiply.hpp"

void apply_alpha_multiply_mve(int LANE_NUM,
                              config_t *config,
                              input_t *input,
                              output_t *output) {
    apply_alpha_multiply_config_t *apply_alpha_multiply_config = (apply_alpha_multiply_config_t *)config;
    apply_alpha_multiply_input_t *apply_alpha_multiply_input = (apply_alpha_multiply_input_t *)input;

    // Dim0: RGB (3)
    // Dim1: pixels
    _mve_set_dim_count(2);

    // Read the same alpha across RGB
    // Read next pixel's alpha stride 4
    _mve_set_load_stride(1, 4);
    __vidx_var alpha_stride = {0, 3, 0, 0};

    // Read adjacent RGB channels
    // Read next pixel's RGB channel stride 4
    __vidx_var rgb_stride = {1, 3, 0, 0};

    // Store adjacent RGB channels
    // Store next pixel's RGB channel stride 4
    _mve_set_store_stride(1, 4);
    __vidx_var output_stride = {1, 3, 0, 0};

    int num_rows = apply_alpha_multiply_config->num_rows;
    int num_cols = apply_alpha_multiply_config->num_cols;

    int total_pixels = num_cols * num_rows;

    _mve_set_dim_length(0, 4);
    LANE_NUM >>= 2;

    int DIM1_TILE = total_pixels > LANE_NUM ? LANE_NUM : total_pixels;

    uint8_t *rgba = apply_alpha_multiply_input->rgba;

    _mve_set_dim_length(1, DIM1_TILE);

    __mdvw one_w = _mve_set1_w(1);

    int pixel = 0;
    while (pixel < total_pixels) {
        int remaining_pixels = total_pixels - pixel;
        remaining_pixels = remaining_pixels > DIM1_TILE ? DIM1_TILE : remaining_pixels;
        if (remaining_pixels != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_pixels);
        }

        __mdvb alpha_b = _mve_load_b(rgba, alpha_stride);
        __mdvw alpha_w = _mve_cvtu_btow(alpha_b);
        // free alpha_b
        _mve_free_b();

        __mdvb rgb_b = _mve_load_b(rgba + 1, rgb_stride);
        __mdvw rgb_w = _mve_cvtu_btow(rgb_b);
        // free rgb_b
        _mve_free_b();

        __mdvw rgb_mult_w = _mve_mul_w(alpha_w, rgb_w);
        // free alpha_w and rgb_w
        _mve_free_w();
        _mve_free_w();

        __mdvw rgb_shr8_w = _mve_shiru_w(rgb_mult_w, 8);

        __mdvw rgb_p1_w = _mve_add_w(rgb_mult_w, one_w);
        // free rgb_mult_w
        _mve_free_w();

        __mdvw rgb_shr8_p1_w = _mve_add_w(rgb_shr8_w, rgb_p1_w);
        // free rgb_shr8_w and rgb_p1_w
        _mve_free_w();
        _mve_free_w();

        __mdvw final_rgb_w = _mve_shiru_w(rgb_shr8_p1_w, 8);
        // free rgb_shr8_p1_w
        _mve_free_w();

        __mdvb final_rgb_b = _mve_cvt_wtob(final_rgb_w);
        // free final_rgb_w
        _mve_free_w();

        _mve_store_b(rgba + 1, final_rgb_b, output_stride);
        // free final_rgb_b
        _mve_free_b();

        pixel += remaining_pixels;
        rgba += remaining_pixels << 2;
    }
    // free one_w
    _mve_free_w();
}