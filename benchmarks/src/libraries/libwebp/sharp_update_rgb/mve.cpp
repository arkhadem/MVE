#include "mve.hpp"
#include "mve_kernels.hpp"

#include "sharp_update_rgb.hpp"

void sharp_update_rgb_mve(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    sharp_update_rgb_config_t *sharp_update_rgb_config = (sharp_update_rgb_config_t *)config;
    sharp_update_rgb_input_t *sharp_update_rgb_input = (sharp_update_rgb_input_t *)input;
    sharp_update_rgb_output_t *sharp_update_rgb_output = (sharp_update_rgb_output_t *)output;

    // Dim0: linear rows and columns
    _mve_set_dim_count(1);

    __vidx_var stride = {1, 0, 0, 0};

    int num_rows = sharp_update_rgb_config->num_rows;
    int num_cols = sharp_update_rgb_config->num_cols;

    int total_pixels = num_cols * num_rows;

    int DIM0_TILE = total_pixels > LANE_NUM ? LANE_NUM : total_pixels;

    int16_t *src = sharp_update_rgb_input->src;
    int16_t *ref = sharp_update_rgb_input->ref;
    int16_t *dst = sharp_update_rgb_output->dst;

    _mve_set_dim_length(0, DIM0_TILE);

    int pixel = 0;
    while (pixel < total_pixels) {
        int remaining_pixels = total_pixels - pixel;
        remaining_pixels = remaining_pixels > DIM0_TILE ? DIM0_TILE : remaining_pixels;
        if (remaining_pixels != DIM0_TILE) {
            _mve_set_dim_length(0, remaining_pixels);
        }

        __mdvw ref_w = _mve_load_w(ref, stride);
        __mdvw src_w = _mve_load_w(src, stride);
        __mdvw diff_uv_w = _mve_sub_w(ref_w, src_w);
        // free ref_w and src_w
        _mve_free_w();
        _mve_free_w();
        __mdvw dst_w = _mve_load_w(dst, stride);
        __mdvw new_dst_w = _mve_add_w(dst_w, diff_uv_w);
        // free dst_w and diff_uv_w
        _mve_free_w();
        _mve_free_w();
        _mve_store_w(dst, new_dst_w, stride);
        // free new_dst_w
        _mve_free_w();

        pixel += remaining_pixels;
        src += remaining_pixels;
        ref += remaining_pixels;
        dst += remaining_pixels;
    }
}