#include "mve.hpp"
#include "mve_kernels.hpp"

#include "dispatch_alpha.hpp"

void dispatch_alpha_mve(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    dispatch_alpha_config_t *dispatch_alpha_config = (dispatch_alpha_config_t *)config;
    dispatch_alpha_input_t *dispatch_alpha_input = (dispatch_alpha_input_t *)input;
    dispatch_alpha_output_t *dispatch_alpha_output = (dispatch_alpha_output_t *)output;

    // Dim0: linear rows and columns
    _mve_set_dim_count(1);

    __mdvb mask_first = _mve_set1_b(0xff);
    __mdvb mask_second;

    __vidx_var load_stride = {1, 0, 0, 0};
    _mve_set_store_stride(0, 4);
    __vidx_var store_stride = {3, 0, 0, 0};

    __vidx_var sequential_stride = {1, 0, 0, 0};

    int num_rows = dispatch_alpha_config->num_rows;
    int num_cols = dispatch_alpha_config->num_cols;

    int total_pixels = num_cols * num_rows;

    int DIM0_TILE = total_pixels > LANE_NUM ? LANE_NUM : total_pixels;

    uint8_t *alpha = dispatch_alpha_input->alpha;
    uint8_t *dst = dispatch_alpha_output->dst;

    uint8_t reduction_mem[8192];

    _mve_set_dim_length(0, DIM0_TILE);

    int pixel = 0;
    while (pixel < total_pixels) {
        int remaining_pixels = total_pixels - pixel;
        remaining_pixels = remaining_pixels > DIM0_TILE ? DIM0_TILE : remaining_pixels;
        if (remaining_pixels != DIM0_TILE) {
            _mve_set_dim_length(0, remaining_pixels);
        }

        __mdvb ref_b = _mve_load_b(alpha, load_stride);
        mask_first = _mve_and_b(mask_first, ref_b);
        // free mask_first
        _mve_free_b();

        _mve_store_b(dst, ref_b, store_stride);
        // free ref_b
        _mve_free_b();

        pixel += remaining_pixels;
        alpha += remaining_pixels;
        dst += remaining_pixels << 2;
    }

    // Start mask_first reduction from here
    _mve_set_dim_length(0, LANE_NUM >> 1);
    _mve_store_b(reduction_mem, mask_first, sequential_stride);

    if (DIM0_TILE > 8192) {
        // Step1: 8192

        _mve_set_dim_length(0, 8192);
        mask_second = _mve_load_b(reduction_mem + 8192, sequential_stride);
        mask_first = _mve_and_b(mask_first, mask_second);
        // free mask_first and mask_second
        _mve_free_b();
        _mve_free_b();
        _mve_store_b(reduction_mem, mask_first, sequential_stride);
    }

    if (DIM0_TILE > 4096) {
        // Step2: 4096

        _mve_set_dim_length(0, 4096);
        mask_second = _mve_load_b(reduction_mem + 4096, sequential_stride);
        mask_first = _mve_and_b(mask_first, mask_second);
        // free mask_first and mask_second
        _mve_free_b();
        _mve_free_b();
        _mve_store_b(reduction_mem, mask_first, sequential_stride);
    }

    if (DIM0_TILE > 2048) {
        // Step3: 2048

        _mve_set_dim_length(0, 2048);
        mask_second = _mve_load_b(reduction_mem + 2048, sequential_stride);
        mask_first = _mve_and_b(mask_first, mask_second);
        // free mask_first and mask_second
        _mve_free_b();
        _mve_free_b();
        _mve_store_b(reduction_mem, mask_first, sequential_stride);
    }

    // Step3: 1024

    _mve_set_dim_length(0, 1024);
    mask_second = _mve_load_b(reduction_mem + 1024, sequential_stride);
    mask_first = _mve_and_b(mask_first, mask_second);
    // free mask_first and mask_second
    _mve_free_b();
    _mve_free_b();
    _mve_store_b(reduction_mem, mask_first, sequential_stride);

    // Step3: 512

    _mve_set_dim_length(0, 512);
    mask_second = _mve_load_b(reduction_mem + 512, sequential_stride);
    mask_first = _mve_and_b(mask_first, mask_second);
    // free mask_first and mask_second
    _mve_free_b();
    _mve_free_b();
    _mve_store_b(reduction_mem, mask_first, sequential_stride);

    // Step3: 256

    _mve_set_dim_length(0, 256);
    mask_second = _mve_load_b(reduction_mem + 256, sequential_stride);
    mask_first = _mve_and_b(mask_first, mask_second);
    // free mask_first and mask_second
    _mve_free_b();
    _mve_free_b();
    _mve_store_b(reduction_mem, mask_first, sequential_stride);

    // Step3: 128

    _mve_set_dim_length(0, 128);
    mask_second = _mve_load_b(reduction_mem + 128, sequential_stride);
    mask_first = _mve_and_b(mask_first, mask_second);
    // free mask_first and mask_second
    _mve_free_b();
    _mve_free_b();
    _mve_store_b(reduction_mem, mask_first, sequential_stride);

    // Step3: 64

    _mve_set_dim_length(0, 64);
    mask_second = _mve_load_b(reduction_mem + 64, sequential_stride);
    mask_first = _mve_and_b(mask_first, mask_second);
    // free mask_first and mask_second
    _mve_free_b();
    _mve_free_b();
    _mve_store_b(reduction_mem, mask_first, sequential_stride);
    // free mask_first
    _mve_free_b();

    dispatch_alpha_output->return_val[0] = 0;
    for (int i = 0; i < 64; i++)
        if (reduction_mem[i] != 0xff) {
            dispatch_alpha_output->return_val[0] = 1;
            break;
        }
}