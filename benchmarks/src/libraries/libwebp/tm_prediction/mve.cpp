#include "mve.hpp"
#include "mve_kernels.hpp"
#include <cstdio>

#include "tm_prediction.hpp"

void tm_prediction_mve(int LANE_NUM,
                       config_t *config,
                       input_t *input,
                       output_t *output) {
    tm_prediction_config_t *tm_prediction_config = (tm_prediction_config_t *)config;
    tm_prediction_output_t *tm_prediction_output = (tm_prediction_output_t *)output;
    // tm_prediction_input_t *tm_prediction_input = (tm_prediction_input_t *)input;

    int num_blocks = tm_prediction_config->num_blocks;
    int BPS = tm_prediction_config->pic_size;

    // Dim0: block row
    // Dim1: block column
    // Dim2: different blocks
    _mve_set_dim_count(3);

    // Loading L[r] to all cells of a row
    _mve_set_load_stride(1, BPS);
    __vidx_var L_r_stride = {0, 3, 0, 0};

    // Loading A[c] to all cells of a column
    __vidx_var A_c_stride = {1, 0, 0, 0};

    // Loading A[-1] to all cells of the block
    __vidx_var A_1_stride = {0, 0, 0, 0};

    // Storing output results sequentially in a row
    // with BPS stride between rows
    // random between blocks
    _mve_set_store_stride(1, BPS);
    __vidx_var output_stride = {1, 3, 0, 0};

    _mve_set_dim_length(0, 16);
    _mve_set_dim_length(1, 16);

    LANE_NUM /= 256;

    int DIM2_TILE = num_blocks > LANE_NUM ? LANE_NUM : num_blocks;

    uint8_t **dst = tm_prediction_output->block_dst;

    _mve_set_dim_length(2, DIM2_TILE);

    __mdvw min_w = _mve_set1_w(0);
    __mdvw max_w = _mve_set1_w(255);

    int blk = 0;
    while (blk < num_blocks) {
        int remaining_blks = num_blocks - blk;
        remaining_blks = remaining_blks > DIM2_TILE ? DIM2_TILE : remaining_blks;
        if (remaining_blks != DIM2_TILE) {
            _mve_set_dim_length(2, remaining_blks);
        }

        __mdvb L_r_b = _mve_loadro_b((const __uint8_t **)dst, -1, L_r_stride);
        __mdvw L_r_w = _mve_cvtu_btow(L_r_b);
        // free L_r_b
        _mve_free_b();
        __mdvb A_c_b = _mve_loadro_b((const __uint8_t **)dst, -1 * BPS, A_c_stride);
        __mdvw A_c_w = _mve_cvtu_btow(A_c_b);
        // free A_c_b
        _mve_free_b();
        __mdvw AL_w = _mve_add_w(L_r_w, A_c_w);
        // free L_r_w and A_c_w
        _mve_free_w();
        _mve_free_w();
        __mdvb A_1_b = _mve_loadro_b((const __uint8_t **)dst, -1 * BPS - 1, A_1_stride);
        __mdvw A_1_w = _mve_cvtu_btow(A_1_b);
        // free A_1_b
        _mve_free_b();
        __mdvw result_w = _mve_sub_w(AL_w, A_1_w);
        // free AL_w and A_1_w
        _mve_free_w();
        _mve_free_w();
        __mdvw result_min_w = _mve_min_w(result_w, max_w);
        // free result_w
        _mve_free_w();
        __mdvw result_min_max_w = _mve_max_w(result_min_w, min_w);
        // free result_min_w
        _mve_free_w();
        __mdvb result_b = _mve_cvt_wtob(result_min_max_w);
        // free result_min_max_w
        _mve_free_w();
        _mve_storer_b(dst, result_b, output_stride);
        // free result_b
        _mve_free_b();

        blk += DIM2_TILE;
        dst += DIM2_TILE;
    }

    // free max_w and min_w
    _mve_free_w();
    _mve_free_w();

    // for (int i = 0; i < num_blocks; i++) {
    //     uint8_t *src = tm_prediction_output->block_dst[i];
    //     int idx = src - tm_prediction_input->dst;
    //     printf("block[%d][%d] %d:\n", idx / BPS, idx % BPS, i);
    //     for (int j = 0; j < 16; j++) {
    //         for (int k = 0; k < 16; k++) {
    //             printf("%d ", src[j * BPS + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}