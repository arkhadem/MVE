#include "csum.hpp"
#include "kvazaar.hpp"
#include "mve.hpp"
#include <cstdint>
#include <cstdio>

void csum_rvv(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t *csum_input = (csum_input_t *)input;
    csum_output_t *csum_output = (csum_output_t *)output;

    int count = csum_config->count;
    int32_t *ptr = csum_input->ptr;
    int16_t *sum = csum_output->sum;

    if (LANE_NUM == 8192) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_8K);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_8K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_8K);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);
            dw_0 = _mve_load_dw(in + BLOCK_8K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_4K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_4K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_2K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_2K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_256; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else if (LANE_NUM == 4096) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_4K);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_4K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_4K);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);

            for (int block_offset = BLOCK_4K; block_offset < BLOCK_16K; block_offset += BLOCK_4K) {
                dw_0 = _mve_load_dw(in + block_offset, seq_stride);
                qw_0 = _mve_cvts_dwtoqw(dw_0);
                qw_sum = _mve_add_qw(qw_sum, qw_0);
            }
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_2K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_2K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_128; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else if (LANE_NUM == 2048) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_2K);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_2K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_2K);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);

            for (int block_offset = BLOCK_2K; block_offset < BLOCK_16K; block_offset += BLOCK_2K) {
                dw_0 = _mve_load_dw(in + block_offset, seq_stride);
                qw_0 = _mve_cvts_dwtoqw(dw_0);
                qw_sum = _mve_add_qw(qw_sum, qw_0);
            }
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_64; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else if (LANE_NUM == 1024) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_1K);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_1K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_1K);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);

            for (int block_offset = BLOCK_1K; block_offset < BLOCK_16K; block_offset += BLOCK_1K) {
                dw_0 = _mve_load_dw(in + block_offset, seq_stride);
                qw_0 = _mve_cvts_dwtoqw(dw_0);
                qw_sum = _mve_add_qw(qw_sum, qw_0);
            }
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_32; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else if (LANE_NUM == 512) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_512);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_512];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_512);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);

            for (int block_offset = BLOCK_512; block_offset < BLOCK_16K; block_offset += BLOCK_512) {
                dw_0 = _mve_load_dw(in + block_offset, seq_stride);
                qw_0 = _mve_cvts_dwtoqw(dw_0);
                qw_sum = _mve_add_qw(qw_sum, qw_0);
            }
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_16); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_16, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_16; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else if (LANE_NUM == 256) {
        _mve_set_dim_count(1);
        _mve_set_dim_length(0, BLOCK_256);
        __vidx_var seq_stride = {1, 0, 0, 0};

        int64_t tmp[BLOCK_256];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_256);

            dw_0 = _mve_load_dw(in, seq_stride);
            qw_sum = _mve_cvts_dwtoqw(dw_0);

            for (int block_offset = BLOCK_256; block_offset < BLOCK_16K; block_offset += BLOCK_256) {
                dw_0 = _mve_load_dw(in + block_offset, seq_stride);
                qw_0 = _mve_cvts_dwtoqw(dw_0);
                qw_sum = _mve_add_qw(qw_sum, qw_0);
            }
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_16); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_16, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_dim_length(0, BLOCK_8); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_8, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_store_qw(tmp, qw_sum, seq_stride);

            for (j = 0; j < BLOCK_8; j++)
                sum_tmp += tmp[j];

            /* Fold 64-bit sum_tmp to 32 bits */
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
            assert(sum_tmp == (uint32_t)sum_tmp);

            /* Fold 32-bit sum_tmp to 16 bits */
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
            assert(sum_tmp == (uint16_t)sum_tmp);

            *(sum + i) = sum_tmp;
        }
    } else {
        printf("Unsupported LANE_NUM: %d\n", LANE_NUM);
        exit(1);
    }
}