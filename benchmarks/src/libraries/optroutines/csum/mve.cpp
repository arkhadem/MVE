#include "mve.hpp"
#include "cstdint"
#include "csum.hpp"
#include "kvazaar.hpp"
#include <cstdint>
#include <cstdio>

void csum_mve(int LANE_NUM,
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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_4K);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_8K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_4K); // # elements returned to CPU
            dw_0 = _mve_load_dw(in, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + BLOCK_8K, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_2K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_4K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_2K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_2K);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_4K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_2K); // # elements returned to CPU
            dw_0 = _mve_load_dw(in + 0 * BLOCK_4K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + 1 * BLOCK_4K, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 2 * BLOCK_4K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 3 * BLOCK_4K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_2K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_1K);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_2K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
            dw_0 = _mve_load_dw(in + 0 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + 1 * BLOCK_2K, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 2 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 3 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 4 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 5 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 6 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 7 * BLOCK_2K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_512);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_1K];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
            dw_0 = _mve_load_dw(in + 0 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + 1 * BLOCK_1K, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 2 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 3 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 4 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 5 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 6 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 7 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 8 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 9 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 10 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 11 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 12 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 13 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 14 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 15 * BLOCK_1K, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_256);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_512];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
            dw_0 = _mve_load_dw(in + 0 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + 1 * BLOCK_512, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 2 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 3 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 4 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 5 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 6 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 7 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 8 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 9 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 10 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 11 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 12 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 13 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 14 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 15 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 16 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 17 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 18 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 19 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 20 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 21 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 22 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 23 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 24 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 25 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 26 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 27 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 28 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 29 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 30 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 31 * BLOCK_512, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_16); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_16, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        _mve_set_dim_count(2);
        _mve_set_dim_length(0, BLOCK_128);
        _mve_set_dim_length(1, 2);
        __vidx_var seq_stride = {2, 2, 0, 0};

        int64_t tmp[BLOCK_256];
        int64_t sum_tmp;
        int32_t *in;

        int i, j;

        __mdvdw dw_0, dw_1;
        __mdvqw qw_0, qw_1, qw_sum;

        for (i = 0; i < count; i++) {
            sum_tmp = 0;
            in = ptr + i * BLOCK_16K;

            // Iteration 0
            _mve_set_dim_length(0, BLOCK_128); // # elements returned to CPU
            dw_0 = _mve_load_dw(in + 0 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            dw_1 = _mve_load_dw(in + 1 * BLOCK_256, seq_stride);
            qw_1 = _mve_cvts_dwtoqw(dw_1);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_0, qw_1);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 2 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 3 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 4 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 5 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 6 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 7 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 8 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 9 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 10 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 11 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 12 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 13 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 14 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 15 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 16 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 17 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 18 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 19 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 20 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 21 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 22 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 23 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 24 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 25 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 26 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 27 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 28 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 29 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 30 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 31 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 32 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 33 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 34 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 35 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 36 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 37 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 38 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 39 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 40 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 41 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 42 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 43 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 44 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 45 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 46 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 47 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 48 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 49 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 50 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 51 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 52 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 53 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 54 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 55 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 56 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 57 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 58 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 59 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 60 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 61 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 62 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            dw_0 = _mve_load_dw(in + 63 * BLOCK_256, seq_stride);
            qw_0 = _mve_cvts_dwtoqw(dw_0);
            _mve_free_dw();
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 16 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 1
            _mve_set_dim_length(0, BLOCK_64); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_128, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 8 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 2
            _mve_set_dim_length(0, BLOCK_32); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_64, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 4 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 3
            _mve_set_dim_length(0, BLOCK_16); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_32, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 2 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 4
            _mve_set_dim_length(0, BLOCK_8); // # elements returned to CPU
            qw_0 = _mve_load_qw(tmp + BLOCK_16, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_unset_active_element(1, 0); // turn off the last 1 SAs
            _mve_store_qw(tmp, qw_sum, seq_stride);

            // Iteration 5
            _mve_set_all_elements(1);
            qw_0 = _mve_load_qw(tmp + BLOCK_8, seq_stride);
            qw_sum = _mve_add_qw(qw_sum, qw_0);
            _mve_free_qw();
            _mve_free_qw();
            _mve_store_qw(tmp, qw_sum, seq_stride);
            _mve_free_qw();

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
        printf("Error: unsupported LANE_NUM %d\n", LANE_NUM);
        exit(1);
    }
}