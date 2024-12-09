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
        dw_1 = _mve_load_dw(in + BLOCK_8K, seq_stride);
        qw_0 = _mve_cvts_dwtoqw(dw_0);
        _mve_free_dw();
        qw_1 = _mve_cvts_dwtoqw(dw_1);
        _mve_free_dw();
        qw_sum = _mve_add_qw(qw_0, qw_1);
        _mve_free_qw();
        _mve_free_qw();
        _mve_unset_active_element(1, 0); // turn off the last 16 SAs
        _mve_store_qw(tmp, qw_sum, seq_stride);
        _mve_free_qw();

        // Iteration 1
        _mve_set_dim_length(0, BLOCK_2K); // # elements returned to CPU
        qw_0 = _mve_load_qw(tmp + BLOCK_4K, seq_stride);
        qw_sum = _mve_add_qw(qw_sum, qw_0);
        _mve_free_qw();
        _mve_unset_active_element(1, 0); // turn off the last 8 SAs
        _mve_store_qw(tmp, qw_sum, seq_stride);
        _mve_free_qw();

        // Iteration 2
        _mve_set_dim_length(0, BLOCK_1K); // # elements returned to CPU
        qw_0 = _mve_load_qw(tmp + BLOCK_2K, seq_stride);
        qw_sum = _mve_add_qw(qw_sum, qw_0);
        _mve_free_qw();
        _mve_unset_active_element(1, 0); // turn off the last 4 SAs
        _mve_store_qw(tmp, qw_sum, seq_stride);
        _mve_free_qw();

        // Iteration 3
        _mve_set_dim_length(0, BLOCK_512); // # elements returned to CPU
        qw_0 = _mve_load_qw(tmp + BLOCK_1K, seq_stride);
        qw_sum = _mve_add_qw(qw_sum, qw_0);
        _mve_free_qw();
        _mve_unset_active_element(1, 0); // turn off the last 2 SAs
        _mve_store_qw(tmp, qw_sum, seq_stride);
        _mve_free_qw();

        // Iteration 4
        _mve_set_dim_length(0, BLOCK_256); // # elements returned to CPU
        qw_0 = _mve_load_qw(tmp + BLOCK_512, seq_stride);
        qw_sum = _mve_add_qw(qw_sum, qw_0);
        _mve_free_qw();
        _mve_unset_active_element(1, 0); // turn off the last 1 SAs
        _mve_store_qw(tmp, qw_sum, seq_stride);
        _mve_free_qw();

        // Iteration 5
        _mve_set_all_elements(1);
        qw_0 = _mve_load_qw(tmp + BLOCK_256, seq_stride);
        _mve_free_qw();
        qw_sum = _mve_add_qw(qw_sum, qw_0);
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
}