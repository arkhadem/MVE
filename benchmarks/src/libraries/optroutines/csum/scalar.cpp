#include "cstdint"
#include "csum.hpp"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void csum_scalar(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {

    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t *csum_input = (csum_input_t *)input;
    csum_output_t *csum_output = (csum_output_t *)output;

    int count = csum_config->count;
    int32_t *ptr = csum_input->ptr;
    int16_t *sum = csum_output->sum;

    int32_t *hptr;

    for (int i = 0; i < count; i++) {

        hptr = ptr + i * BLOCK_16K;
        int64_t sum_tmp = 0; /* Need 64-bit accumulator when nbytes > 64K */

        /* sum_tmp all halfwords, assume misaligned accesses are handled in HW */
        for (int j = 0; j < BLOCK_16K; j++) {
            sum_tmp += *hptr++;
        }

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