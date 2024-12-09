#include "csum.hpp"
#include "kvazaar.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

void csum_neon(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {

    csum_config_t *csum_config = (csum_config_t *)config;
    csum_input_t *csum_input = (csum_input_t *)input;
    csum_output_t *csum_output = (csum_output_t *)output;

    int count = csum_config->count;
    uint32_t *ptr = (uint32_t *)csum_input->ptr;
    int16_t *sum = csum_output->sum;

    uint32_t *hptr;

    for (int i = 0; i < count; i++) {

        hptr = ptr + i * BLOCK_16K;

        uint64x2_t vsum0 = {0, 0};
        uint64x2_t vsum1 = {0, 0};
        uint64x2_t vsum2 = {0, 0};
        uint64x2_t vsum3 = {0, 0};
        /* sum_tmp all halfwords, assume misaligned accesses are handled in HW */
        for (int j = 0; j < BLOCK_16K; j += 16) {
            vsum0 = vpadalq_u32(vsum0, vld1q_u32(hptr + 0));
            vsum1 = vpadalq_u32(vsum1, vld1q_u32(hptr + 4));
            vsum2 = vpadalq_u32(vsum2, vld1q_u32(hptr + 8));
            vsum3 = vpadalq_u32(vsum3, vld1q_u32(hptr + 12));
            hptr += 16;
        }

        /* Fold vsum2 and vsum3 into vsum0 and vsum1 */
        vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum2));
        vsum1 = vpadalq_u32(vsum1, vreinterpretq_u32_u64(vsum3));

        /* Fold vsum1 into vsum0 */
        vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum1));
        uint64_t sum_tmp = vaddlvq_u32(vreinterpretq_u32_u64(vsum0));

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