#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "aes.hpp"
#include "boringssl.hpp"

// Does vertical convolution to produce one output row. The filter values and
// length are given in the first two parameters. These are applied to each
// of the rows pointed to in the |source_data_rows| array, with each row
// being |pixel_width| wide.
//
// The output must have room for |pixel_width * 4| bytes.
void aes_neon(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {
    aes_config_t *aes_config = (aes_config_t *)config;
    aes_input_t *aes_input = (aes_input_t *)input;

    int num_blocks = aes_config->num_blocks;

    unsigned char *state = aes_input->state;
    unsigned char *RoundKey = aes_config->RoundKey;

    uint8x16_t RoundKey_v[11];

    uint8x16_t zero_v = vdupq_n_u8(0);

    for (int round = 0; round < 11; round++) {
        RoundKey_v[round] = vld1q_u8(RoundKey);
        RoundKey += 16;
    }

    for (int sample = 0; sample < num_blocks; sample++) {
        uint8x16_t state_v = vld1q_u8(state);

        state_v = veorq_u8(state_v, RoundKey_v[0]);

        int round = 0;

        for (round = 1;; ++round) {
            // SubBytes
            // ShiftRows
            state_v = vaeseq_u8(state_v, zero_v);
            if (round == 10) {
                break;
            }
            // MixColumns
            state_v = vaesmcq_u8(state_v);
            // AddRoundKey
            state_v = veorq_u8(state_v, RoundKey_v[round]);
        }
        state_v = veorq_u8(state_v, RoundKey_v[10]);

        vst1q_u8(state, state_v);

        state += 16;
    }
}
