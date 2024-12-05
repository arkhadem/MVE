#include "aes.hpp"
#include "scalar_kernels.hpp"

static unsigned char xtime(unsigned char x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

void aes_scalar(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {
    aes_config_t *aes_config = (aes_config_t *)config;
    aes_input_t *aes_input = (aes_input_t *)input;

    int num_blocks = aes_config->num_blocks;

    unsigned char *state = aes_input->state;
    unsigned char *RoundKey = aes_config->RoundKey;

    int round = 0;

    // Add the First round key to the state before starting the rounds.
    // AddRoundKey(num_blocks, 0, state, RoundKey);
    unsigned char *my_state = state;
    for (int __i = 0; __i < num_blocks; __i++) {
        for (int i = 0; i < 16; ++i) {
            my_state[i] ^= RoundKey[i];
        }

        my_state += 16;
    }

    // There will be 10 rounds.
    // The first 10-1 rounds are identical.
    // These 10 rounds are executed in the loop below.
    // Last one without MixColumns()
    for (round = 1;; ++round) {
        // SubBytes(num_blocks, state);
        my_state = state;
        for (int __i = 0; __i < num_blocks; __i++) {
            for (int i = 0; i < 16; ++i) {
                my_state[i] = sbox[my_state[i]];
            }

            my_state += 16;
        }

        // ShiftRows(num_blocks, state);
        unsigned char temp;
        my_state = state;
        for (int __i = 0; __i < num_blocks; __i++) {
            // Rotate first row 1 columns to left
            temp = my_state[1];
            my_state[1] = my_state[5];
            my_state[5] = my_state[9];
            my_state[9] = my_state[13];
            my_state[13] = temp;

            // Rotate second row 2 columns to left
            temp = my_state[2];
            my_state[2] = my_state[10];
            my_state[10] = temp;

            temp = my_state[6];
            my_state[6] = my_state[14];
            my_state[14] = temp;

            // Rotate third row 3 columns to left
            temp = my_state[3];
            my_state[3] = my_state[15];
            my_state[15] = my_state[11];
            my_state[11] = my_state[7];
            my_state[7] = temp;

            my_state += 16;
        }

        if (round == 10) {
            break;
        }

        // printf("\n%d %d %d %d\n", state[0], state[1], state[2], state[3]);
        // printf("%d %d %d %d\n", state[4], state[5], state[6], state[7]);
        // printf("%d %d %d %d\n", state[8], state[9], state[10], state[11]);
        // printf("%d %d %d %d\n\n", state[12], state[13], state[14], state[15]);
        // MixColumns(num_blocks, state);
        unsigned char Tmp, Tm, t;
        my_state = state;
        for (int __i = 0; __i < num_blocks; __i++) {
            for (int i = 0; i < 16; i += 4) {
                t = my_state[i + 0];
                Tmp = my_state[i + 0] ^ my_state[i + 1] ^ my_state[i + 2] ^ my_state[i + 3];
                Tm = my_state[i + 0] ^ my_state[i + 1];
                // printf("Tm = %d\n", Tm);
                Tm = xtime(Tm);
                // printf("xtime(Tm) = %d\n", Tm);
                my_state[i + 0] ^= Tm ^ Tmp;
                Tm = my_state[i + 1] ^ my_state[i + 2];
                // printf("Tm = %d\n", Tm);
                Tm = xtime(Tm);
                // printf("xtime(Tm) = %d\n", Tm);
                my_state[i + 1] ^= Tm ^ Tmp;
                Tm = my_state[i + 2] ^ my_state[i + 3];
                // printf("Tm = %d\n", Tm);
                Tm = xtime(Tm);
                // printf("xtime(Tm) = %d\n", Tm);
                my_state[i + 2] ^= Tm ^ Tmp;
                Tm = my_state[i + 3] ^ t;
                // printf("Tm = %d\n", Tm);
                Tm = xtime(Tm);
                // printf("xtime(Tm) = %d\n", Tm);
                my_state[i + 3] ^= Tm ^ Tmp;
                // printf("%d %d %d %d\n", my_state[i + 0], my_state[i + 1], my_state[i + 2], my_state[i + 3]);
            }
            my_state += 16;
        }

        // printf("\n%d %d %d %d\n", state[0], state[1], state[2], state[3]);
        // printf("%d %d %d %d\n", state[4], state[5], state[6], state[7]);
        // printf("%d %d %d %d\n", state[8], state[9], state[10], state[11]);
        // printf("%d %d %d %d\n\n", state[12], state[13], state[14], state[15]);
        // AddRoundKey(num_blocks, round, state, RoundKey);
        const unsigned char *RoundKey_addr = RoundKey + (round * 16);
        my_state = state;
        for (int __i = 0; __i < num_blocks; __i++) {
            for (int i = 0; i < 16; ++i) {
                my_state[i] ^= RoundKey_addr[i];
            }

            my_state += 16;
        }
    }
    // Add round key to last round
    // AddRoundKey(num_blocks, 10, state, RoundKey);
    const unsigned char *RoundKey_addr = RoundKey + 160;
    my_state = state;
    for (int __i = 0; __i < num_blocks; __i++) {
        for (int i = 0; i < 16; ++i) {
            my_state[i] ^= RoundKey_addr[i];
        }

        my_state += 16;
    }
}