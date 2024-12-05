#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "boringssl.hpp"

extern const unsigned char sbox[256];

typedef struct aes_config_s : config_t {
    int num_blocks;
    unsigned char RoundKey[176];
} aes_config_t;

typedef struct aes_input_s : input_t {
    unsigned char *state;
} aes_input_t;

typedef struct aes_output_s : output_t {
    uint32_t dummy[1];
} aes_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
