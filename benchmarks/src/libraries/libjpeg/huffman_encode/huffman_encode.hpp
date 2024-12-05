#ifndef EF3A51F5_E377_4D38_9626_BD4B472EEB14
#define EF3A51F5_E377_4D38_9626_BD4B472EEB14

#include <stddef.h>
#include <stdint.h>

#include "libjpeg.hpp"

extern const uint8_t huffman_encode_consts[64];
extern const unsigned char bit_loc_constants[64];

typedef struct huffman_encode_config_s : config_t {
    // Number of 8x8 blocks: 1024
    int num_blocks;
} huffman_encode_config_t;

typedef struct huffman_encode_input_s : input_t {
    JCOEFARRAY input_buf;
    JCOEFARRAY *input_addr;
} huffman_encode_input_t;

typedef struct huffman_encode_output_s : output_t {
    // 2 output values for each pixel: temp and temp2
    UJCOEFARRAY output_buf;

    // Each block is 64 pixels, zero bits will be 32 bits
    uint32_t *zero_bits;
} huffman_encode_output_t;

#endif /* EF3A51F5_E377_4D38_9626_BD4B472EEB14 */
