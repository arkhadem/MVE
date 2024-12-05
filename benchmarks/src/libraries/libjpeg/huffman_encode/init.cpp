#include "huffman_encode.hpp"

#include "libjpeg.hpp"

#include "benchmark.hpp"

#include "init.hpp"

#include <stdint.h>

int huffman_encode_init(size_t cache_size,
                        int LANE_NUM,
                        config_t *&config,
                        input_t **&input,
                        output_t **&output) {

    huffman_encode_config_t *huffman_encode_config = (huffman_encode_config_t *)config;
    huffman_encode_input_t **huffman_encode_input = (huffman_encode_input_t **)input;
    huffman_encode_output_t **huffman_encode_output = (huffman_encode_output_t **)output;

    // configuration
    init_1D<huffman_encode_config_t>(1, huffman_encode_config);
    huffman_encode_config->num_blocks = 1024;

    // in/output versions
    size_t input_size = (huffman_encode_config->num_blocks * 64) * sizeof(JCOEF);
    size_t output_size = (2 * huffman_encode_config->num_blocks * 64) * sizeof(JCOEF) +
                         (2 * huffman_encode_config->num_blocks * 2) * sizeof(uint32_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<huffman_encode_input_t *>(count, huffman_encode_input);
    init_1D<huffman_encode_output_t *>(count, huffman_encode_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<huffman_encode_input_t>(1, huffman_encode_input[i]);
        init_1D<huffman_encode_output_t>(1, huffman_encode_output[i]);

        random_init_1D<JCOEF>(huffman_encode_config->num_blocks * 64, huffman_encode_input[i]->input_buf);
        init_1D<JCOEFARRAY>(64, huffman_encode_input[i]->input_addr);
        for (int j = 0; j < 64; j++) {
            huffman_encode_input[i]->input_addr[j] = huffman_encode_input[i]->input_buf + bit_loc_constants[j];
        }
        random_init_1D<UJCOEF>(2 * huffman_encode_config->num_blocks * 64, huffman_encode_output[i]->output_buf);
        random_init_1D<uint32_t>(2 * huffman_encode_config->num_blocks, huffman_encode_output[i]->zero_bits);
    }

    config = (config_t *)huffman_encode_config;
    input = (input_t **)huffman_encode_input;
    output = (output_t **)huffman_encode_output;

    return count;
}

const uint8_t huffman_encode_consts[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};

const unsigned char bit_loc_constants[64] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7};
