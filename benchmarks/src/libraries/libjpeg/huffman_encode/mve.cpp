
#include "mve.hpp"
#include "mve_kernels.hpp"

#include "huffman_encode.hpp"

void huffman_encode_mve(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    huffman_encode_config_t *huffman_encode_config = (huffman_encode_config_t *)config;
    huffman_encode_input_t *huffman_encode_input = (huffman_encode_input_t *)input;
    huffman_encode_output_t *huffman_encode_output = (huffman_encode_output_t *)output;
    int16_t **input_addr = huffman_encode_input->input_addr;
    // Dim0: different blocks
    // Dim1: each of 64 pixels
    _mve_set_dim_count(2);

    // Loading input: with 64 stride in DIM0, randomly in DIM1
    _mve_set_load_stride(0, 64);
    __vidx_var input_load_stride = {3, 0, 0, 0};

    // Stride for loading bit locations
    __vidx_var bit_loc_stride = {0, 1, 0, 0};
    __vidx_var sequential_stride = {2, 2, 0, 0};
    unsigned char zero_bits_storage[8192];

    __vidx_var output_stride = {3, 1, 0, 0};
    // Storing output values: with 128 stride in DIM0, with 1 stride in DIM1
    // This is deprecated to the code itself
    // _mve_set_store_stride(0, 128);
    uint16_t *output_addr = huffman_encode_output->output_buf;
    uint32_t *zero_addr = huffman_encode_output->zero_bits;

    // Storing output values: 64 bits per block (8x8-bit)
    // Storing with stride 8 in DIM0
    // Storing with stride 1 in DIM1
    // This is deprecated to the code itself
    // _mve_set_store_stride(0, 8);

    // R5H
    __mdvw zero_w = _mve_set1_w(0);
    // R5LH
    __mdvb zero_b = _mve_set1_b(0);

    int num_blocks = huffman_encode_config->num_blocks;

    // Second dimension is a block of 64 pixels
    LANE_NUM /= 64;

    int DIM0_TILE = num_blocks > LANE_NUM ? LANE_NUM : num_blocks;

    JDIMENSION blk = 0;
    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, 64);

    // R5LL
    __mdvb bit_loc_b = _mve_load_b(bit_loc_constants, bit_loc_stride);

    while (blk < num_blocks) {
        _mve_set_dim_length(1, 64);

        unsigned char *my_zero_bits_storage = zero_bits_storage;

        int remaining_blks = num_blocks - blk;
        remaining_blks = remaining_blks > DIM0_TILE ? DIM0_TILE : remaining_blks;
        if (remaining_blks != DIM0_TILE) {
            _mve_set_dim_length(0, remaining_blks);
        }

        // R0L
        __mdvw temp1 = _mve_loadro_w((const int16_t **)input_addr, blk << 6, input_load_stride);
        // R1L
        __mdvw temp2 = _mve_shiru_w(temp1, 15);

        // R0H
        __mdvw temp3 = _mve_xor_w(temp1, temp2);
        // free temp1 (R0L)
        _mve_free_w();

        // R0L
        __mdvw temp4 = _mve_sub_w(temp3, temp2);
        // free temp3 (R0H)
        _mve_free_w();

        // R0H
        __mdvw temp5 = _mve_shirs_w(temp4, 4);
        // free temp4 (R0L)
        _mve_free_w();

        // R1H
        __mdvw temp6 = _mve_xor_w(temp2, temp5);
        // free temp2 (R1L)
        _mve_free_w();

        _mve_set_store_stride(0, 128);
        _mve_store_w((int16_t *)output_addr, temp5, output_stride);

        _mve_store_w((int16_t *)output_addr + 64, temp6, output_stride);
        // free temp6 (R1H)
        _mve_free_w();

        // Start zero bit calculation and reduction from here

        // Step1: calculate zero[63:0]

        _mve_cmpeq_w(temp5, zero_w);
        // free temp5 (R0H)
        _mve_free_w();
        // R1
        __mdvb zero_bits1 = _mve_assign_b(bit_loc_b, zero_b);
        _mve_set_mask();
        _mve_store_b(my_zero_bits_storage, zero_bits1, sequential_stride);

        // Step2: calculate zero[31:0]

        my_zero_bits_storage += 4096;
        _mve_set_dim_length(1, 32);
        // R2
        __mdvb zero_bits2 = _mve_load_b(my_zero_bits_storage, sequential_stride);
        // R3
        __mdvb zero_bits3 = _mve_xor_b(zero_bits1, zero_bits2);
        // free zero_bits1 (R1) and zero_bits2 (R2)
        _mve_free_b();
        _mve_free_b();
        _mve_store_b(my_zero_bits_storage, zero_bits3, sequential_stride);

        // Step3: calculate zero[15:0]

        my_zero_bits_storage += 2048;
        _mve_set_dim_length(1, 16);
        // R4
        __mdvb zero_bits4 = _mve_load_b(my_zero_bits_storage, sequential_stride);
        // R1
        __mdvb zero_bits5 = _mve_xor_b(zero_bits3, zero_bits4);
        // free zero_bits3 (R3) and zero_bits4 (R4)
        _mve_free_b();
        _mve_free_b();
        _mve_store_b(my_zero_bits_storage, zero_bits5, sequential_stride);

        // Step4: calculate zero[7:0] and store

        my_zero_bits_storage += 1024;
        _mve_set_dim_length(1, 8);
        // R2
        __mdvb zero_bits6 = _mve_load_b(my_zero_bits_storage, sequential_stride);
        // R3
        __mdvb zero_bits7 = _mve_xor_b(zero_bits5, zero_bits6);
        // free zero_bits5 (R1) and zero_bits6 (R2)
        _mve_free_b();
        _mve_free_b();

        _mve_set_store_stride(0, 8);
        _mve_store_b((uint8_t *)zero_addr, zero_bits7, output_stride);
        // free zero_bits7 (R3)
        _mve_free_b();

        blk += DIM0_TILE;
        output_addr += 16384;
        zero_addr += 256;
    }

    // free zero_w (R5H), zero_b (R5LH), and bit_loc_b (R5LL)
    _mve_free_w();
    _mve_free_b();
    _mve_free_b();
}