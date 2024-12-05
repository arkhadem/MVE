#include "mve.hpp"
#include "mve_kernels.hpp"
#include <cstdio>

#include "memcmp.hpp"

#define REDUCTION(NEW_DIM1_BLOCK, NEW_TOTAL_BLOCK, OLD_TOTAL_BLOCK)                                \
    _mve_set_dim_length(1, NEW_DIM1_BLOCK);                                                        \
    acc_temp_b = _mve_load_b((const __uint8_t *)forward_mve_memory + NEW_TOTAL_BLOCK, seq_stride); \
    acc_b = _mve_or_b(acc_temp_b, acc_b);                                                          \
    /* free acc_temp_b and acc_b */                                                                \
    _mve_free_b();                                                                                 \
    _mve_free_b();                                                                                 \
    forward_mve_memory += OLD_TOTAL_BLOCK;                                                         \
    _mve_cmpneq_b(acc_b, zero_b);                                                                  \
    _mve_store_b(forward_mve_memory, acc_b, seq_stride);                                           \
    _mve_set_mask();

#define LOAD()                                                   \
    src1_b = _mve_loadro_b(src1_addr, offset, buff_load_stride); \
    src2_b = _mve_loadro_b(src2_addr, offset, buff_load_stride); \
    offset += 8192;                                              \
    _mve_cmpneq_b(src1_b, src2_b);                               \
    /* free src1_b and src2_b */                                 \
    _mve_free_b();                                               \
    _mve_free_b();                                               \
    acc_b = _mve_assign_b(acc_b, round_value_b);                 \
    /* free acc_b */                                             \
    _mve_free_b();                                               \
    _mve_set_mask();                                             \
    round_value_b = _mve_shil_b(round_value_b, 1);               \
    /* free round_value_b */                                     \
    _mve_free_b();

#define APPLY_MASK(value) ((value) & mask)

#define FIND_MASK()                     \
    if (my_mve_element & 0x01) {        \
        mask = 0x01;                    \
        mask_shift = 0;                 \
    } else if (my_mve_element & 0x02) { \
        mask = 0x02;                    \
        mask_shift = 1;                 \
    } else if (my_mve_element & 0x04) { \
        mask = 0x04;                    \
        mask_shift = 2;                 \
    } else if (my_mve_element & 0x08) { \
        mask = 0x08;                    \
        mask_shift = 3;                 \
    } else if (my_mve_element & 0x10) { \
        mask = 0x10;                    \
        mask_shift = 4;                 \
    } else if (my_mve_element & 0x20) { \
        mask = 0x20;                    \
        mask_shift = 5;                 \
    } else if (my_mve_element & 0x40) { \
        mask = 0x40;                    \
        mask_shift = 6;                 \
    } else {                            \
        mask = 0x80;                    \
        mask_shift = 7;                 \
    }

void memcmp_mve(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {
    memcmp_input_t *memcmp_input = (memcmp_input_t *)input;
    memcmp_output_t *memcmp_output = (memcmp_output_t *)output;

    const __uint8_t *src1 = (const __uint8_t *)memcmp_input->src1;
    const __uint8_t *src2 = (const __uint8_t *)memcmp_input->src2;

    const __uint8_t **src1_addr = (const __uint8_t **)memcmp_input->src1_addr;
    const __uint8_t **src2_addr = (const __uint8_t **)memcmp_input->src2_addr;

    __vidx_var seq_stride = {2, 2, 0, 0};

    __uint8_t mve_memory[16384];
    __mdvb zero_b = _mve_set1_b(0);
    _mve_store_b(mve_memory, zero_b, seq_stride);
    _mve_store_b(mve_memory + 8192, zero_b, seq_stride);

    // DIM0: 64 (groups of 128 characters)
    // DIM1: 128 (a single group)
    _mve_set_dim_count(2);
    _mve_set_dim_length(0, 64);
    _mve_set_dim_length(1, 128);

    // When loading input buffer
    // There are 128 chars in a group
    // DIM0: load (64 characters) with stride 128
    // DIM1: load 128 characters randomely
    _mve_set_load_stride(0, 128);
    __vidx_var buff_load_stride = {3, 0, 0, 0};

    __mdvb acc_b = _mve_set1_b(0);
    __mdvb round_value_b = _mve_set1_b(1);
    __mdvb src1_b, src2_b;
    int offset = 0;
    LOAD()
    LOAD()
    LOAD()
    LOAD()
    LOAD()
    LOAD()
    LOAD()
    LOAD()

    // free round_value_b and value_b
    _mve_free_b();

    __mdvb acc_temp_b;

    __uint8_t *forward_mve_memory = mve_memory;

    _mve_store_b(forward_mve_memory, acc_b, seq_stride);

    REDUCTION(64, 4096, 8192)

    REDUCTION(32, 2048, 4096)

    REDUCTION(16, 1024, 2048)

    REDUCTION(8, 512, 1024)

    REDUCTION(4, 256, 512)

    REDUCTION(2, 128, 256)

    REDUCTION(1, 64, 128)

    // free zero_b and acc_b
    _mve_free_b();
    _mve_free_b();

    int min_idx = 10000000;
    for (int i = 0; i < 64; i++) {
        if (forward_mve_memory[i] != 0) {
            __uint8_t *backward_mve_memory = forward_mve_memory;
            int mask = 0;
            int mask_shift = 0;
            char my_mve_element = forward_mve_memory[i];
            FIND_MASK()
            int current_idx = i;
            for (int j = 128; j <= 8192; j <<= 1) {
                backward_mve_memory -= j;
                if ((APPLY_MASK(backward_mve_memory[current_idx]) == 0) && (APPLY_MASK(backward_mve_memory[current_idx + (j / 2)]) != 0))
                    current_idx += (j / 2);
            }
            int group_id = current_idx % 64;
            int element_id = current_idx >> 6;
            current_idx = (8192 * mask_shift) + (group_id << 7) + rand_index[element_id];
            if (current_idx < min_idx)
                min_idx = current_idx;
        }
    }

    if (min_idx != 10000000)
        memcmp_output->return_val[0] = src1[min_idx] < src2[min_idx] ? -1 : 1;
    else
        memcmp_output->return_val[0] = 0;
}