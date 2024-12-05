#include "mve.hpp"
#include "mve_kernels.hpp"

#include "adler32.hpp"

void adler32_mve(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    int32_t sum2_mem_dw[BLOCK_8K];
    int16_t *sum2_mem_w = (int16_t *)sum2_mem_dw;
    int16_t sum1_mem_w[BLOCK_8K];

    uint32_t total_sum1 = (adler >> 16) & 0xffff;
    uint32_t total_sum2 = adler & 0xffff;

    _mve_set_dim_count(1);

    int shift_val = 13;

    int blocks = len / LANE_NUM;

    int my_blocks = blocks;
    int blocks_shift = 0;
    while (my_blocks != 0) {
        blocks_shift++;
        my_blocks >>= 1;
    }

    __mdvw sum1_w = _mve_set1_w(0);
    __mdvw sum2_w = _mve_set1_w(0);

    _mve_set_load_stride(0, blocks);
    __vidx_var stride = {3, 0, 0, 0};
    while (blocks > 0) {

        __mdvb buff_b = _mve_load_b(buf, stride);
        __mdvw buff_w = _mve_cvtu_btow(buff_b);
        // free buff_b
        _mve_free_b();

        sum1_w = _mve_add_w(sum1_w, buff_w);
        // free sum1_w and buff_w
        _mve_free_w();
        _mve_free_w();
        sum2_w = _mve_add_w(sum1_w, sum2_w);
        // free sum2_w
        _mve_free_w();

        blocks--;
        buf++;
    }

    // DIM0: Writing sequentially
    __vidx_var write_stride = {1, 0, 0, 0};
    // DIM0: Reading every other element
    _mve_set_load_stride(0, 2);
    __vidx_var read_stride = {3, 0, 0, 0};

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_w(sum2_mem_w, sum2_w, write_stride);
    // free sum2_w
    _mve_free_w();

    // STEP1: 4K output, shift: 2^3, sum1 12 bits, sum2 15 bits

    _mve_set_dim_length(0, BLOCK_4K);

    // 11 bits
    __mdvw sum1_buff1_w = _mve_load_w(sum1_mem_w, read_stride);
    __mdvw sum1_buff2_w = _mve_load_w(sum1_mem_w + 1, read_stride);
    // 12 bits
    sum1_w = _mve_add_w(sum1_buff1_w, sum1_buff2_w);
    // free sum1_buff2_w
    _mve_free_w();

    // 13 bits
    __mdvw sum2_buff1_w = _mve_load_w(sum2_mem_w, read_stride);
    // 13 bits
    __mdvw sum2_buff2_w = _mve_load_w(sum2_mem_w + 1, read_stride);
    // 14 bits
    sum1_buff1_w = _mve_shil_w(sum1_buff1_w, blocks_shift);
    // free sum1_buff1_w
    _mve_free_w();
    blocks_shift++;
    // 14 bits
    sum2_w = _mve_add_w(sum2_buff1_w, sum2_buff2_w);
    // free sum2_buff1_w and sum2_buff2_w
    _mve_free_w();
    _mve_free_w();
    // 15 bits
    sum2_w = _mve_add_w(sum2_w, sum1_buff1_w);
    // free sum1_buff1_w and sum2_w
    _mve_free_w();
    _mve_free_w();

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_w(sum2_mem_w, sum2_w, write_stride);
    // free sum2_w
    _mve_free_w();

    // STEP2: 2K output, shift: 2^4, sum1 13 bits, sum2 17 bits

    _mve_set_dim_length(0, BLOCK_2K);

    // 12 bits
    sum1_buff1_w = _mve_load_w(sum1_mem_w, read_stride);
    sum1_buff2_w = _mve_load_w(sum1_mem_w + 1, read_stride);
    // 13 bits
    sum1_w = _mve_add_w(sum1_buff1_w, sum1_buff2_w);
    // free sum1_buff2_w
    _mve_free_w();

    // 15 bits
    sum2_buff1_w = _mve_load_w(sum2_mem_w, read_stride);
    // 15 bits
    sum2_buff2_w = _mve_load_w(sum2_mem_w + 1, read_stride);
    // 16 bits
    sum1_buff1_w = _mve_shil_w(sum1_buff1_w, blocks_shift);
    // free sum1_buff1_w
    _mve_free_w();
    __mdvdw sum1_buff1_dw = _mve_cvtu_wtodw(sum1_buff1_w);
    // free sum1_buff1_w
    _mve_free_w();
    blocks_shift++;
    // 16 bits
    sum2_w = _mve_add_w(sum2_buff1_w, sum2_buff2_w);
    // free sum2_buff1_w and sum2_buff2_w
    _mve_free_w();
    _mve_free_w();
    __mdvdw sum2_dw = _mve_cvtu_wtodw(sum2_w);
    // free sum2_w
    _mve_free_w();
    // 17 bits
    sum2_dw = _mve_add_dw(sum2_dw, sum1_buff1_dw);
    // free sum2_dw and sum1_buff1_dw
    _mve_free_dw();
    _mve_free_dw();

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
    // free sum2_dw
    _mve_free_dw();

    // STEP3: 1K output, shift: 2^5, sum1 14 bits, sum2 19 bits

    _mve_set_dim_length(0, BLOCK_1K);

    // 13 bits
    sum1_buff1_w = _mve_load_w(sum1_mem_w, read_stride);
    sum1_buff2_w = _mve_load_w(sum1_mem_w + 1, read_stride);
    // 14 bits
    sum1_w = _mve_add_w(sum1_buff1_w, sum1_buff2_w);
    // free sum1_buff2_w
    _mve_free_w();

    // 17 bits
    __mdvdw sum2_buff1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
    // 17 bits
    __mdvdw sum2_buff2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);
    // 13 bits
    sum1_buff1_dw = _mve_cvtu_wtodw(sum1_buff1_w);
    // free sum1_buff1_w
    _mve_free_w();
    // 18 bits
    sum1_buff1_dw = _mve_shil_dw(sum1_buff1_dw, blocks_shift);
    // free sum1_buff1_dw
    _mve_free_dw();
    blocks_shift++;
    // 18 bits
    sum2_dw = _mve_add_dw(sum2_buff1_dw, sum2_buff2_dw);
    // free sum2_buff1_dw and sum2_buff2_dw
    _mve_free_dw();
    _mve_free_dw();
    // 19 bits
    sum2_dw = _mve_add_dw(sum2_dw, sum1_buff1_dw);
    // free sum2_dw and sum1_buff1_dw
    _mve_free_dw();
    _mve_free_dw();

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
    // free sum2_dw
    _mve_free_dw();

    // STEP4: 512 output, shift: 2^6, sum1 15 bits, sum2 21 bits

    _mve_set_dim_length(0, BLOCK_512);

    // 14 bits
    sum1_buff1_w = _mve_load_w(sum1_mem_w, read_stride);
    sum1_buff2_w = _mve_load_w(sum1_mem_w + 1, read_stride);
    // 15 bits
    sum1_w = _mve_add_w(sum1_buff1_w, sum1_buff2_w);
    // free sum1_buff2_w
    _mve_free_w();

    // 19 bits
    sum2_buff1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
    // 19 bits
    sum2_buff2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);
    // 14 bits
    sum1_buff1_dw = _mve_cvtu_wtodw(sum1_buff1_w);
    // free sum1_buff1_w
    _mve_free_w();
    // 20 bits
    sum1_buff1_dw = _mve_shil_dw(sum1_buff1_dw, blocks_shift);
    // free sum1_buff1_dw
    _mve_free_dw();
    blocks_shift++;
    // 20 bits
    sum2_dw = _mve_add_dw(sum2_buff1_dw, sum2_buff2_dw);
    // free sum2_buff1_dw and sum2_buff2_dw
    _mve_free_dw();
    _mve_free_dw();
    // 21 bits
    sum2_dw = _mve_add_dw(sum2_dw, sum1_buff1_dw);
    // free sum2_dw and sum1_buff1_dw
    _mve_free_dw();
    _mve_free_dw();

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
    // free sum2_dw
    _mve_free_dw();

    // STEP5: 256 output, shift: 2^7, sum1 16 bits, sum2 23 bits

    _mve_set_dim_length(0, BLOCK_256);

    // 15 bits
    sum1_buff1_w = _mve_load_w(sum1_mem_w, read_stride);
    sum1_buff2_w = _mve_load_w(sum1_mem_w + 1, read_stride);
    // 16 bits
    sum1_w = _mve_add_w(sum1_buff1_w, sum1_buff2_w);
    // free sum1_buff2_w
    _mve_free_w();

    // 21 bits
    sum2_buff1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
    // 21 bits
    sum2_buff2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);
    // 15 bits
    sum1_buff1_dw = _mve_cvtu_wtodw(sum1_buff1_w);
    // free sum1_buff1_w
    _mve_free_w();
    // 22 bits
    sum1_buff1_dw = _mve_shil_dw(sum1_buff1_dw, blocks_shift);
    // free sum1_buff1_dw
    _mve_free_dw();
    blocks_shift++;
    // 22 bits
    sum2_dw = _mve_add_dw(sum2_buff1_dw, sum2_buff2_dw);
    // free sum2_buff1_dw and sum2_buff2_dw
    _mve_free_dw();
    _mve_free_dw();
    // 23 bits
    sum2_dw = _mve_add_dw(sum2_dw, sum1_buff1_dw);
    // free sum2_dw and sum1_buff1_dw
    _mve_free_dw();
    _mve_free_dw();

    _mve_store_w(sum1_mem_w, sum1_w, write_stride);
    // free sum1_w
    _mve_free_w();
    _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
    // free sum2_dw
    _mve_free_dw();

    // STEP6: the rest, shift: 2^8
#pragma unroll(4)
    for (int i = 0; i < 256; i++) {
        uint16_t sum_1 = sum1_mem_w[i];
        uint32_t sum_2 = sum2_mem_dw[i];
        total_sum2 = (total_sum2 + (total_sum1 << shift_val) + sum_2) % BASE;
        total_sum1 = (total_sum1 + sum_1) % BASE;
    }

    adler32_output->return_value[0] = total_sum1 | (total_sum2 << 16);
}

#define STORE_ALL_BACK_BACK(MEM_1, MEM_2)     \
    _mve_store_w(sum1_mem_w, sum1_w, stride); \
    _mve_store_dw(sum2_mem_dw, sum2_dw, stride);

#define STORE_HALF_BACK_BACK                     \
    _mve_unset_only_element(0, 1);               \
    STORE_ALL_BACK_BACK(sum1_mem_w, sum2_mem_dw) \
    _mve_set_all_elements(0);

#define ADLER_ROUND_BACK_BACK(BLOCK_SIZE)                 \
    _mve_set_dim_length(1, BLOCK_SIZE);                   \
                                                          \
    sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, stride);    \
    sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + 1, stride); \
                                                          \
    sum1_w = _mve_add_w(sum1_w, sum1_buf2_w);             \
    /* free sum1_w and sum1_buf2_w */                     \
    _mve_free_w();                                        \
    _mve_free_w();                                        \
    sum2_dw = _mve_add_dw(sum2_dw, sum2_buf2_dw);         \
    /* free sum2_dw and sum2_buf2_dw */                   \
    _mve_free_dw();                                       \
    _mve_free_dw();

#define CPU_REDUCE_BACK_BACK(BLOCK_SIZE)                                   \
    for (i = 0, j = 0; j < BLOCK_SIZE; i += 1, j += 2) {                   \
        final_sum1[i] = (final_sum1[j] + final_sum1[j + 1]) % BASE;        \
        final_sum2[i] = ((final_sum2[j] << 1) + final_sum2[j + 1]) % BASE; \
    }

void adler32_back_back_back_mve(int LANE_NUM,
                                config_t *config,
                                input_t *input,
                                output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    int i = 0;
    int j = 0;

    // DIM0: 2
    // DIM1: 4K
    _mve_set_dim_count(2);
    _mve_set_dim_length(0, 2);
    _mve_set_dim_length(1, BLOCK_4K);

    // DIM0: Reading 2 adjacent elements
    // DIM1: Reading next 2 adjacent elements
    _mve_set_load_stride(1, 2);
    __vidx_var stride = {1, 2, 0, 0};

    __mdvw coeff_w = _mve_load_w((const __int16_t *)mve_adler_coeff, stride);

    int32_t sum2_mem_dw[BLOCK_8K];
    int16_t sum1_mem_w[BLOCK_8K];

    int32_t final_sum2[MIN_BLOCK];
    int16_t final_sum1[MIN_BLOCK];

    uint32_t total_sum1 = (adler >> 16) & 0xffff;
    uint32_t total_sum2 = adler & 0xffff;

    __mdvdw sum2_buf2_dw;
    __mdvb sum1_buf1_b;
    __mdvw sum1_buf1_w;
    __mdvdw sum1_buf1_dw;
    __mdvb sum1_buf2_b;
    __mdvw sum1_buf2_w;
    __mdvdw sum1_buf2_dw;
    __mdvw sum1_w;
    __mdvw sum2_w;
    __mdvdw sum2_dw;

    bool first_round = true;

    while (len > 0) {
        _mve_set_dim_length(1, BLOCK_4K);

        // iteration 1: 8192 additions

        sum1_buf1_b = _mve_load_b(buf, stride);
        sum1_buf1_w = _mve_cvtu_btow(sum1_buf1_b);
        // free sum1_buf1_b
        _mve_free_b();

        sum1_buf2_b = _mve_load_b(buf + 1, stride);
        sum1_buf2_w = _mve_cvtu_btow(sum1_buf2_b);
        // free sum1_buf2_b
        _mve_free_b();

        // 9 bits
        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);

        // log2(8K/MIN_BLOCK) + 8 = 15 (64), 14 (128), 13 (256)
        sum1_buf1_w = _mve_mul_w(sum1_buf1_w, coeff_w);
        // free sum1_buf1_w
        _mve_free_w();
        // log2(8K/MIN_BLOCK) + 8 = 15 (64), 14 (128), 13 (256)
        sum1_buf2_w = _mve_mul_w(sum1_buf2_w, coeff_w);
        // free sum1_buf2_w
        _mve_free_w();
        // 16 (64), 15 (128), 14 (256)
        sum1_buf2_w = _mve_shil_w(sum1_buf2_w, 1);
        // free sum1_buf2_w
        _mve_free_w();

        // 17 (64, ERROR), 16 (128), 15 (256)
        sum2_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w and sum1_buf1_w
        _mve_free_w();
        _mve_free_w();

        sum2_dw = _mve_cvtu_wtodw(sum2_w);
        // free sum2_w
        _mve_free_w();

        if (!first_round) {
#if MIN_BLOCK == BLOCK_256
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_256)
#endif
#if MIN_BLOCK >= BLOCK_128
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_128)
#endif
#if MIN_BLOCK >= BLOCK_64
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_64)
#endif
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_32)
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_16)
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_8)
#pragma unroll(BLOCK_4)
            CPU_REDUCE_BACK_BACK(BLOCK_4)
#pragma unroll(BLOCK_2)
            CPU_REDUCE_BACK_BACK(BLOCK_2)
#pragma unroll(BLOCK_1)
            CPU_REDUCE_BACK_BACK(BLOCK_1)
            total_sum2 = ((total_sum1 << 13) + total_sum2 + final_sum2[0]) % BASE;
            total_sum1 = (total_sum1 + final_sum1[0]) % BASE;
        } else {
            first_round = false;
        }

        STORE_HALF_BACK_BACK

        // iteration 2: 4096 additions (sum1_w 10 bits, sum2_dw 18/17/16 bits)
#if MIN_BLOCK <= BLOCK_4K
        ADLER_ROUND_BACK_BACK(BLOCK_2K)
#endif
#if MIN_BLOCK < BLOCK_4K
        STORE_HALF_BACK_BACK
#endif

        // iteration 3: 2048 additions (sum1_w 11 bits, sum2_dw 19/18/17 bits)
#if MIN_BLOCK <= BLOCK_2K
        ADLER_ROUND_BACK_BACK(BLOCK_1K)
#endif
#if MIN_BLOCK < BLOCK_2K
        STORE_HALF_BACK_BACK
#endif

        // iteration 4: 1024 additions (sum1_w 12 bits, sum2_dw 20/19/18 bits)
#if MIN_BLOCK <= BLOCK_1K
        ADLER_ROUND_BACK_BACK(BLOCK_512)
#endif
#if MIN_BLOCK < BLOCK_1K
        STORE_HALF_BACK_BACK
#endif

        // iteration 5: 512 additions (sum1_w 13 bits, sum2_dw 21/20/19 bits)
#if MIN_BLOCK <= BLOCK_512
        ADLER_ROUND_BACK_BACK(BLOCK_256)
#endif
#if MIN_BLOCK < BLOCK_512
        STORE_HALF_BACK_BACK
#endif

        // iteration 6: 256 additions (sum1_w 14 bits, sum2_dw 22/21/20 bits)
#if MIN_BLOCK <= BLOCK_256
        ADLER_ROUND_BACK_BACK(BLOCK_128)
#endif
#if MIN_BLOCK < BLOCK_256
        STORE_HALF_BACK_BACK
#endif

        // iteration 7: 128 additions (sum1_w 15 bits, sum2_dw 23/22/21 bits)
#if MIN_BLOCK <= BLOCK_128
        ADLER_ROUND_BACK_BACK(BLOCK_64)
#endif
#if MIN_BLOCK < BLOCK_128
        STORE_HALF_BACK_BACK
#endif

        // iteration 8: 64 additions (sum1_w 16 bits, sum2_dw 30 bits)
#if MIN_BLOCK <= BLOCK_64
        ADLER_ROUND_BACK_BACK(BLOCK_32)
#endif
#if MIN_BLOCK < BLOCK_64
        assert(false);
#endif

        STORE_ALL_BACK_BACK(final_sum1, final_sum2)

        // free sum1_w and sum2_dw
        _mve_free_w();
        _mve_free_dw();

        buf += BLOCK_16K;
        len -= BLOCK_16K;
    }

#if MIN_BLOCK == BLOCK_256
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_256)
#endif
#if MIN_BLOCK >= BLOCK_128
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_128)
#endif
#if MIN_BLOCK >= BLOCK_64
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_64)
#endif
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_32)
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_16)
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_8)
#pragma unroll(BLOCK_4)
    CPU_REDUCE_BACK_BACK(BLOCK_4)
#pragma unroll(BLOCK_2)
    CPU_REDUCE_BACK_BACK(BLOCK_2)
#pragma unroll(BLOCK_1)
    CPU_REDUCE_BACK_BACK(BLOCK_1)
    total_sum2 = ((total_sum1 << 13) + total_sum2 + final_sum2[0]) % BASE;
    total_sum1 = (total_sum1 + final_sum1[0]) % BASE;

    // free coeff_w
    _mve_free_w();

    adler32_output->return_value[0] = total_sum1 | (total_sum2 << 16);
}

#define STORE_ALL_BACK(MEM_1, MEM_2)          \
    _mve_store_w(sum1_mem_w, sum1_w, stride); \
    _mve_store_dw(sum2_mem_dw, sum2_dw, stride);

#define STORE_HALF_BACK                     \
    _mve_unset_only_element(1, 1);          \
    STORE_ALL_BACK(sum1_mem_w, sum2_mem_dw) \
    _mve_set_all_elements(1);

#define ADLER_ROUND_BACK(BLOCK_SIZE)                               \
    _mve_set_dim_length(0, BLOCK_SIZE);                            \
                                                                   \
    sum1_buf2_w = _mve_load_w(sum1_mem_w + BLOCK_SIZE, stride);    \
    sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + BLOCK_SIZE, stride); \
                                                                   \
    sum1_w = _mve_add_w(sum1_w, sum1_buf2_w);                      \
    /* free sum1_w and sum1_buf2_w */                              \
    _mve_free_w();                                                 \
    _mve_free_w();                                                 \
    sum2_dw = _mve_add_dw(sum2_dw, sum2_buf2_dw);                  \
    /* free sum2_dw and sum2_buf2_dw */                            \
    _mve_free_dw();                                                \
    _mve_free_dw();

#define CPU_REDUCE_BACK                                   \
    for (int i = 0; i < MIN_BLOCK; i++) {                 \
        total_sum1 = (total_sum1 + final_sum1[i]) % BASE; \
        total_sum2 = (total_sum2 + final_sum2[i]) % BASE; \
    }

void adler32_back_back_mve(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    // DIM0: 4K
    // DIM1: 2
    _mve_set_dim_count(2);
    _mve_set_dim_length(0, BLOCK_4K);
    _mve_set_dim_length(1, 2);

    // Read every other element
    __vidx_var stride = {2, 2, 0, 0};

    __mdvdw coeff_dw = _mve_load_dw((const __int32_t *)mve_adler_coeff, stride);

    int32_t sum2_mem_dw[BLOCK_8K];
    int16_t sum1_mem_w[BLOCK_8K];

    int32_t final_sum2[MIN_BLOCK];
    int16_t final_sum1[MIN_BLOCK];

    uint32_t total_sum1 = (adler >> 16) & 0xffff;
    uint32_t total_sum2 = adler & 0xffff;

    __mdvdw sum2_buf2_dw;
    __mdvb sum1_buf1_b;
    __mdvw sum1_buf1_w;
    __mdvdw sum1_buf1_dw;
    __mdvb sum1_buf2_b;
    __mdvw sum1_buf2_w;
    __mdvdw sum1_buf2_dw;
    __mdvw sum1_w;
    __mdvdw sum2_dw;

    bool first_round = true;

    while (len > 0) {
        _mve_set_dim_length(0, BLOCK_4K);

        // iteration 1: 8192 additions

        sum1_buf1_b = _mve_load_b(buf, stride);
        sum1_buf1_w = _mve_cvtu_btow(sum1_buf1_b);
        sum1_buf1_dw = _mve_cvtu_btodw(sum1_buf1_b);
        // free sum1_buf1_b
        _mve_free_b();

        sum1_buf2_b = _mve_load_b(buf + BLOCK_8K, stride);
        sum1_buf2_w = _mve_cvtu_btow(sum1_buf2_b);
        sum1_buf2_dw = _mve_cvtu_btodw(sum1_buf2_b);
        // free sum1_buf2_b
        _mve_free_b();

        // 9 bits
        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf1_w and sum1_buf2_w
        _mve_free_w();
        _mve_free_w();

        // 21 bits
        sum1_buf1_dw = _mve_mul_dw(sum1_buf1_dw, coeff_dw);
        // free sum1_buf1_dw
        _mve_free_dw();
        // 21 bits
        sum1_buf2_dw = _mve_mul_dw(sum1_buf2_dw, coeff_dw);
        // free sum1_buf2_dw
        _mve_free_dw();
        // 22 bits
        sum1_buf2_dw = _mve_shil_dw(sum1_buf2_dw, 1);
        // free sum1_buf2_dw
        _mve_free_dw();

        // 23 bits
        sum2_dw = _mve_add_dw(sum1_buf1_dw, sum1_buf2_dw);
        // free sum1_buf2_dw and sum1_buf1_dw
        _mve_free_dw();
        _mve_free_dw();

        if (!first_round) {
#pragma unroll(4)
            CPU_REDUCE_BACK
        } else {
            first_round = false;
        }

        STORE_HALF_BACK

        // iteration 2: 4096 additions (sum1_w 10 bits, sum2_dw 24 bits)
#if MIN_BLOCK <= BLOCK_4K
        ADLER_ROUND_BACK(BLOCK_2K)
#endif
#if MIN_BLOCK < BLOCK_4K
        STORE_HALF_BACK
#endif

        // iteration 3: 2048 additions (sum1_w 11 bits, sum2_dw 25 bits)
#if MIN_BLOCK <= BLOCK_2K
        ADLER_ROUND_BACK(BLOCK_1K)
#endif
#if MIN_BLOCK < BLOCK_2K
        STORE_HALF_BACK
#endif

        // iteration 4: 1024 additions (sum1_w 12 bits, sum2_dw 26 bits)
#if MIN_BLOCK <= BLOCK_1K
        ADLER_ROUND_BACK(BLOCK_512)
#endif
#if MIN_BLOCK < BLOCK_1K
        STORE_HALF_BACK
#endif

        // iteration 5: 512 additions (sum1_w 13 bits, sum2_dw 27 bits)
#if MIN_BLOCK <= BLOCK_512
        ADLER_ROUND_BACK(BLOCK_256)
#endif
#if MIN_BLOCK < BLOCK_512
        STORE_HALF_BACK
#endif

        // iteration 6: 256 additions (sum1_w 14 bits, sum2_dw 28 bits)
#if MIN_BLOCK <= BLOCK_256
        ADLER_ROUND_BACK(BLOCK_128)
#endif
#if MIN_BLOCK < BLOCK_256
        STORE_HALF_BACK
#endif

        // iteration 7: 128 additions (sum1_w 15 bits, sum2_dw 29 bits)
#if MIN_BLOCK <= BLOCK_128
        ADLER_ROUND_BACK(BLOCK_64)
#endif
#if MIN_BLOCK < BLOCK_128
        STORE_HALF_BACK
#endif

        // iteration 8: 64 additions (sum1_w 16 bits, sum2_dw 30 bits)
#if MIN_BLOCK <= BLOCK_64
        ADLER_ROUND_BACK(BLOCK_32)
#endif
#if MIN_BLOCK < BLOCK_64
        assert(false);
#endif

        STORE_ALL_BACK(final_sum1, final_sum2)

        // free sum1_w and sum2_dw
        _mve_free_w();
        _mve_free_dw();

        buf += BLOCK_16K;
        len -= BLOCK_16K;
    }

#pragma unroll(4)
    CPU_REDUCE_BACK

    // free coeff_dw
    _mve_free_dw();

    adler32_output->return_value[0] = total_sum1 | (total_sum2 << 16);
}

void adler32_back_mve(int LANE_NUM,
                      config_t *config,
                      input_t *input,
                      output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    // adler32_output_t *adler32_output = (adler32_output_t *)output;

    // uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    _mve_set_dim_count(1);
    _mve_set_dim_length(0, BLOCK_8K);

    // Read every other element
    _mve_set_load_stride(0, 2);
    __vidx_var read_stride = {3, 0, 0, 0};

    // Write sequentially
    __vidx_var write_stride = {1, 0, 0, 0};

    int64_t sum1_mem_qw[BLOCK_8K];
    int64_t sum2_mem_qw[BLOCK_8K];

    // int32_t *sum1_mem_dw = (int32_t *)sum1_mem_qw;
    int32_t *sum2_mem_dw = (int32_t *)sum2_mem_qw;

    int16_t *sum1_mem_w = (int16_t *)sum1_mem_qw;
    int16_t *sum2_mem_w = (int16_t *)sum2_mem_qw;

    // uint32_t total_sum1 = (adler >> 16) & 0xffff;
    // uint32_t total_sum2 = adler & 0xffff;

    while (len > 0) {
        _mve_set_dim_length(0, BLOCK_8K);

        // iteration 1: 8192 additions (sum1 9bits) (sum2 10bits)

        __mdvb sum1_buf1_b = _mve_load_b(buf, read_stride);
        __mdvw sum1_buf1_w = _mve_cvtu_btow(sum1_buf1_b);
        // free sum1_buf1_b
        _mve_free_b();

        __mdvb sum1_buf2_b = _mve_load_b(buf + 1, read_stride);
        __mdvw sum1_buf2_w = _mve_cvtu_btow(sum1_buf2_b);
        // free sum1_buf2_b
        _mve_free_b();

        __mdvw sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);

        __mdvw sum2_w = _mve_add_w(sum1_w, sum1_buf1_w);
        // free sum1_buf1_w and sum1_w
        _mve_free_w();
        _mve_free_w();
        _mve_store_w(sum2_mem_w, sum2_w, write_stride);
        // free sum2_w
        _mve_free_w();

        // iteration 2: 4096 additions (sum1 10bits) (sum2 12bits)

        _mve_set_dim_length(0, BLOCK_4K);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        __mdvw sum2_buf1_w = _mve_load_w(sum2_mem_w, read_stride);
        __mdvw sum2_buf2_w = _mve_load_w(sum2_mem_w + 1, read_stride);

        sum2_w = _mve_add_w(sum2_buf1_w, sum2_buf2_w);
        // free sum2_buf1_w and sum2_buf2_w
        _mve_free_w();
        _mve_free_w();
        __mdvw sum1_buf1_sh_w = _mve_shil_w(sum1_buf1_w, 1);
        // free sum1_buf1_w
        _mve_free_w();
        sum2_w = _mve_add_w(sum2_w, sum1_buf1_sh_w);
        // free sum2_w and sum1_buf1_sh_w
        _mve_free_w();
        _mve_free_w();
        _mve_store_w(sum2_mem_w, sum2_w, write_stride);
        // free sum2_w
        _mve_free_w();

        // iteration 3: 2048 additions (sum1 11bits) (sum2 14bits)

        _mve_set_dim_length(0, BLOCK_2K);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_w = _mve_load_w(sum2_mem_w, read_stride);
        sum2_buf2_w = _mve_load_w(sum2_mem_w + 1, read_stride);

        sum2_w = _mve_add_w(sum2_buf1_w, sum2_buf2_w);
        // free sum2_buf1_w and sum2_buf2_w
        _mve_free_w();
        _mve_free_w();
        sum1_buf1_sh_w = _mve_shil_w(sum1_buf1_w, 2);
        // free sum1_buf1_w
        _mve_free_w();
        sum2_w = _mve_add_w(sum2_w, sum1_buf1_sh_w);
        // free sum2_w and sum1_buf1_sh_w
        _mve_free_w();
        _mve_free_w();
        _mve_store_w(sum2_mem_w, sum2_w, write_stride);
        // free sum2_w
        _mve_free_w();

        // iteration 4: 1024 additions (sum1 12bits) (sum2 16bits)

        _mve_set_dim_length(0, BLOCK_1K);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_w = _mve_load_w(sum2_mem_w, read_stride);
        __mdvdw sum2_buf1_dw = _mve_cvtu_wtodw(sum2_buf1_w);
        // free sum2_buf1_w
        _mve_free_w();
        sum2_buf2_w = _mve_load_w(sum2_mem_w + 1, read_stride);
        __mdvdw sum2_buf2_dw = _mve_cvtu_wtodw(sum2_buf2_w);
        // free sum2_buf2_w
        _mve_free_w();

        __mdvdw sum2_dw = _mve_add_dw(sum2_buf1_dw, sum2_buf2_dw);
        // free sum2_buf1_dw and sum2_buf2_dw
        _mve_free_dw();
        _mve_free_dw();
        __mdvdw sum1_buf1_dw = _mve_cvtu_wtodw(sum1_buf1_w);
        // free sum1_buf1_w
        _mve_free_w();
        __mdvdw sum1_buf1_sh_dw = _mve_shil_dw(sum1_buf1_dw, 3);
        // free sum1_buf1_dw
        _mve_free_dw();
        sum2_dw = _mve_add_dw(sum2_dw, sum1_buf1_sh_dw);
        // free sum2_dw and sum1_buf1_sh_dw
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
        // free sum2_dw
        _mve_free_dw();

        // iteration 5: 512 additions (sum1 13bits) (sum2 18bits)

        _mve_set_dim_length(0, BLOCK_512);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
        sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);

        sum2_dw = _mve_add_dw(sum2_buf1_dw, sum2_buf2_dw);
        // free sum2_buf1_dw and sum2_buf2_dw
        _mve_free_dw();
        _mve_free_dw();
        sum1_buf1_dw = _mve_cvtu_wtodw(sum1_buf1_w);
        // free sum1_buf1_w
        _mve_free_w();
        sum1_buf1_sh_dw = _mve_shil_dw(sum1_buf1_dw, 4);
        // free sum1_buf1_dw
        _mve_free_dw();
        sum2_dw = _mve_add_dw(sum2_dw, sum1_buf1_sh_dw);
        // free sum2_dw and sum1_buf1_sh_dw
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
        // free sum2_dw
        _mve_free_dw();

        // iteration 6: 256 additions (sum1 14bits) (sum2 20bits)

        _mve_set_dim_length(0, BLOCK_256);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
        sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);

        sum2_dw = _mve_add_dw(sum2_buf1_dw, sum2_buf2_dw);
        // free sum2_buf1_dw and sum2_buf2_dw
        _mve_free_dw();
        _mve_free_dw();
        sum1_buf1_dw = _mve_cvtu_wtodw(sum1_buf1_w);
        // free sum1_buf1_w
        _mve_free_w();
        sum1_buf1_sh_dw = _mve_shil_dw(sum1_buf1_dw, 5);
        // free sum1_buf1_dw
        _mve_free_dw();
        sum2_dw = _mve_add_dw(sum2_dw, sum1_buf1_sh_dw);
        // free sum2_dw and sum1_buf1_sh_dw
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
        // free sum2_dw
        _mve_free_dw();

        // iteration 7: 128 additions (sum1 15bits) (sum2 22bits)

        _mve_set_dim_length(0, BLOCK_256);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
        sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);

        sum2_dw = _mve_add_dw(sum2_buf1_dw, sum2_buf2_dw);
        // free sum2_buf1_dw and sum2_buf2_dw
        _mve_free_dw();
        _mve_free_dw();
        sum1_buf1_dw = _mve_cvtu_wtodw(sum1_buf1_w);
        // free sum1_buf1_w
        _mve_free_w();
        sum1_buf1_sh_dw = _mve_shil_dw(sum1_buf1_dw, 6);
        // free sum1_buf1_dw
        _mve_free_dw();
        sum2_dw = _mve_add_dw(sum2_dw, sum1_buf1_sh_dw);
        // free sum2_dw and sum1_buf1_sh_dw
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
        // free sum2_dw
        _mve_free_dw();

        // iteration 8: 64 additions (sum1 16bits) (sum2 24bits)

        _mve_set_dim_length(0, BLOCK_64);

        sum1_buf1_w = _mve_load_w(sum1_mem_w, read_stride);
        sum1_buf2_w = _mve_load_w(sum1_mem_w + 1, read_stride);

        sum1_w = _mve_add_w(sum1_buf1_w, sum1_buf2_w);
        // free sum1_buf2_w
        _mve_free_w();
        _mve_store_w(sum1_mem_w, sum1_w, write_stride);
        // free sum1_w
        _mve_free_w();

        sum2_buf1_dw = _mve_load_dw(sum2_mem_dw, read_stride);
        sum2_buf2_dw = _mve_load_dw(sum2_mem_dw + 1, read_stride);

        sum2_dw = _mve_add_dw(sum2_buf1_dw, sum2_buf2_dw);
        // free sum2_buf1_dw and sum2_buf2_dw
        _mve_free_dw();
        _mve_free_dw();
        sum1_buf1_dw = _mve_cvtu_wtodw(sum1_buf1_w);
        // free sum1_buf1_w
        _mve_free_w();
        sum1_buf1_sh_dw = _mve_shil_dw(sum1_buf1_dw, 7);
        // free sum1_buf1_dw
        _mve_free_dw();
        sum2_dw = _mve_add_dw(sum2_dw, sum1_buf1_sh_dw);
        // free sum2_dw and sum1_buf1_sh_dw
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw(sum2_mem_dw, sum2_dw, write_stride);
        // free sum2_dw
        _mve_free_dw();

        //         // iteration 9: 32 additions (sum1 17bits) (sum2 26bits)
        //         int idx = 0;
        // #pragma unroll(32)
        //         for (int j = 0; j < 32; j++, idx += 2) {
        //             uint16_t first_sum_1 = sum1_mem_w[idx];
        //             uint16_t second_sum_1 = sum1_mem_w[idx + 1];
        //             sum1_mem_dw[j] = first_sum_1 + second_sum_1;
        //             uint32_t first_sum_2 = sum2_mem_dw[idx];
        //             uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
        //             uint32_t sum2 = first_sum_2 + second_sum_2;
        //             sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 8);
        //         }

        //         // iteration 10: 16 additions (sum1 18bits) (sum2 28bits)
        //         idx = 0;
        // #pragma unroll(16)
        //         for (int j = 0; j < 16; j++, idx += 2) {
        //             uint32_t first_sum_1 = sum1_mem_dw[idx];
        //             uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
        //             sum1_mem_dw[j] = first_sum_1 + second_sum_1;
        //             uint32_t first_sum_2 = sum2_mem_dw[idx];
        //             uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
        //             uint32_t sum2 = first_sum_2 + second_sum_2;
        //             sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 9);
        //         }

        //         // iteration 11: 8 additions (sum1 19bits) (sum2 30bits)
        //         idx = 0;
        // #pragma unroll(8)
        //         for (int j = 0; j < 8; j++, idx += 2) {
        //             uint32_t first_sum_1 = sum1_mem_dw[idx];
        //             uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
        //             sum1_mem_dw[j] = first_sum_1 + second_sum_1;
        //             uint32_t first_sum_2 = sum2_mem_dw[idx];
        //             uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
        //             uint32_t sum2 = first_sum_2 + second_sum_2;
        //             sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 10);
        //         }

        //         // iteration 12: 4 additions
        //         idx = 0;
        // #pragma unroll(4)
        //         for (int j = 0; j < 4; j++, idx += 2) {
        //             uint32_t first_sum_1 = sum1_mem_dw[idx];
        //             uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
        //             sum1_mem_dw[j] = (first_sum_1 + second_sum_1) % BASE;
        //             uint32_t first_sum_2 = sum2_mem_dw[idx];
        //             uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
        //             uint32_t sum2 = first_sum_2 + second_sum_2;
        //             sum2_mem_dw[j] = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 11)) % BASE;
        //         }

        //         // iteration 12: 2 additions
        //         idx = 0;
        // #pragma unroll(2)
        //         for (int j = 0; j < 2; j++, idx += 2) {
        //             uint32_t first_sum_1 = sum1_mem_dw[idx];
        //             uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
        //             sum1_mem_dw[j] = (first_sum_1 + second_sum_1) % BASE;
        //             uint32_t first_sum_2 = sum2_mem_dw[idx];
        //             uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
        //             uint32_t sum2 = first_sum_2 + second_sum_2;
        //             sum2_mem_dw[j] = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 12)) % BASE;
        //         }

        //         // iteration 13: 1 addition
        //         uint32_t first_sum_1 = sum1_mem_dw[0];
        //         uint32_t second_sum_1 = sum1_mem_dw[1];
        //         uint32_t sum1 = (first_sum_1 + second_sum_1) % BASE;
        //         uint32_t first_sum_2 = sum2_mem_dw[0];
        //         uint32_t second_sum_2 = sum2_mem_dw[1];
        //         uint32_t sum2 = first_sum_2 + second_sum_2;
        //         sum2 = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 13)) % BASE;

        //         total_sum2 = ((uint64_t)total_sum2 + ((uint64_t)sum1 << 13) + (uint64_t)sum2) % BASE;
        //         total_sum1 = (total_sum1 + sum1) % BASE;

        buf += BLOCK_16K;
        len -= BLOCK_16K;
    }

    // adler32_output->return_value[0] = total_sum1 | (total_sum2 << 16);
}