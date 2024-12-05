#include "mve.hpp"
#include "mve_kernels.hpp"

#include "crc32.hpp"
#include "zlib.hpp"

#define STORE_HALF                         \
    _mve_set_only_element(1, 1);           \
    _mve_store_dw(memory, crc_dw, stride); \
    _mve_set_all_elements(1);

#define STORE_ALL                          \
    _mve_store_dw(memory, crc_dw, stride); \
    /* free crc_dw */                      \
    _mve_free_dw();

#define REDUCTION(BLOCK_SIZE)                                                   \
    _mve_set_dim_length(0, BLOCK_SIZE);                                         \
    crc_buff_dw = _mve_load_dw((const __int32_t *)memory + BLOCK_SIZE, stride); \
    crc_dw = _mve_xor_dw(crc_dw, crc_buff_dw);                                  \
    /* free crc_buff_dw and crc_dw */                                           \
    _mve_free_dw();                                                             \
    _mve_free_dw();

void crc32_mve(int LANE_NUM,
               config_t *config,
               input_t *input,
               output_t *output) {
    crc32_config_t *crc32_config = (crc32_config_t *)config;
    crc32_input_t *crc32_input = (crc32_input_t *)input;
    crc32_output_t *crc32_output = (crc32_output_t *)output;

    z_crc_t crc = crc32_config->crc;
    unsigned char *buf = crc32_input->buf;
    z_size_t len = crc32_config->len;

    _mve_set_dim_count(2);
    _mve_set_dim_length(0, BLOCK_4K);
    _mve_set_dim_length(1, 2);

    int num_blocks = len / (LANE_NUM * 8);

    __vidx_var stride = {1, 2, 0, 0};

    __mdvdw mve_crc_coeff_dw = _mve_load_dw((const int *)mve_crc_coeff, stride);

    int32_t memory[8192];

    __mdvdw crc_dw, crc_buff_dw;

    for (int block = 0; block < num_blocks; block++) {
        _mve_set_dim_length(0, BLOCK_4K);

        __mdvb buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        crc_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[0], buf_b);
        // free buf_b
        _mve_free_b();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        __mdvdw table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[1], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[2], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[3], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[4], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[5], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[6], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        buf_b = _mve_load_b(buf, stride);
        buf += BLOCK_8K;
        table_dw = _mve_dict_dw((const __int32_t *)crc_braid_table[7], buf_b);
        // free buf_b
        _mve_free_b();
        crc_dw = _mve_xor_dw(crc_dw, table_dw);
        // free crc_dw and table_dw
        _mve_free_dw();
        _mve_free_dw();

        crc_dw = _mve_mulmodp_dw(crc_dw, mve_crc_coeff_dw);
        // free crc_dw
        _mve_free_dw();

        STORE_HALF
        REDUCTION(BLOCK_2K)
        STORE_HALF
        REDUCTION(BLOCK_1K)
        STORE_HALF
        REDUCTION(BLOCK_512)
        STORE_HALF
        REDUCTION(BLOCK_256)
        STORE_HALF
        REDUCTION(BLOCK_128)
        STORE_HALF
        REDUCTION(BLOCK_64)
        STORE_HALF
        REDUCTION(BLOCK_32)
        STORE_ALL

#pragma unroll(4)
        for (int i = 0; i < BLOCK_64; i++) {
            crc ^= memory[i];
        }
    }

    // free mve_crc_coeff_dw
    _mve_free_dw();

    crc32_output->return_value[0] = crc;
}