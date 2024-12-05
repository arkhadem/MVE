#include "mve.hpp"
#include "mve_kernels.hpp"

#include "chacha20.hpp"

// QUARTERROUND updates x0, x1, x2, x3.
#define QUARTERROUND()          \
    x0 = _mve_add_dw(x0, x1);   \
    /* free x0 */               \
    _mve_free_dw();             \
    x3 = _mve_xor_dw(x0, x3);   \
    /* free x3 */               \
    _mve_free_dw();             \
    x3 = _mve_rotil_dw(x3, 16); \
    /* free x3 */               \
    _mve_free_dw();             \
    x2 = _mve_add_dw(x2, x3);   \
    /* free x2 */               \
    _mve_free_dw();             \
    x1 = _mve_xor_dw(x2, x1);   \
    /* free x1 */               \
    _mve_free_dw();             \
    x1 = _mve_rotil_dw(x1, 12); \
    /* free x1 */               \
    _mve_free_dw();             \
    x0 = _mve_add_dw(x0, x1);   \
    /* free x0 */               \
    _mve_free_dw();             \
    x3 = _mve_xor_dw(x0, x3);   \
    /* free x3 */               \
    _mve_free_dw();             \
    x3 = _mve_rotil_dw(x3, 8);  \
    /* free x3 */               \
    _mve_free_dw();             \
    x2 = _mve_add_dw(x2, x3);   \
    /* free x2 */               \
    _mve_free_dw();             \
    x1 = _mve_xor_dw(x2, x1);   \
    /* free x1 */               \
    _mve_free_dw();             \
    x1 = _mve_rotil_dw(x1, 7);  \
    /* free x1 */               \
    _mve_free_dw();

#define TRANSPOSE()                                                      \
    _mve_store_dw((__int32_t *)temp_mem, x1, seq_stride);                \
    /* free x1 */                                                        \
    _mve_free_dw();                                                      \
    x1 = _mve_loadr_dw((const __int32_t **)(temp_addr + 1), seq_stride); \
    _mve_store_dw((__int32_t *)temp_mem, x2, seq_stride);                \
    /* free x2 */                                                        \
    _mve_free_dw();                                                      \
    x2 = _mve_loadr_dw((const __int32_t **)(temp_addr + 2), seq_stride); \
    _mve_store_dw((__int32_t *)temp_mem, x3, seq_stride);                \
    /* free x3 */                                                        \
    _mve_free_dw();                                                      \
    x3 = _mve_loadr_dw((const __int32_t **)(temp_addr + 3), seq_stride);

#define DETRANSPOSE()                                                    \
    _mve_store_dw((__int32_t *)temp_mem, x1, seq_stride);                \
    /* free x1 */                                                        \
    _mve_free_dw();                                                      \
    x1 = _mve_loadr_dw((const __int32_t **)(temp_addr + 3), seq_stride); \
    _mve_store_dw((__int32_t *)temp_mem, x2, seq_stride);                \
    /* free x2 */                                                        \
    _mve_free_dw();                                                      \
    x2 = _mve_loadr_dw((const __int32_t **)(temp_addr + 2), seq_stride); \
    _mve_store_dw((__int32_t *)temp_mem, x3, seq_stride);                \
    /* free x3 */                                                        \
    _mve_free_dw();                                                      \
    x3 = _mve_loadr_dw((const __int32_t **)(temp_addr + 1), seq_stride);

#define TRANS_ROUND_DETRANS() \
    TRANSPOSE()               \
    QUARTERROUND()            \
    DETRANSPOSE()

void chacha20_mve(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    chacha20_config_t *chacha20_config = (chacha20_config_t *)config;
    chacha20_input_t *chacha20_input = (chacha20_input_t *)input;
    chacha20_output_t *chacha20_output = (chacha20_output_t *)output;

    uint8_t *out = chacha20_output->out;
    const uint8_t *in = chacha20_input->in;
    size_t in_len = chacha20_config->in_len;
    const uint8_t *key = chacha20_input->key;
    const uint8_t *nonce = chacha20_input->nonce;
    uint32_t counter = chacha20_config->counter;

    int num_blocks = (int)in_len >> 6;

    uint32_t input_buffer[16];

    uint32_t *my_sigma = (uint32_t *)sigma;
    input_buffer[0] = my_sigma[0];
    input_buffer[1] = my_sigma[1];
    input_buffer[2] = my_sigma[2];
    input_buffer[3] = my_sigma[3];

    uint32_t *my_key = (uint32_t *)key;
    input_buffer[4] = my_key[0];
    input_buffer[5] = my_key[1];
    input_buffer[6] = my_key[2];
    input_buffer[7] = my_key[3];

    input_buffer[8] = my_key[4];
    input_buffer[9] = my_key[5];
    input_buffer[10] = my_key[6];
    input_buffer[11] = my_key[7];

    uint32_t *my_nonce = (uint32_t *)nonce;
    input_buffer[12] = counter;
    input_buffer[13] = my_nonce[0];
    input_buffer[14] = my_nonce[1];
    input_buffer[15] = my_nonce[2];

    // Dim0: different blocks
    // Dim1: 4 cells
    _mve_set_dim_count(2);

    // when loading from input_buffer:
    // same element between different blocks
    // adjacent cells between different cells
    __vidx_var input_buffer_stride = {0, 1, 0, 0};

    __vidx_var block_counter_stride = {1, 0, 0, 0};

    // DIM0: load every other 16 elements from in
    // DIM1: load adjacent elements
    _mve_set_load_stride(0, 16);
    _mve_set_store_stride(0, 16);
    __vidx_var in_out_stride = {3, 1, 0, 0};

    int DIM_TILE = LANE_NUM >> 2;
    _mve_set_dim_length(0, DIM_TILE);
    _mve_set_dim_length(1, 4);

    __mdvdw block_counter_v;

    uint32_t temp_mem[8192];
    _mve_set_load_stride(1, DIM_TILE);
    _mve_set_store_stride(1, DIM_TILE);
    __vidx_var seq_stride = {1, 3, 0, 0};
    uint32_t *temp_addr[8];
    temp_addr[0] = temp_mem;
    temp_addr[1] = temp_addr[0] + DIM_TILE;
    temp_addr[2] = temp_addr[1] + DIM_TILE;
    temp_addr[3] = temp_addr[2] + DIM_TILE;
    temp_addr[4] = temp_mem;
    temp_addr[5] = temp_addr[0] + DIM_TILE;
    temp_addr[6] = temp_addr[1] + DIM_TILE;
    temp_addr[7] = temp_addr[2] + DIM_TILE;

    __mdvdw temp_var;

    block_counter_v = _mve_set1_dw(0);
    _mve_set_only_element(1, 0);
    block_counter_v = _mve_load_dw((const __int32_t *)block_counter, block_counter_stride);
    // free previous block_counter_v
    _mve_free_dw();
    _mve_set_all_elements(1);

    while (num_blocks > 0) {
        if (num_blocks != DIM_TILE) {
            _mve_set_dim_length(0, num_blocks);
            block_counter_v = _mve_set1_dw(0);
            // free previous block_counter_v
            _mve_free_dw();
            _mve_set_only_element(1, 0);
            block_counter_v = _mve_load_dw((const __int32_t *)block_counter, block_counter_stride);
            // free previous block_counter_v
            _mve_free_dw();
            _mve_set_all_elements(1);
        }

        __mdvdw x0 = _mve_load_dw((const __int32_t *)input_buffer, input_buffer_stride);
        __mdvdw x1 = _mve_load_dw((const __int32_t *)input_buffer + 4, input_buffer_stride);
        __mdvdw x2 = _mve_load_dw((const __int32_t *)input_buffer + 8, input_buffer_stride);
        __mdvdw x3 = _mve_load_dw((const __int32_t *)input_buffer + 12, input_buffer_stride);

        x3 = _mve_add_dw(x3, block_counter_v);
        // free prev x3
        _mve_free_dw();

#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            TRANS_ROUND_DETRANS()
        }

        temp_var = _mve_load_dw((const __int32_t *)input_buffer, input_buffer_stride);
        x0 = _mve_add_dw(x0, temp_var);
        // free x0 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        temp_var = _mve_load_dw(((const __int32_t *)in), in_out_stride);
        x0 = _mve_xor_dw(x0, temp_var);
        // free x0 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw((__int32_t *)out, x0, in_out_stride);
        // free x0
        _mve_free_dw();

        temp_var = _mve_load_dw((const __int32_t *)input_buffer + 4, input_buffer_stride);
        x1 = _mve_add_dw(x1, temp_var);
        // free x1 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        temp_var = _mve_load_dw(((const __int32_t *)in) + 4, in_out_stride);
        x1 = _mve_xor_dw(x1, temp_var);
        // free x1 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw((__int32_t *)out, x1, in_out_stride);
        // free x1
        _mve_free_dw();

        temp_var = _mve_load_dw((const __int32_t *)input_buffer + 8, input_buffer_stride);
        x2 = _mve_add_dw(x2, temp_var);
        // free x2 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        temp_var = _mve_load_dw(((const __int32_t *)in) + 8, in_out_stride);
        x2 = _mve_xor_dw(x2, temp_var);
        // free x2 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw((__int32_t *)out, x2, in_out_stride);
        // free x2
        _mve_free_dw();

        temp_var = _mve_load_dw((const __int32_t *)input_buffer + 12, input_buffer_stride);
        temp_var = _mve_add_dw(temp_var, block_counter_v);
        // free temp_var
        _mve_free_dw();
        x3 = _mve_add_dw(x3, temp_var);
        // free x3 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        temp_var = _mve_load_dw(((const __int32_t *)in) + 12, in_out_stride);
        x3 = _mve_xor_dw(x3, temp_var);
        // free x3 and temp_var
        _mve_free_dw();
        _mve_free_dw();
        _mve_store_dw((__int32_t *)out, x3, in_out_stride);
        // free x3
        _mve_free_dw();

        in += DIM_TILE << 6;
        out += DIM_TILE << 6;
        input_buffer[12] += DIM_TILE;
        num_blocks -= DIM_TILE;
    }

    // free block_counter_v
    _mve_free_dw();
}