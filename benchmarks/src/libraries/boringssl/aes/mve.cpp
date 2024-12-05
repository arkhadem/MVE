#include "mve.hpp"
#include "mve_kernels.hpp"

#include "aes.hpp"

void aes_mve(int LANE_NUM,
             config_t *config,
             input_t *input,
             output_t *output) {
    aes_config_t *aes_config = (aes_config_t *)config;
    aes_input_t *aes_input = (aes_input_t *)input;

    int num_blocks = aes_config->num_blocks;

    unsigned char *state = aes_input->state;
    unsigned char *RoundKey = aes_config->RoundKey;

    int round;

    _mve_set_dim_count(1);

    int count_per_iter = LANE_NUM;

    // Loading element wise
    _mve_set_load_stride(0, 16);

    // Storing column-wise
    _mve_set_store_stride(0, 16);

    // Load the same column for all cells of the row
    __vidx_var elementwise_stride = {3, 0, 0, 0};

    _mve_set_dim_length(0, count_per_iter);

    __mdvb sub_e_0;
    __mdvb sub_e_1;
    __mdvb sub_e_2;
    __mdvb sub_e_3;
    __mdvb sub_e_4;
    __mdvb sub_e_5;
    __mdvb sub_e_6;
    __mdvb sub_e_7;
    __mdvb sub_e_8;
    __mdvb sub_e_9;
    __mdvb sub_e_10;
    __mdvb sub_e_11;
    __mdvb sub_e_12;
    __mdvb sub_e_13;
    __mdvb sub_e_14;
    __mdvb sub_e_15;

    while (num_blocks > 0) {

        round = 0;

        _mve_set_dim_length(0, num_blocks > count_per_iter ? count_per_iter : num_blocks);
        num_blocks -= count_per_iter;

        __mdvb key, e;
        // using r0-15 for rounded_e s
        // using r16 and r17 for key e and key

        const unsigned char *RoundKey_addr = RoundKey;
        unsigned char *state_addr = state;

        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_0 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_1 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_2 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_3 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_4 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_5 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_6 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_7 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_8 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_9 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_10 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_11 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_12 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_13 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_14 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();
        e = _mve_load_b(state_addr, elementwise_stride);
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;
        state_addr += 1;
        __mdvb rounded_e_15 = _mve_xor_b(e, key);
        // free e (r16) and key (r17)
        _mve_free_b();
        _mve_free_b();

        for (round = 1;; ++round) {

            // [r1-16] <- [r0-15]

            // Sub Bytes
            // using r0-15 for new_rounded_e s
            // r0
            sub_e_0 = _mve_dict_b(sbox, rounded_e_0);
            // free rounded_e_0 (r1)
            _mve_free_b();
            // r1
            sub_e_1 = _mve_dict_b(sbox, rounded_e_1);
            // free rounded_e_1 (r2)
            _mve_free_b();
            // r2
            sub_e_2 = _mve_dict_b(sbox, rounded_e_2);
            // free rounded_e_2 (r3)
            _mve_free_b();
            // r3
            sub_e_3 = _mve_dict_b(sbox, rounded_e_3);
            // free rounded_e_3 (r4)
            _mve_free_b();
            // r4
            sub_e_4 = _mve_dict_b(sbox, rounded_e_4);
            // free rounded_e_4 (r5)
            _mve_free_b();
            // r5
            sub_e_5 = _mve_dict_b(sbox, rounded_e_5);
            // free rounded_e_5 (r6)
            _mve_free_b();
            // r6
            sub_e_6 = _mve_dict_b(sbox, rounded_e_6);
            // free rounded_e_6 (r7)
            _mve_free_b();
            // r7
            sub_e_7 = _mve_dict_b(sbox, rounded_e_7);
            // free rounded_e_7 (r8)
            _mve_free_b();
            // r8
            sub_e_8 = _mve_dict_b(sbox, rounded_e_8);
            // free rounded_e_8 (r9)
            _mve_free_b();
            // r9
            sub_e_9 = _mve_dict_b(sbox, rounded_e_9);
            // free rounded_e_9 (r10)
            _mve_free_b();
            // r10
            sub_e_10 = _mve_dict_b(sbox, rounded_e_10);
            // free rounded_e_10 (r11)
            _mve_free_b();
            // r11
            sub_e_11 = _mve_dict_b(sbox, rounded_e_11);
            // free rounded_e_11 (r12)
            _mve_free_b();
            // r12
            sub_e_12 = _mve_dict_b(sbox, rounded_e_12);
            // free rounded_e_12 (r13)
            _mve_free_b();
            // r13
            sub_e_13 = _mve_dict_b(sbox, rounded_e_13);
            // free rounded_e_13 (r14)
            _mve_free_b();
            // r14
            sub_e_14 = _mve_dict_b(sbox, rounded_e_14);
            // free rounded_e_14 (r15)
            _mve_free_b();
            // r15
            sub_e_15 = _mve_dict_b(sbox, rounded_e_15);
            // free rounded_e_15 (r16)
            _mve_free_b();

            // [r1-16] <- [r0-15]

            // Shift Rows
            // Do nothing

            if (round == 10) {
                break;
            }

            // [r1-16] <- [r0-15]

            // Mix Columns
            // using r0-15 for mixed_e s
            __mdvb allxored, twoxored, Tm;
            __mdvb Tm_shl, Tm_plus_1b;
            __mdvb TmAllxored;
            __mdvb C1b, C127;
            // r17
            Tm = _mve_xor_b(sub_e_0, sub_e_5);
            // r18
            twoxored = _mve_xor_b(sub_e_10, sub_e_15);
            // r19
            allxored = _mve_xor_b(Tm, twoxored);
            // free twoxored (r18)
            _mve_free_b();

            // [r0, r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r0, r20, r21]

            // r20
            C1b = _mve_set1_b(0x1b);

            // [r0, r21]

            // r21
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r0, r20]

            // r20
            C127 = _mve_set1_b(127);

            // [r0]

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128(r20)
            _mve_free_b();
            _mve_free_b();

            // [r0, r17, r20]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r21)
            _mve_free_b();
            _mve_free_b();

            // [r0, r18, r20, r21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [r0, r17, r20, r21]

            // r0
            __mdvb mixed_e_0 = _mve_xor_b(sub_e_0, TmAllxored);
            // free TmAllxored (r18) and sub_e_0(1)
            _mve_free_b();
            _mve_free_b();

            // [r17, r18, r20, r21]

            // r17
            Tm = _mve_xor_b(sub_e_5, sub_e_10);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r1
            __mdvb mixed_e_1 = _mve_xor_b(sub_e_5, TmAllxored);
            // free TmAllxored (r18) and sub_e_5 (r2)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_10, sub_e_15);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r2
            __mdvb mixed_e_2 = _mve_xor_b(sub_e_10, TmAllxored);
            // free TmAllxored (r18) and sub_e_10 (r3)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_15, sub_e_0);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Rm (r17) and allxored (r19)
            _mve_free_b();
            _mve_free_b();

            // [17, 19, 20, 21]

            // r3
            __mdvb mixed_e_3 = _mve_xor_b(sub_e_15, TmAllxored);
            // free TmAllxored (r18) and sub_e_15 (r4)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 19, 20, 21]
            // r17
            Tm = _mve_xor_b(sub_e_4, sub_e_9);
            // r18
            twoxored = _mve_xor_b(sub_e_14, sub_e_3);
            // r19
            allxored = _mve_xor_b(Tm, twoxored);
            // free twoxored (r18)
            _mve_free_b();

            // [r0, r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r0, r20, r21]

            // r20
            C1b = _mve_set1_b(0x1b);

            // [r0, r21]

            // r21
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r0, r20]

            // r20
            C127 = _mve_set1_b(127);

            // [r0]

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128(r20)
            _mve_free_b();
            _mve_free_b();

            // [r0, r17, r20]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r21)
            _mve_free_b();
            _mve_free_b();

            // [r0, r18, r20, r21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [r0, r17, r20, r21]

            // r4
            __mdvb mixed_e_4 = _mve_xor_b(sub_e_4, TmAllxored);
            // free TmAllxored (r18) and sub_e_4(5)
            _mve_free_b();
            _mve_free_b();

            // [r17, r18, r20, r21]

            // r17
            Tm = _mve_xor_b(sub_e_9, sub_e_14);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r5
            __mdvb mixed_e_5 = _mve_xor_b(sub_e_9, TmAllxored);
            // free TmAllxored (r18) and sub_e_9 (r6)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_14, sub_e_3);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r6
            __mdvb mixed_e_6 = _mve_xor_b(sub_e_14, TmAllxored);
            // free TmAllxored (r18) and sub_e_14 (r7)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_3, sub_e_4);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Rm (r17) and allxored (r19)
            _mve_free_b();
            _mve_free_b();

            // [17, 19, 20, 21]

            // r7
            __mdvb mixed_e_7 = _mve_xor_b(sub_e_3, TmAllxored);
            // free TmAllxored (r18) and sub_e_3 (r8)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 19, 20, 21]
            // r17
            Tm = _mve_xor_b(sub_e_8, sub_e_13);
            // r18
            twoxored = _mve_xor_b(sub_e_2, sub_e_7);
            // r19
            allxored = _mve_xor_b(Tm, twoxored);
            // free twoxored (r18)
            _mve_free_b();

            // [r0, r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r0, r20, r21]

            // r20
            C1b = _mve_set1_b(0x1b);

            // [r0, r21]

            // r21
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r0, r20]

            // r20
            C127 = _mve_set1_b(127);

            // [r0]

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128(r20)
            _mve_free_b();
            _mve_free_b();

            // [r0, r17, r20]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r21)
            _mve_free_b();
            _mve_free_b();

            // [r0, r18, r20, r21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [r0, r17, r20, r21]

            // r8
            __mdvb mixed_e_8 = _mve_xor_b(sub_e_8, TmAllxored);
            // free TmAllxored (r18) and sub_e_8(9)
            _mve_free_b();
            _mve_free_b();

            // [r17, r18, r20, r21]

            // r17
            Tm = _mve_xor_b(sub_e_13, sub_e_2);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r9
            __mdvb mixed_e_9 = _mve_xor_b(sub_e_13, TmAllxored);
            // free TmAllxored (r18) and sub_e_13 (r10)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_2, sub_e_7);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r10
            __mdvb mixed_e_10 = _mve_xor_b(sub_e_2, TmAllxored);
            // free TmAllxored (r18) and sub_e_2 (r11)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_7, sub_e_8);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Rm (r17) and allxored (r19)
            _mve_free_b();
            _mve_free_b();

            // [17, 19, 20, 21]

            // r11
            __mdvb mixed_e_11 = _mve_xor_b(sub_e_7, TmAllxored);
            // free TmAllxored (r18) and sub_e_7 (r12)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 19, 20, 21]
            // r17
            Tm = _mve_xor_b(sub_e_12, sub_e_1);
            // r18
            twoxored = _mve_xor_b(sub_e_6, sub_e_11);
            // r19
            allxored = _mve_xor_b(Tm, twoxored);
            // free twoxored (r18)
            _mve_free_b();

            // [r0, r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r0, r20, r21]

            // r20
            C1b = _mve_set1_b(0x1b);

            // [r0, r21]

            // r21
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r0, r20]

            // r20
            C127 = _mve_set1_b(127);

            // [r0]

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128(r20)
            _mve_free_b();
            _mve_free_b();

            // [r0, r17, r20]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r21)
            _mve_free_b();
            _mve_free_b();

            // [r0, r18, r20, r21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [r0, r17, r20, r21]

            // r12
            __mdvb mixed_e_12 = _mve_xor_b(sub_e_12, TmAllxored);
            // free TmAllxored (r18) and sub_e_12(13)
            _mve_free_b();
            _mve_free_b();

            // [r17, r18, r20, r21]

            // r17
            Tm = _mve_xor_b(sub_e_1, sub_e_6);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r13
            __mdvb mixed_e_13 = _mve_xor_b(sub_e_1, TmAllxored);
            // free TmAllxored (r18) and sub_e_1 (r14)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_6, sub_e_11);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r21)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Tm (r17)
            _mve_free_b();

            // [17, 20, 21]

            // r14
            __mdvb mixed_e_14 = _mve_xor_b(sub_e_6, TmAllxored);
            // free TmAllxored (r18) and sub_e_6 (r15)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 20, 21]

            // r17
            Tm = _mve_xor_b(sub_e_11, sub_e_12);

            // [r18, r20, r21]

            // r18
            Tm_shl = _mve_shil_b(Tm, 1);

            // [r20, r21]

            // r21
            C1b = _mve_set1_b(0x1b);

            // [r20]

            // r20
            Tm_plus_1b = _mve_xor_b(Tm_shl, C1b);
            // free C1b (r20)
            _mve_free_b();

            // [r21]

            // r21
            C127 = _mve_set1_b(127);

            // []

            _mve_cmpgt_b(Tm, C127);
            // free Tm (r17) and C128 (r21)
            _mve_free_b();
            _mve_free_b();

            // [17, 21]

            // r17
            Tm = _mve_assign_b(Tm_shl, Tm_plus_1b);
            // free Tm_shl (r18) and Tm_plus_1b (r20)
            _mve_free_b();
            _mve_free_b();

            // [18, 20, 21]

            _mve_set_mask();
            // r18
            TmAllxored = _mve_xor_b(Tm, allxored);
            // free Rm (r17) and allxored (r19)
            _mve_free_b();
            _mve_free_b();

            // [17, 19, 20, 21]

            // r15
            __mdvb mixed_e_15 = _mve_xor_b(sub_e_11, TmAllxored);
            // free TmAllxored (r18) and sub_e_11 (r16)
            _mve_free_b();
            _mve_free_b();

            // [17, 18, 19, 20, 21]

            // [r1-16] <- [r0-r15]

            // Add Round Key
            // using r0-15 for rounded_e s
            RoundKey_addr = RoundKey + (round * 16);

            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r0
            rounded_e_0 = _mve_xor_b(mixed_e_0, key);
            // free mixed_e_0 (r1) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r1
            rounded_e_1 = _mve_xor_b(mixed_e_1, key);
            // free mixed_e_1 (r2) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r2
            rounded_e_2 = _mve_xor_b(mixed_e_2, key);
            // free mixed_e_2 (r3) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r3
            rounded_e_3 = _mve_xor_b(mixed_e_3, key);
            // free mixed_e_3 (r4) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r4
            rounded_e_4 = _mve_xor_b(mixed_e_4, key);
            // free mixed_e_4 (r5) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r5
            rounded_e_5 = _mve_xor_b(mixed_e_5, key);
            // free mixed_e_5 (r6) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r6
            rounded_e_6 = _mve_xor_b(mixed_e_6, key);
            // free mixed_e_6 (r7) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r7
            rounded_e_7 = _mve_xor_b(mixed_e_7, key);
            // free mixed_e_7 (r8) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r8
            rounded_e_8 = _mve_xor_b(mixed_e_8, key);
            // free mixed_e_8 (r9) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r9
            rounded_e_9 = _mve_xor_b(mixed_e_9, key);
            // free mixed_e_9 (r10) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r10
            rounded_e_10 = _mve_xor_b(mixed_e_10, key);
            // free mixed_e_10 (r11) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r11
            rounded_e_11 = _mve_xor_b(mixed_e_11, key);
            // free mixed_e_11 (r12) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r12
            rounded_e_12 = _mve_xor_b(mixed_e_12, key);
            // free mixed_e_12 (r13) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r13
            rounded_e_13 = _mve_xor_b(mixed_e_13, key);
            // free mixed_e_13 (r14) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r14
            rounded_e_14 = _mve_xor_b(mixed_e_14, key);
            // free mixed_e_14 (r15) and key (r17)
            _mve_free_b();
            _mve_free_b();
            // r17
            key = _mve_set1_b(*RoundKey_addr);
            RoundKey_addr += 1;
            // r15
            rounded_e_15 = _mve_xor_b(mixed_e_15, key);
            // free mixed_e_15 (r16) and key (r17)
            _mve_free_b();
            _mve_free_b();

            mve_flusher();
        }

        // Add Round Key

        RoundKey_addr = RoundKey + 160;

        __mdvb added_key;
        state_addr = state;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_0, key);
        // free sub_e_0(r0) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_5, key);
        // free sub_e_5(r5) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_10, key);
        // free sub_e_10(r10) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_15, key);
        // free sub_e_15(r15) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_4, key);
        // free sub_e_4(r4) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_9, key);
        // free sub_e_9(r9) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_14, key);
        // free sub_e_14(r14) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_3, key);
        // free sub_e_3(r3) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_8, key);
        // free sub_e_8(r8) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_13, key);
        // free sub_e_13(r13) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_2, key);
        // free sub_e_2(r2) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_7, key);
        // free sub_e_7(r7) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_12, key);
        // free sub_e_12(r12) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_1, key);
        // free sub_e_1(r1) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_6, key);
        // free sub_e_6(r6) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;
        // r16
        key = _mve_set1_b(*RoundKey_addr);
        RoundKey_addr += 1;

        // r17
        added_key = _mve_xor_b(sub_e_11, key);
        // free sub_e_11(r11) and key (r16)
        _mve_free_b();
        _mve_free_b();

        _mve_store_b(state_addr, added_key, elementwise_stride);
        // free added_key (r17)
        _mve_free_b();

        state_addr += 1;

        state += (count_per_iter * 16);
    }
}