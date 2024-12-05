#include "mve.hpp"
#include "mve_kernels.hpp"

#include "des.hpp"

#define PERM_OP(a, b, n, m)                   \
    do {                                      \
        /* consumed 2 registers (a R5 b R4)*/ \
        /*R0*/                                \
        __mdvdw t1 = _mve_shiru_dw(a, n);     \
                                              \
        /*R1*/                                \
        __mdvdw t2 = _mve_xor_dw(t1, b);      \
        /* free t1 (R0)*/                     \
        _mve_free_dw();                       \
                                              \
        /*R0*/                                \
        __mdvdw val = _mve_set1_dw(m);        \
                                              \
        /*R2*/                                \
        __mdvdw t3 = _mve_and_dw(t2, val);    \
        /* free t2 (R1) and val (R0)*/        \
        _mve_free_dw();                       \
        _mve_free_dw();                       \
                                              \
        /*R0*/                                \
        b = _mve_xor_dw(b, t3);               \
        /* b (R4)*/                           \
        _mve_free_dw();                       \
                                              \
        /*R1*/                                \
        __mdvdw t4 = _mve_shil_dw(t3, n);     \
        /* free t3 (R2)*/                     \
        _mve_free_dw();                       \
                                              \
        /*R2*/                                \
        a = _mve_xor_dw(a, t4);               \
        /* free t4 (R1) and free a (R5)*/     \
        _mve_free_dw();                       \
        _mve_free_dw();                       \
                                              \
        /* now a and b are in R2 and R0*/     \
    } while (0)

#define IP(l, r)                        \
    do {                                \
        PERM_OP(r, l, 4, 0x0f0f0f0fL);  \
        PERM_OP(l, r, 16, 0x0000ffffL); \
        PERM_OP(r, l, 2, 0x33333333L);  \
        PERM_OP(l, r, 8, 0x00ff00ffL);  \
        PERM_OP(r, l, 1, 0x55555555L);  \
    } while (0)

#define FP(l, r)                        \
    do {                                \
        PERM_OP(l, r, 1, 0x55555555L);  \
        PERM_OP(r, l, 8, 0x00ff00ffL);  \
        PERM_OP(l, r, 2, 0x33333333L);  \
        PERM_OP(r, l, 16, 0x0000ffffL); \
        PERM_OP(l, r, 4, 0x0f0f0f0fL);  \
    } while (0)

#define DICTIONARY(table, ut, shift_val, result)                     \
    do {                                                             \
        __mdvdw shift = _mve_shiru_dw(ut, shift_val);                \
        __mdvb key1 = _mve_cvt_dwtob(shift);                         \
        /* free shift*/                                              \
        _mve_free_dw();                                              \
        __mdvb key2 = _mve_and_b(key1, const_3f);                    \
        /* free key1*/                                               \
        _mve_free_b();                                               \
        result = _mve_set1_dw(0);                                    \
        for (int i = 0; i < 64; i++) {                               \
            __mdvb key_b = _mve_set1_b(i);                           \
            __mdvdw val_dw = _mve_set1_dw(table[i]);                 \
            _mve_cmpeq_b(key2, key_b);                               \
            /* free key_b */                                         \
            _mve_free_b();                                           \
            result = _mve_assign_dw(result, val_dw);                 \
            /* free result and val_dw */                             \
            _mve_free_dw();                                          \
            _mve_free_dw();                                          \
            _mve_set_mask();                                         \
        }                                                            \
        /* result = _mve_dict_dw((const __int32_t *)table, key2); */ \
        /* free key2*/                                               \
        _mve_free_b();                                               \
    } while (0)

#define D_ENCRYPT(subkeys, l_dw, r_dw, round)                    \
    do {                                                         \
        /* R0 */                                                 \
        __mdvdw SK0_dw = _mve_set1_dw(subkeys[(round) * 2]);     \
        /* R1 */                                                 \
        __mdvdw SK1_dw = _mve_set1_dw(subkeys[(round) * 2 + 1]); \
        /* R2 */                                                 \
        __mdvdw u = _mve_xor_dw(r_dw, SK0_dw);                   \
        /* free SK0_dw (R0) */                                   \
        _mve_free_dw();                                          \
        /* R0 */                                                 \
        __mdvdw temp = _mve_xor_dw(r_dw, SK1_dw);                \
        /* free SK1_dw (R1) */                                   \
        _mve_free_dw();                                          \
        /* R1 */                                                 \
        __mdvdw t = _mve_rotir_dw(temp, 4);                      \
        /* free temp (R0) */                                     \
        _mve_free_dw();                                          \
        /* [R0 R3] */                                            \
        DICTIONARY(DES_SPtrans[0], u, 2, l_0_dw);                \
        DICTIONARY(DES_SPtrans[2], u, 10, l_2_dw);               \
        /* R0 */                                                 \
        __mdvdw l_02_dw = _mve_xor_dw(l_0_dw, l_2_dw);           \
        /* free l_0_dw and l_2_dw */                             \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        DICTIONARY(DES_SPtrans[4], u, 18, l_4_dw);               \
        DICTIONARY(DES_SPtrans[6], u, 26, l_6_dw);               \
        /* free u (R2) */                                        \
        _mve_free_dw();                                          \
        /* R3 */                                                 \
        __mdvdw l_46_dw = _mve_xor_dw(l_4_dw, l_6_dw);           \
        /* free l_4_dw and l_6_dw */                             \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        /* R2 */                                                 \
        __mdvdw l_0246_dw = _mve_xor_dw(l_02_dw, l_46_dw);       \
        /* free l_02_dw (R0) and l_46_dw (R3) */                 \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        /* [R0 R3] */                                            \
        DICTIONARY(DES_SPtrans[1], t, 2, l_1_dw);                \
        DICTIONARY(DES_SPtrans[3], t, 10, l_3_dw);               \
        /* R0 */                                                 \
        __mdvdw l_13_dw = _mve_xor_dw(l_1_dw, l_3_dw);           \
        /* free l_1_dw and l_3_dw */                             \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        DICTIONARY(DES_SPtrans[5], t, 18, l_5_dw);               \
        DICTIONARY(DES_SPtrans[7], t, 26, l_7_dw);               \
        /* free t (R1) */                                        \
        _mve_free_dw();                                          \
        /* R3 */                                                 \
        __mdvdw l_57_dw = _mve_xor_dw(l_5_dw, l_7_dw);           \
        /* free l_5_dw and l_7_dw */                             \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        /* R1 */                                                 \
        __mdvdw l_1357_dw = _mve_xor_dw(l_13_dw, l_57_dw);       \
        /* free l_13_dw (R0) and l_57_dw (R3) */                 \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        /* R0 */                                                 \
        __mdvdw l_all_dw = _mve_xor_dw(l_0246_dw, l_1357_dw);    \
        /* free l_0246_dw (R2) and l_1357_dw (R1) */             \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
        /* R4 */                                                 \
        l_dw = _mve_xor_dw(l_dw, l_all_dw);                      \
        /* free l_dw (R4) and l_all_dw (R0) */                   \
        _mve_free_dw();                                          \
        _mve_free_dw();                                          \
    } while (0)

void des_mve(int LANE_NUM,
             config_t *config,
             input_t *input,
             output_t *output) {
    des_config_t *des_config = (des_config_t *)config;
    des_input_t *des_input = (des_input_t *)input;

    int num_blocks = des_config->num_blocks;

    uint32_t *state = (uint32_t *)des_input->state;
    uint32_t *subkeys = (uint32_t *)des_config->subkeys;

    _mve_set_dim_count(1);

    int count_per_iter = LANE_NUM;

    // Loading and storing every other 32-bit
    _mve_set_load_stride(0, 2);
    _mve_set_store_stride(0, 2);
    __vidx_var elementwise_stride = {3, 0, 0, 0};

    _mve_set_dim_length(0, count_per_iter);

    __mdvdw l_0_dw, l_1_dw, l_2_dw, l_3_dw, l_4_dw, l_5_dw, l_6_dw, l_7_dw;

    __mdvb const_3f = _mve_set1_b(3);

    while (num_blocks > 0) {

        _mve_set_dim_length(0, num_blocks > count_per_iter ? count_per_iter : num_blocks);
        num_blocks -= count_per_iter;

        __mdvdw r_dw = _mve_load_dw((const __int32_t *)state, elementwise_stride);
        __mdvdw l_dw = _mve_load_dw((const __int32_t *)state + 1, elementwise_stride);

        IP(r_dw, l_dw);

        r_dw = _mve_rotir_dw(r_dw, 29);
        // free r_dw
        _mve_free_dw();
        l_dw = _mve_rotir_dw(l_dw, 29);
        // free l_dw
        _mve_free_dw();

        // printf("l = %s\n", _mve_dwtos(l_dw));
        // printf("r = %s\n", _mve_dwtos(r_dw));

        D_ENCRYPT(subkeys, l_dw, r_dw, 0);
        D_ENCRYPT(subkeys, r_dw, l_dw, 1);
        D_ENCRYPT(subkeys, l_dw, r_dw, 2);
        D_ENCRYPT(subkeys, r_dw, l_dw, 3);
        D_ENCRYPT(subkeys, l_dw, r_dw, 4);
        D_ENCRYPT(subkeys, r_dw, l_dw, 5);
        D_ENCRYPT(subkeys, l_dw, r_dw, 6);
        D_ENCRYPT(subkeys, r_dw, l_dw, 7);
        D_ENCRYPT(subkeys, l_dw, r_dw, 8);
        D_ENCRYPT(subkeys, r_dw, l_dw, 9);
        D_ENCRYPT(subkeys, l_dw, r_dw, 10);
        D_ENCRYPT(subkeys, r_dw, l_dw, 11);
        D_ENCRYPT(subkeys, l_dw, r_dw, 12);
        D_ENCRYPT(subkeys, r_dw, l_dw, 13);
        D_ENCRYPT(subkeys, l_dw, r_dw, 14);
        D_ENCRYPT(subkeys, r_dw, l_dw, 15);

        r_dw = _mve_rotir_dw(r_dw, 29);
        // free r_dw
        _mve_free_dw();
        l_dw = _mve_rotir_dw(l_dw, 29);
        // free l_dw
        _mve_free_dw();

        FP(r_dw, l_dw);

        _mve_store_dw((__int32_t *)state, r_dw, elementwise_stride);
        // free r_dw
        _mve_free_dw();
        _mve_store_dw((__int32_t *)state + 1, l_dw, elementwise_stride);
        // free l_dw
        _mve_free_dw();

        state += (count_per_iter * 2);
    }

    // free const_3f
    _mve_free_b();

    // state = (uint32_t *)des_input->state;
    // for (int blk = 0; blk < des_config->num_blocks; blk++) {
    //     printf("%x %x\n", state[0], state[1]);
    //     state += 2;
    // }
}