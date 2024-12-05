#include "des.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdio>

void print_first_element(uint32x4_t l, uint32x4_t r) {
    /* printf("%x %x\n", vgetq_lane_u32(l, 0), vgetq_lane_u32(r, 0)); */
}

#define CRYPTO_rotr_u32(value_0, value_1, shift) \
    val1_0 = vshlq_n_u32(value_0, 32 - shift);   \
    val1_1 = vshlq_n_u32(value_1, 32 - shift);   \
    val2_0 = vshrq_n_u32(value_0, shift);        \
    val2_1 = vshrq_n_u32(value_1, shift);        \
    value_0 = vorrq_u32(val1_0, val2_0);         \
    value_1 = vorrq_u32(val1_1, val2_1);

#define PERM_OP(a_0, a_1, b_0, b_1, t_0, t_1, n, m)                        \
    do {                                                                   \
        (t_0) = vandq_u32(veorq_u32(vshrq_n_u32((a_0), (n)), (b_0)), (m)); \
        (t_1) = vandq_u32(veorq_u32(vshrq_n_u32((a_1), (n)), (b_1)), (m)); \
        (b_0) = veorq_u32((b_0), (t_0));                                   \
        (b_1) = veorq_u32((b_1), (t_1));                                   \
        (a_0) = veorq_u32((a_0), vshlq_n_u32((t_0), (n)));                 \
        (a_1) = veorq_u32((a_1), vshlq_n_u32((t_1), (n)));                 \
    } while (0)

#define IP(l_0, l_1, r_0, r_1)                                        \
    do {                                                              \
        uint32x4_t tt_0, tt_1;                                        \
        PERM_OP(r_0, r_1, l_0, l_1, tt_0, tt_1, 4, const_0f0f0f0fL);  \
        PERM_OP(l_0, l_1, r_0, r_1, tt_0, tt_1, 16, const_0000ffffL); \
        PERM_OP(r_0, r_1, l_0, l_1, tt_0, tt_1, 2, const_33333333L);  \
        PERM_OP(l_0, l_1, r_0, r_1, tt_0, tt_1, 8, const_00ff00ffL);  \
        PERM_OP(r_0, r_1, l_0, l_1, tt_0, tt_1, 1, const_55555555L);  \
    } while (0)

#define FP(l_0, l_1, r_0, r_1)                                        \
    do {                                                              \
        uint32x4_t tt_0, tt_1;                                        \
        PERM_OP(l_0, l_1, r_0, r_1, tt_0, tt_1, 1, const_55555555L);  \
        PERM_OP(r_0, r_1, l_0, l_1, tt_0, tt_1, 8, const_00ff00ffL);  \
        PERM_OP(l_0, l_1, r_0, r_1, tt_0, tt_1, 2, const_33333333L);  \
        PERM_OP(r_0, r_1, l_0, l_1, tt_0, tt_1, 16, const_0000ffffL); \
        PERM_OP(l_0, l_1, r_0, r_1, tt_0, tt_1, 4, const_0f0f0f0fL);  \
    } while (0)

#define LOOKUP(RETURN, ADDRESS, VALUE, SHIFT)                            \
    do {                                                                 \
        uint32x4_t idx = vshrq_n_u32(VALUE, SHIFT);                      \
        idx = vandq_u32(idx, const_3fL);                                 \
        uint32x4_t temp = vdupq_n_u32(ADDRESS[vgetq_lane_u32(idx, 0)]);  \
        temp = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 1)], temp, 1); \
        temp = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 2)], temp, 2); \
        temp = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 3)], temp, 3); \
        (RETURN) = veorq_u32((RETURN), temp);                            \
    } while (0)

// Method 2
// uint32_t temp_mem[4];
// vst1q_u32(temp_mem, idx);
// temp_mem[0] = ADDRESS[temp_mem[0]];
// temp_mem[1] = ADDRESS[temp_mem[1]];
// temp_mem[2] = ADDRESS[temp_mem[2]];
// temp_mem[3] = ADDRESS[temp_mem[3]];
// RETURN = vld1q_u32(temp_mem);

// Method  1
// RETURN = vdupq_n_u32(ADDRESS[vgetq_lane_u32(idx, 0)]);
// RETURN = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 1)], RETURN, 1);
// RETURN = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 2)], RETURN, 2);
// RETURN = vsetq_lane_u32(ADDRESS[vgetq_lane_u32(idx, 3)], RETURN, 3);

#define D_ENCRYPT(SK, LL_0, LL_1, R_0, R_1, S)    \
    do {                                          \
        (SK0) = SK[(S) * 2];                      \
        (SK0_v) = vdupq_n_u32((SK0));             \
        (SK1) = SK[(S) * 2 + 1];                  \
        (SK1_v) = vdupq_n_u32((SK1));             \
        (u_0) = veorq_u32((R_0), (SK0_v));        \
        (u_1) = veorq_u32((R_1), (SK0_v));        \
        (t_0) = veorq_u32((R_0), (SK1_v));        \
        (t_1) = veorq_u32((R_1), (SK1_v));        \
        CRYPTO_rotr_u32(t_0, t_1, 4);             \
        LOOKUP((LL_0), DES_SPtrans[0], u_0, 2L);  \
        LOOKUP((LL_1), DES_SPtrans[0], u_1, 2L);  \
        LOOKUP((LL_0), DES_SPtrans[1], t_0, 2L);  \
        LOOKUP((LL_1), DES_SPtrans[1], t_1, 2L);  \
        LOOKUP((LL_0), DES_SPtrans[2], u_0, 10L); \
        LOOKUP((LL_1), DES_SPtrans[2], u_1, 10L); \
        LOOKUP((LL_0), DES_SPtrans[3], t_0, 10L); \
        LOOKUP((LL_1), DES_SPtrans[3], t_1, 10L); \
        LOOKUP((LL_0), DES_SPtrans[4], u_0, 18L); \
        LOOKUP((LL_1), DES_SPtrans[4], u_1, 18L); \
        LOOKUP((LL_0), DES_SPtrans[5], t_0, 18L); \
        LOOKUP((LL_1), DES_SPtrans[5], t_1, 18L); \
        LOOKUP((LL_0), DES_SPtrans[6], u_0, 26L); \
        LOOKUP((LL_1), DES_SPtrans[6], u_1, 26L); \
        LOOKUP((LL_0), DES_SPtrans[7], t_0, 26L); \
        LOOKUP((LL_1), DES_SPtrans[7], t_1, 26L); \
    } while (0)

void des_neon(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {
    des_config_t *des_config = (des_config_t *)config;
    des_input_t *des_input = (des_input_t *)input;

    int num_blocks = des_config->num_blocks;

    uint32_t *state = (uint32_t *)des_input->state;
    uint32_t *subkeys = (uint32_t *)des_config->subkeys;

    uint32_t SK0, SK1;
    uint32x4_t t_0, t_1, u_0, u_1;
    uint32x4_t SK0_v, SK1_v;

    uint32x4_t val1_0, val1_1;
    uint32x4_t val2_0, val2_1;

    uint32x4_t const_0f0f0f0fL = vdupq_n_u32(0x0f0f0f0fL);
    uint32x4_t const_0000ffffL = vdupq_n_u32(0x0000ffffL);
    uint32x4_t const_33333333L = vdupq_n_u32(0x33333333L);
    uint32x4_t const_00ff00ffL = vdupq_n_u32(0x00ff00ffL);
    uint32x4_t const_55555555L = vdupq_n_u32(0x55555555L);
    uint32x4_t const_3fL = vdupq_n_u32(0x3fL);

    for (int blk = 0; blk < num_blocks; blk += 8) {
        uint32x4_t r_0, l_0, r_1, l_1;

        uint32x4x2_t data = vld2q_u32(state);
        r_0 = data.val[0];
        l_0 = data.val[1];
        data = vld2q_u32(state + 8);
        r_1 = data.val[0];
        l_1 = data.val[1];
        /* print_first_element(l_0, r_0); */

        /* printf("IP: \n"); */

        IP(r_0, r_1, l_0, l_1);
        /* print_first_element(l_0, r_0); */
        // Things have been modified so that the initial rotate is done outside
        // the loop.  This required the DES_SPtrans values in sp.h to be
        // rotated 1 bit to the right. One perl script later and things have a
        // 5% speed up on a sparc2. Thanks to Richard Outerbridge
        // <71755.204@CompuServe.COM> for pointing this out.
        // clear the top bits on machines with 8byte longs
        // shift left by 2
        /* printf("Rotate: \n"); */

        CRYPTO_rotr_u32(r_0, r_1, 29);
        CRYPTO_rotr_u32(l_0, l_1, 29);
        /* print_first_element(l_0, r_0); */

        /* printf("Rounds: \n"); */

        // I don't know if it is worth the effort of loop unrolling the
        // inner loop
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 0);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 1);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 2);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 3);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 4);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 5);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 6);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 7);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 8);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 9);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 10);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 11);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 12);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 13);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, l_0, l_1, r_0, r_1, 14);
        /* print_first_element(l_0, r_0); */
        D_ENCRYPT(subkeys, r_0, r_1, l_0, l_1, 15);
        /* print_first_element(l_0, r_0); */
        // rotate and clear the top bits on machines with 8byte longs
        CRYPTO_rotr_u32(l_0, l_1, 3);
        CRYPTO_rotr_u32(r_0, r_1, 3);
        /* print_first_element(l_0, r_0); */

        FP(r_0, r_1, l_0, l_1);
        /* print_first_element(l_0, r_0); */

        data.val[0] = l_0;
        data.val[1] = r_0;
        vst2q_u32(state, data);
        data.val[0] = l_1;
        data.val[1] = r_1;
        vst2q_u32(state + 8, data);
        state += 16;
    }

    // state = (uint32_t *)des_input->state;
    // for (int blk = 0; blk < num_blocks; blk++) {
    //     printf("%x %x\n", state[0], state[1]);
    //     state += 2;
    // }
}