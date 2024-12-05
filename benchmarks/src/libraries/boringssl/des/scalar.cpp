#include "des.hpp"
#include "scalar_kernels.hpp"
#include <cstdio>

static inline uint32_t CRYPTO_rotr_u32(uint32_t value, int shift) {
    /* printf("shifting left %d and right %d\n", ((-shift) & 31), shift); */
    return (value >> shift) | (value << ((-shift) & 31));
}

#define PERM_OP(a, b, t, n, m)                                             \
    do {                                                                   \
        /* printf("a(%d), b(%d), n(%d), m(%lx)\n", (a), (b), (n), (m)); */ \
        (t) = ((((a) >> (n)) ^ (b)) & (m));                                \
        /* printf("t = %d\n", (t));                                    */  \
        (b) ^= (t);                                                        \
        (a) ^= ((t) << (n));                                               \
    } while (0)

#define IP(l, r)                            \
    do {                                    \
        uint32_t tt;                        \
        PERM_OP(r, l, tt, 4, 0x0f0f0f0fL);  \
        /* printf("%d %d\n", l, r); */      \
        PERM_OP(l, r, tt, 16, 0x0000ffffL); \
        /* printf("%d %d\n", l, r); */      \
        PERM_OP(r, l, tt, 2, 0x33333333L);  \
        /* printf("%d %d\n", l, r); */      \
        PERM_OP(l, r, tt, 8, 0x00ff00ffL);  \
        /* printf("%d %d\n", l, r); */      \
        PERM_OP(r, l, tt, 1, 0x55555555L);  \
        /* printf("%d %d\n", l, r); */      \
    } while (0)

#define FP(l, r)                            \
    do {                                    \
        uint32_t tt;                        \
        PERM_OP(l, r, tt, 1, 0x55555555L);  \
        PERM_OP(r, l, tt, 8, 0x00ff00ffL);  \
        PERM_OP(l, r, tt, 2, 0x33333333L);  \
        PERM_OP(r, l, tt, 16, 0x0000ffffL); \
        PERM_OP(l, r, tt, 4, 0x0f0f0f0fL);  \
    } while (0)

#define LOAD_DATA(SK0, SK1, R, S, u, t) \
    do {                                \
        (u) = (R) ^ SK0;                \
        (t) = (R) ^ SK1;                \
    } while (0)

#define D_ENCRYPT(SK, LL, R, S)                                                    \
    do {                                                                           \
        (SK0) = SK[(S) * 2];                                                       \
        (SK1) = SK[(S) * 2 + 1];                                                   \
        (u) = (R) ^ (SK0);                                                         \
        (t) = (R) ^ (SK1);                                                         \
        t = CRYPTO_rotr_u32(t, 4);                                                 \
        (LL) ^=                                                                    \
            DES_SPtrans[0][(u >> 2L) & 0x3f] ^ DES_SPtrans[2][(u >> 10L) & 0x3f] ^ \
            DES_SPtrans[4][(u >> 18L) & 0x3f] ^                                    \
            DES_SPtrans[6][(u >> 26L) & 0x3f] ^ DES_SPtrans[1][(t >> 2L) & 0x3f] ^ \
            DES_SPtrans[3][(t >> 10L) & 0x3f] ^                                    \
            DES_SPtrans[5][(t >> 18L) & 0x3f] ^ DES_SPtrans[7][(t >> 26L) & 0x3f]; \
    } while (0)

void des_scalar(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {
    des_config_t *des_config = (des_config_t *)config;
    des_input_t *des_input = (des_input_t *)input;

    int num_blocks = des_config->num_blocks;

    uint32_t *state = (uint32_t *)des_input->state;
    uint32_t *subkeys = (uint32_t *)des_config->subkeys;

    uint32_t l, r, t, u, SK0, SK1;

    for (int blk = 0; blk < num_blocks; blk++) {

        r = state[0];
        l = state[1];

        IP(r, l);
        // printf("%d %d\n", l, r);
        // Things have been modified so that the initial rotate is done outside
        // the loop.  This required the DES_SPtrans values in sp.h to be
        // rotated 1 bit to the right. One perl script later and things have a
        // 5% speed up on a sparc2. Thanks to Richard Outerbridge
        // <71755.204@CompuServe.COM> for pointing this out.
        // clear the top bits on machines with 8byte longs
        // shift left by 2
        /* printf("Rotate: \n"); */

        r = CRYPTO_rotr_u32(r, 29);
        l = CRYPTO_rotr_u32(l, 29);
        printf("%d %d\n", l, r);

        /* printf("Rounds: \n"); */

        // I don't know if it is worth the effort of loop unrolling the
        // inner loop
        D_ENCRYPT(subkeys, l, r, 0);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 1);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 2);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 3);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 4);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 5);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 6);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 7);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 8);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 9);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 10);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 11);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 12);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 13);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, l, r, 14);
        /* printf("%d %d\n", l, r); */
        D_ENCRYPT(subkeys, r, l, 15);
        /* printf("%d %d\n", l, r); */

        // rotate and clear the top bits on machines with 8byte longs
        l = CRYPTO_rotr_u32(l, 3);
        r = CRYPTO_rotr_u32(r, 3);
        /* printf("%d %d\n", l, r); */

        FP(r, l);
        /* printf("%d %d\n", l, r); */
        state[0] = l;
        state[1] = r;
        state += 2;
    }

    state = (uint32_t *)des_input->state;
    for (int blk = 0; blk < num_blocks; blk++) {
        printf("%d %d\n", state[0], state[1]);
        state += 2;
    }
}