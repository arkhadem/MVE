#include "crc32.hpp"
#include "scalar_kernels.hpp"
#include "zlib.hpp"
#include <stdio.h>

z_crc_t crc32_z(z_crc_t crc, const unsigned char *buf, z_size_t len) {

    crc = (~crc) & 0xffffffff;

    /* If provided enough bytes, do a braided CRC calculation. */
    if (len >= N * W + W - 1) {
        z_size_t blks;
        z_word_t const *words;
        int k;

        /* Compute the CRC up to a z_word_t boundary. */
        while (len && ((z_size_t)buf & (W - 1)) != 0) {
            len--;
            crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        }

        /* Compute the CRC on as many N z_word_t blocks as are available. */
        blks = len / (N * W);
        len -= blks * N * W;
        words = (z_word_t const *)buf;

        /* Do endian check at execution time instead of compile time, since ARM
           processors can change the endianess at execution time. If the
           compiler knows what the endianess will be, it can optimize out the
           check and the unused branch. */
        /* Little endian. */

        z_crc_t crc0;
        z_word_t word0;
        z_crc_t crc1;
        z_word_t word1;
        z_crc_t crc2;
        z_word_t word2;
        z_crc_t crc3;
        z_word_t word3;
        z_crc_t crc4;
        z_word_t word4;

        /* Initialize the CRC for each braid. */
        crc0 = crc;
        crc1 = 0;
        crc2 = 0;
        crc3 = 0;
        crc4 = 0;

        /*
              Process the first blks-1 blocks, computing the CRCs on each braid
              independently.
             */
        while (--blks) {
            /* Load the word for each braid into registers. */
            word0 = crc0 ^ words[0];
            word1 = crc1 ^ words[1];
            word2 = crc2 ^ words[2];
            word3 = crc3 ^ words[3];
            word4 = crc4 ^ words[4];

            words += N;

            /* Compute and update the CRC for each word. The loop should
                   get unrolled. */
            crc0 = crc_braid_table[0][word0 & 0xff];
            crc1 = crc_braid_table[0][word1 & 0xff];
            crc2 = crc_braid_table[0][word2 & 0xff];
            crc3 = crc_braid_table[0][word3 & 0xff];
            crc4 = crc_braid_table[0][word4 & 0xff];

            for (k = 1; k < W; k++) {
                crc0 ^= crc_braid_table[k][(word0 >> (k << 3)) & 0xff];
                crc1 ^= crc_braid_table[k][(word1 >> (k << 3)) & 0xff];
                crc2 ^= crc_braid_table[k][(word2 >> (k << 3)) & 0xff];
                crc3 ^= crc_braid_table[k][(word3 >> (k << 3)) & 0xff];
                crc4 ^= crc_braid_table[k][(word4 >> (k << 3)) & 0xff];
            }
        }

        /*
            Process the last block, combining the CRCs of the N braids at the
            same time.
        */

        crc = crc_word(crc0 ^ words[0]);
        crc = crc_word(crc1 ^ words[1] ^ crc);
        crc = crc_word(crc2 ^ words[2] ^ crc);
        crc = crc_word(crc3 ^ words[3] ^ crc);
        crc = crc_word(crc4 ^ words[4] ^ crc);
        words += N;

        /*
          Update the pointer to the remaining bytes to process.
         */
        buf = (unsigned char const *)words;
    }

    /* Complete the computation of the CRC on any remaining bytes. */
    while (len >= 8) {
        len -= 8;
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
    }
    while (len) {
        len--;
        crc = (crc >> 8) ^ crc_table[(crc ^ *buf++) & 0xff];
    }

    /* Return the CRC, post-conditioned. */
    return crc ^ 0xffffffff;
}

void crc32_scalar(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    crc32_config_t *crc32_config = (crc32_config_t *)config;
    crc32_input_t *crc32_input = (crc32_input_t *)input;
    crc32_output_t *crc32_output = (crc32_output_t *)output;
    crc32_output->return_value[0] = crc32_z(crc32_config->crc, crc32_input->buf, crc32_config->len);
}

int __main() {
    int number = (unsigned char)('~') - (unsigned char)(' ');
    unsigned char buf[65536 + 1];
    for (int i = 0; i < 65536; i++) {
        buf[i] = (unsigned char)(21) + (unsigned char)(i % number);
    }
    buf[65536] = 0;

    z_crc_t first_crc = rand() % 0xFFFFFFFF;

    printf("%x\n", crc32_z(first_crc, buf, 65536));

    z_crc_t coeffs[8192];
    for (int i = 0; i < 8192; i++) {
        // offline: calculating shift coeffs
        coeffs[i] = x2nmodp(8 * (8191 - i), 3);
    }
    z_crc_t Coeff_64K_shift = x2nmodp(64 * 1024, 3);

    z_crc_t crcs[8192];
    for (int i = 0; i < 8192; i++) {
        // calculating crc of a 64-bit block
        crcs[i] = crc32_z(0, buf + 8 * i, 8);
        // shifting it to the right
        crcs[i] = multmodp(coeffs[i], crcs[i]);
    }

    // reduction
    z_crc_t crc = multmodp(Coeff_64K_shift, first_crc);
    for (int i = 0; i < 8192; i++) {
        crc = crc ^ crcs[i];
    }

    printf("%x\n", crc);

    // z_crc_t crc1 = crc32_z(0, buf + 0, 8);
    // z_crc_t crc2 = crc32_z(0, buf + 8, 8);
    // z_crc_t crc3 = crc32_z(0, buf + 16, 8);
    // z_crc_t crc4 = crc32_z(0, buf + 24, 8);

    // z_crc_t crc12 = crc32_combine64(crc1, crc2, 8);
    // z_crc_t crc34 = crc32_combine64(crc3, crc4, 8);

    // crc = crc32_combine64(crc12, crc34, 16);
    // printf("%x\n", crc);
    // crc = crc32_z(0, buf, 32);
    // printf("%x\n", crc);

    return 0;
}