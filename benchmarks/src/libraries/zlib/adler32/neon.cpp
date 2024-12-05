#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdio>
#include <stdint.h>

#include "adler32.hpp"
#include "zlib.hpp"

// Does vertical convolution to produce one output row. The filter values and
// length are given in the first two parameters. These are applied to each
// of the rows pointed to in the |source_data_rows| array, with each row
// being |pixel_width| wide.
//
// The output must have room for |pixel_width * 4| bytes.
void adler32_neon(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    /*
     * Split Adler-32 into component sums.
     */
    uint32_t s1 = adler & 0xffff;
    uint32_t s2 = adler >> 16;

    /*
     * Process the data in blocks.
     */
    const unsigned BLOCK_SIZE = 1 << 5;

    z_size_t blocks = len / BLOCK_SIZE;
    len -= blocks * BLOCK_SIZE;

    while (blocks) {
        unsigned n = NMAX / BLOCK_SIZE; /* The NMAX constraint. */
        if (n > blocks)
            n = (unsigned)blocks;
        blocks -= n;

        /*
         * Process n blocks of data. At most NMAX data bytes can be
         * processed before s2 must be reduced modulo BASE.
         */
        uint32x4_t v_s2 = (uint32x4_t){0, 0, 0, s1 * n};
        uint32x4_t v_s1 = (uint32x4_t){0, 0, 0, 0};

        uint16x8_t v_column_sum_1 = vdupq_n_u16(0);
        uint16x8_t v_column_sum_2 = vdupq_n_u16(0);
        uint16x8_t v_column_sum_3 = vdupq_n_u16(0);
        uint16x8_t v_column_sum_4 = vdupq_n_u16(0);

        do {
            /*
             * Load 32 input bytes.
             */
            const uint8x16_t bytes1 = vld1q_u8((uint8_t *)(buf));
            const uint8x16_t bytes2 = vld1q_u8((uint8_t *)(buf + 16));

            /*
             * Add previous block byte sum to v_s2.
             */
            v_s2 = vaddq_u32(v_s2, v_s1);

            /*
             * Horizontally add the bytes for s1.
             */
            v_s1 = vpadalq_u16(v_s1, vpadalq_u8(vpaddlq_u8(bytes1), bytes2));

            /*
             * Vertically add the bytes for s2.
             */
            v_column_sum_1 = vaddw_u8(v_column_sum_1, vget_low_u8(bytes1));
            v_column_sum_2 = vaddw_u8(v_column_sum_2, vget_high_u8(bytes1));
            v_column_sum_3 = vaddw_u8(v_column_sum_3, vget_low_u8(bytes2));
            v_column_sum_4 = vaddw_u8(v_column_sum_4, vget_high_u8(bytes2));

            buf += BLOCK_SIZE;

        } while (--n);

        v_s2 = vshlq_n_u32(v_s2, 5);

        /*
         * Multiply-add bytes by [ 32, 31, 30, ... ] for s2.
         */
        v_s2 = vmlal_u16(v_s2, vget_low_u16(v_column_sum_1),
                         (uint16x4_t){32, 31, 30, 29});
        v_s2 = vmlal_u16(v_s2, vget_high_u16(v_column_sum_1),
                         (uint16x4_t){28, 27, 26, 25});
        v_s2 = vmlal_u16(v_s2, vget_low_u16(v_column_sum_2),
                         (uint16x4_t){24, 23, 22, 21});
        v_s2 = vmlal_u16(v_s2, vget_high_u16(v_column_sum_2),
                         (uint16x4_t){20, 19, 18, 17});
        v_s2 = vmlal_u16(v_s2, vget_low_u16(v_column_sum_3),
                         (uint16x4_t){16, 15, 14, 13});
        v_s2 = vmlal_u16(v_s2, vget_high_u16(v_column_sum_3),
                         (uint16x4_t){12, 11, 10, 9});
        v_s2 = vmlal_u16(v_s2, vget_low_u16(v_column_sum_4),
                         (uint16x4_t){8, 7, 6, 5});
        v_s2 = vmlal_u16(v_s2, vget_high_u16(v_column_sum_4),
                         (uint16x4_t){4, 3, 2, 1});

        /*
         * Sum epi32 ints v_s1(s2) and accumulate in s1(s2).
         */
        uint32x2_t sum1 = vpadd_u32(vget_low_u32(v_s1), vget_high_u32(v_s1));
        uint32x2_t sum2 = vpadd_u32(vget_low_u32(v_s2), vget_high_u32(v_s2));
        uint32x2_t s1s2 = vpadd_u32(sum1, sum2);

        s1 += vget_lane_u32(s1s2, 0);
        s2 += vget_lane_u32(s1s2, 1);

        /*
         * Reduce.
         */
        s1 %= BASE;
        s2 %= BASE;
    }

    /*
     * Handle leftover data.
     */
    if (len) {
        if (len >= 16) {
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);

            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);

            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);

            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);
            s2 += (s1 += *buf++);

            len -= 16;
        }

        while (len--) {
            s2 += (s1 += *buf++);
        }

        if (s1 >= BASE)
            s1 -= BASE;
        s2 %= BASE;
    }

    /*
     * Return the recombined sums.
     */
    adler32_output->return_value[0] = s1 | (s2 << 16);
}
