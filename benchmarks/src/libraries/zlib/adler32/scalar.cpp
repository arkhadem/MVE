#include "adler32.hpp"
#include "scalar_kernels.hpp"

void adler32_scalar(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t *adler32_input = (adler32_input_t *)input;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    const Bytef *buf = adler32_input->buf;
    z_size_t len = adler32_config->len;

    unsigned long sum2;
    unsigned n;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX / 16; /* NMAX is divisible by 16 */
        do {
            DO16(buf); /* 16 sums unrolled */
            buf += 16;
        } while (--n);
        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) { /* avoid modulos if none remaining */
        while (len >= 16) {
            len -= 16;
            DO16(buf);
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }

    /* return recombined sums */
    adler32_output->return_value[0] = adler | (sum2 << 16);
}

void adler32_mve_scalar(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_output_t *adler32_output = (adler32_output_t *)output;

    uLong adler = adler32_config->adler;
    z_size_t len = adler32_config->len;

    int64_t sum1_mem_qw[BLOCK_8K];
    int64_t sum2_mem_qw[BLOCK_8K];

    int32_t *sum1_mem_dw = (int32_t *)sum1_mem_qw;
    int32_t *sum2_mem_dw = (int32_t *)sum2_mem_qw;

    int16_t *sum1_mem_w = (int16_t *)sum1_mem_qw;

    uint32_t total_sum1 = (adler >> 16) & 0xffff;
    uint32_t total_sum2 = adler & 0xffff;

    while (len > 0) {

        // iteration 9: 32 additions (sum1 17bits) (sum2 26bits)
        int idx = 0;
#pragma unroll(32)
        for (int j = 0; j < 32; j++, idx += 2) {
            uint16_t first_sum_1 = sum1_mem_w[idx];
            uint16_t second_sum_1 = sum1_mem_w[idx + 1];
            sum1_mem_dw[j] = first_sum_1 + second_sum_1;
            uint32_t first_sum_2 = sum2_mem_dw[idx];
            uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
            uint32_t sum2 = first_sum_2 + second_sum_2;
            sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 8);
        }

        // iteration 10: 16 additions (sum1 18bits) (sum2 28bits)
        idx = 0;
#pragma unroll(16)
        for (int j = 0; j < 16; j++, idx += 2) {
            uint32_t first_sum_1 = sum1_mem_dw[idx];
            uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
            sum1_mem_dw[j] = first_sum_1 + second_sum_1;
            uint32_t first_sum_2 = sum2_mem_dw[idx];
            uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
            uint32_t sum2 = first_sum_2 + second_sum_2;
            sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 9);
        }

        // iteration 11: 8 additions (sum1 19bits) (sum2 30bits)
        idx = 0;
#pragma unroll(8)
        for (int j = 0; j < 8; j++, idx += 2) {
            uint32_t first_sum_1 = sum1_mem_dw[idx];
            uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
            sum1_mem_dw[j] = first_sum_1 + second_sum_1;
            uint32_t first_sum_2 = sum2_mem_dw[idx];
            uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
            uint32_t sum2 = first_sum_2 + second_sum_2;
            sum2_mem_dw[j] = sum2 + ((uint32_t)first_sum_1 << 10);
        }

        // iteration 12: 4 additions
        idx = 0;
#pragma unroll(4)
        for (int j = 0; j < 4; j++, idx += 2) {
            uint32_t first_sum_1 = sum1_mem_dw[idx];
            uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
            sum1_mem_dw[j] = (first_sum_1 + second_sum_1) % BASE;
            uint32_t first_sum_2 = sum2_mem_dw[idx];
            uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
            uint32_t sum2 = first_sum_2 + second_sum_2;
            sum2_mem_dw[j] = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 11)) % BASE;
        }

        // iteration 12: 2 additions
        idx = 0;
#pragma unroll(2)
        for (int j = 0; j < 2; j++, idx += 2) {
            uint32_t first_sum_1 = sum1_mem_dw[idx];
            uint32_t second_sum_1 = sum1_mem_dw[idx + 1];
            sum1_mem_dw[j] = (first_sum_1 + second_sum_1) % BASE;
            uint32_t first_sum_2 = sum2_mem_dw[idx];
            uint32_t second_sum_2 = sum2_mem_dw[idx + 1];
            uint32_t sum2 = first_sum_2 + second_sum_2;
            sum2_mem_dw[j] = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 12)) % BASE;
        }

        // iteration 13: 1 addition
        uint32_t first_sum_1 = sum1_mem_dw[0];
        uint32_t second_sum_1 = sum1_mem_dw[1];
        uint32_t sum1 = (first_sum_1 + second_sum_1) % BASE;
        uint32_t first_sum_2 = sum2_mem_dw[0];
        uint32_t second_sum_2 = sum2_mem_dw[1];
        uint32_t sum2 = first_sum_2 + second_sum_2;
        sum2 = (uint32_t)((uint64_t)sum2 + ((uint64_t)first_sum_1 << 13)) % BASE;

        total_sum2 = ((uint64_t)total_sum2 + ((uint64_t)sum1 << 13) + (uint64_t)sum2) % BASE;
        total_sum1 = (total_sum1 + sum1) % BASE;

        len -= BLOCK_16K;
    }

    adler32_output->return_value[0] = total_sum1 | (total_sum2 << 16);
}