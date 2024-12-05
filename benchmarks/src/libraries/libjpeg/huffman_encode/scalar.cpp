#include "huffman_encode.hpp"
#include "scalar_kernels.hpp"

void huffman_encode_scalar(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    huffman_encode_config_t *huffman_encode_config = (huffman_encode_config_t *)config;
    huffman_encode_input_t *huffman_encode_input = (huffman_encode_input_t *)input;
    huffman_encode_output_t *huffman_encode_output = (huffman_encode_output_t *)output;
    int16_t *input_addr = huffman_encode_input->input_buf;
    uint16_t *output_addr = huffman_encode_output->output_buf;
    uint32_t *zero_addr = huffman_encode_output->zero_bits;

    for (int b_id = 0; b_id < huffman_encode_config->num_blocks; b_id++) {
        int k, temp, temp2;
        uint32_t zerobits1 = 0U;
        uint32_t zerobits2 = 0U;

        for (k = 0; k < 32; k++) {
            temp = input_addr[huffman_encode_consts[k]];
            if (temp == 0) {
                /* We must apply the point transform by 4.  For AC coefficients this \
                * is an integer division with rounding towards 0.  To do this portably \
                * in C, we shift after obtaining the absolute value; so the code is \
                * interwoven with finding the abs value (temp) and output bits (temp2). \
                */
                continue;
            }
            temp2 = temp >> (8 * sizeof(int) - 1);
            temp ^= temp2;
            temp -= temp2;                              /* temp is abs value of input */
            temp >>= 4; /* apply the point transform */ /* Watch out for case that nonzero coef is zero after point transform */
            if (temp == 0)
                continue; /* For a negative coef, want temp2 = bitwise complement of abs(coef) */
            temp2 ^= temp;
            output_addr[k] = (UJCOEF)temp;
            output_addr[k + 64] = (UJCOEF)temp2;
            zerobits1 |= ((uint32_t)1U) << k;
        }

        for (k = 32; k < 64; k++) {
            temp = input_addr[huffman_encode_consts[k]];
            if (temp == 0) {
                /* We must apply the point transform by 4.  For AC coefficients this \
                * is an integer division with rounding towards 0.  To do this portably \
                * in C, we shift after obtaining the absolute value; so the code is \
                * interwoven with finding the abs value (temp) and output bits (temp2). \
                */
                continue;
            }
            temp2 = temp >> (8 * sizeof(int) - 1);
            temp ^= temp2;
            temp -= temp2;                              /* temp is abs value of input */
            temp >>= 4; /* apply the point transform */ /* Watch out for case that nonzero coef is zero after point transform */
            if (temp == 0)
                continue; /* For a negative coef, want temp2 = bitwise complement of abs(coef) */
            temp2 ^= temp;
            output_addr[k] = (UJCOEF)temp;
            output_addr[k + 64] = (UJCOEF)temp2;
            zerobits2 |= ((uint32_t)1U) << (k - 32);
        }

        zero_addr[0] = zerobits1;
        zero_addr[1] = zerobits2;

        input_addr += 64;
        output_addr += 128;
        zero_addr += 2;
    }
}