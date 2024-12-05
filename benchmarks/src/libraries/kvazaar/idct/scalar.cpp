#include "cstdint"
#include "idct.hpp"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void partial_butterfly_inverse_generic(const short *src, short *dst, int32_t shift);

void idct_scalar(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {

    idct_config_t *idct_config = (idct_config_t *)config;
    idct_input_t *idct_input = (idct_input_t *)input;
    idct_output_t *idct_output = (idct_output_t *)output;

    int count = idct_config->count;
    int8_t *bitdepth = idct_config->bitdepth;
    int16_t *in = idct_input->input;
    int16_t *out = idct_output->output;

    int16_t tmp[8 * 8];
    int32_t shift_1st = 7;
    int32_t shift_2nd = 12 - (bitdepth[0] - 8);
    for (int __i = 0; __i < count; __i++) {
        partial_butterfly_inverse_generic(in, tmp, shift_1st);
        partial_butterfly_inverse_generic(tmp, out, shift_2nd);
        in += 64;
        out += 64;
    }
}

void partial_butterfly_inverse_generic(const int16_t *src, int16_t *dst, int32_t shift) {
    int32_t j, k;
    int32_t e[4], o[4];
    int32_t ee[2], eo[2];
    int32_t add = 1 << (shift - 1);
    const int32_t line = 8;

    for (j = 0; j < line; j++) {
        // Utilizing symmetry properties to the maximum to minimize the number of multiplications
        for (k = 0; k < 4; k++) {
            o[k] = kvz_g_dct_8_s16_2D[1][k] * src[1 * line] + kvz_g_dct_8_s16_2D[3][k] * src[3 * line] +
                   kvz_g_dct_8_s16_2D[5][k] * src[5 * line] + kvz_g_dct_8_s16_2D[7][k] * src[7 * line];
        }
        eo[0] = kvz_g_dct_8_s16_2D[2][0] * src[2 * line] + kvz_g_dct_8_s16_2D[6][0] * src[6 * line];
        eo[1] = kvz_g_dct_8_s16_2D[2][1] * src[2 * line] + kvz_g_dct_8_s16_2D[6][1] * src[6 * line];
        ee[0] = kvz_g_dct_8_s16_2D[0][0] * src[0 * line] + kvz_g_dct_8_s16_2D[4][0] * src[4 * line];
        ee[1] = kvz_g_dct_8_s16_2D[0][1] * src[0 * line] + kvz_g_dct_8_s16_2D[4][1] * src[4 * line];

        // Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector
        e[0] = ee[0] + eo[0];
        e[3] = ee[0] - eo[0];
        e[1] = ee[1] + eo[1];
        e[2] = ee[1] - eo[1];

        for (k = 0; k < 4; k++) {
            dst[k] = (int16_t)MAX(-32768, MIN(32767, (e[k] + o[k] + add) >> shift));
            dst[k + 4] = (int16_t)MAX(-32768, MIN(32767, (e[3 - k] - o[3 - k] + add) >> shift));
        }
        src++;
        dst += 8;
    }
}