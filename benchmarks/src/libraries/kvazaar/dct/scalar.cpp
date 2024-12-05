#include "cstdint"
#include "dct.hpp"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void partial_butterfly_8_generic(const short *src, short *dst, int32_t shift);

void dct_scalar(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t *dct_input = (dct_input_t *)input;
    dct_output_t *dct_output = (dct_output_t *)output;

    int count = dct_config->count;
    int8_t *bitdepth = dct_config->bitdepth;
    int16_t *in = dct_input->input;
    int16_t *out = dct_output->output;

    int16_t tmp[8 * 8]; // shift_1st = 2, shift_2nd = 9
    int32_t shift_1st = kvz_g_convert_to_bit[8] + 1 + (bitdepth[0] - 8);
    int32_t shift_2nd = kvz_g_convert_to_bit[8] + 8;
    for (int __i = 0; __i < count; __i++) {
        partial_butterfly_8_generic(in, tmp, shift_1st);
        partial_butterfly_8_generic(tmp, out, shift_2nd);
        in += 64;
        out += 64;
    }
}

void partial_butterfly_8_generic(const short *src, short *dst, int32_t shift) {
    int32_t j, k;
    // int32_t e[4], o[4];
    int32_t e[4], o[4];
    int32_t ee[2], eo[2];
    int32_t add = 1 << (shift - 1);
    const int32_t line = 8;

    for (j = 0; j < line; j++) {
        // E and O
        for (k = 0; k < 4; k++) {
            e[k] = src[k] + src[7 - k];
            o[k] = src[k] - src[7 - k];
        }
        // EE and EO
        ee[0] = e[0] + e[3];
        eo[0] = e[0] - e[3];
        ee[1] = e[1] + e[2];
        eo[1] = e[1] - e[2];

        dst[0 * line] = (short)((kvz_g_dct_8_s16_2D[0][0] * ee[0] +
                                 kvz_g_dct_8_s16_2D[0][1] * ee[1] + add) >>
                                shift);
        dst[4 * line] = (short)((kvz_g_dct_8_s16_2D[4][0] * ee[0] +
                                 kvz_g_dct_8_s16_2D[4][1] * ee[1] + add) >>
                                shift);
        dst[2 * line] = (short)((kvz_g_dct_8_s16_2D[2][0] * eo[0] +
                                 kvz_g_dct_8_s16_2D[2][1] * eo[1] + add) >>
                                shift);
        dst[6 * line] = (short)((kvz_g_dct_8_s16_2D[6][0] * eo[0] +
                                 kvz_g_dct_8_s16_2D[6][1] * eo[1] + add) >>
                                shift);

        dst[line] = (short)((kvz_g_dct_8_s16_2D[1][0] * o[0] +
                             kvz_g_dct_8_s16_2D[1][1] * o[1] +
                             kvz_g_dct_8_s16_2D[1][2] * o[2] +
                             kvz_g_dct_8_s16_2D[1][3] * o[3] + add) >>
                            shift);
        dst[3 * line] = (short)((kvz_g_dct_8_s16_2D[3][0] * o[0] +
                                 kvz_g_dct_8_s16_2D[3][1] * o[1] +
                                 kvz_g_dct_8_s16_2D[3][2] * o[2] +
                                 kvz_g_dct_8_s16_2D[3][3] * o[3] + add) >>
                                shift);
        dst[5 * line] = (short)((kvz_g_dct_8_s16_2D[5][0] * o[0] +
                                 kvz_g_dct_8_s16_2D[5][1] * o[1] +
                                 kvz_g_dct_8_s16_2D[5][2] * o[2] +
                                 kvz_g_dct_8_s16_2D[5][3] * o[3] + add) >>
                                shift);
        dst[7 * line] = (short)((kvz_g_dct_8_s16_2D[7][0] * o[0] +
                                 kvz_g_dct_8_s16_2D[7][1] * o[1] +
                                 kvz_g_dct_8_s16_2D[7][2] * o[2] +
                                 kvz_g_dct_8_s16_2D[7][3] * o[3] + add) >>
                                shift);

        src += 8;
        dst++;
    }
}