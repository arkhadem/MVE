
/* RGB -> Grayscale conversion is defined by the following equation:
 *    Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *
 * Avoid floating point arithmetic by using shifted integer constants:
 *    0.29899597 = 19595 * 2^-16
 *    0.58700561 = 38470 * 2^-16
 *    0.11399841 =  7471 * 2^-16
 * These constants are defined in jcgray-neon.c
 *
 * This is the same computation as the RGB -> Y portion of RGB -> YCbCr.
 */

#include "scalar_kernels.hpp"
#include "upsample.hpp"

void upsample_scalar(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    upsample_config_t *upsample_config = (upsample_config_t *)config;
    upsample_input_t *upsample_input = (upsample_input_t *)input;
    upsample_output_t *upsample_output = (upsample_output_t *)output;

    JSAMPROW outptr0;
    JSAMPROW outptr1;
    /* Pointers to Y (both rows), Cb, and Cr data */
    JSAMPROW inptr0_0;
    JSAMPROW inptr0_1;
    JSAMPROW inptr1;
    JSAMPROW inptr2;

    for (JDIMENSION row = 0; row < upsample_config->num_rows; row++) {
        inptr0_0 = upsample_input->input_buf[0][row * 2];
        inptr0_1 = upsample_input->input_buf[0][row * 2 + 1];
        inptr1 = upsample_input->input_buf[1][row];
        inptr2 = upsample_input->input_buf[2][row];
        outptr0 = upsample_output->output_buf[row * 2];
        outptr1 = upsample_output->output_buf[row * 2 + 1];

        for (JDIMENSION col = 0; col < upsample_config->num_cols; col += 1) {
            /* Do the chroma part of the calculation */
            JSAMPLE CB = *inptr1++;
            JSAMPLE CR = *inptr2++;
            int16_t CR_128 = CR - 128;
            int16_t CB_128 = CB - 128;

            int32_t CRed = (int32_t)((int16_t)F_1_402 * CR_128) >> 15;
            int32_t CGreen = ((int32_t)((int16_t)F_0_344 * -CB_128) + (int32_t)((int16_t)F_0_714 * -CR_128)) >> 14;
            int32_t CBlue = (int32_t)((int16_t)F_1_772 * CB_128) >> 15;
            /* Fetch 4 Y values and emit 4 pixels */
            JSAMPLE Y = *inptr0_0++;
            outptr0[RGB_RED] = (JSAMPLE)((int32_t)Y + CRed);
            outptr0[RGB_GREEN] = (JSAMPLE)((int32_t)Y + CGreen);
            outptr0[RGB_BLUE] = (JSAMPLE)((int32_t)Y + CBlue);
            outptr0 += RGB_PIXELSIZE;
            Y = *inptr0_0++;
            outptr0[RGB_RED] = (JSAMPLE)((int32_t)Y + CRed);
            outptr0[RGB_GREEN] = (JSAMPLE)((int32_t)Y + CGreen);
            outptr0[RGB_BLUE] = (JSAMPLE)((int32_t)Y + CBlue);
            outptr0 += RGB_PIXELSIZE;
            Y = *inptr0_1++;
            outptr1[RGB_RED] = (JSAMPLE)((int32_t)Y + CRed);
            outptr1[RGB_GREEN] = (JSAMPLE)((int32_t)Y + CGreen);
            outptr1[RGB_BLUE] = (JSAMPLE)((int32_t)Y + CBlue);
            outptr1 += RGB_PIXELSIZE;
            Y = *inptr0_1++;
            outptr1[RGB_RED] = (JSAMPLE)((int32_t)Y + CRed);
            outptr1[RGB_GREEN] = (JSAMPLE)((int32_t)Y + CGreen);
            outptr1[RGB_BLUE] = (JSAMPLE)((int32_t)Y + CBlue);
            outptr1 += RGB_PIXELSIZE;
        }
    }
}