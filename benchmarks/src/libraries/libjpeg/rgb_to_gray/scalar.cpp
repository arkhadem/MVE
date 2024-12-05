
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

#include "rgb_to_gray.hpp"
#include "scalar_kernels.hpp"

void rgb_to_gray_scalar(int LANE_NUM,
                        config_t *config,
                        input_t *input,
                        output_t *output) {
    rgb_to_gray_config_t *rgb_to_gray_config = (rgb_to_gray_config_t *)config;
    rgb_to_gray_input_t *rgb_to_gray_input = (rgb_to_gray_input_t *)input;
    rgb_to_gray_output_t *rgb_to_gray_output = (rgb_to_gray_output_t *)output;
    JSAMPROW outptr;
    JSAMPROW inptr;

    for (JDIMENSION row = 0; row < rgb_to_gray_config->num_rows; row++) {
        inptr = rgb_to_gray_input->input_buf[row];
        outptr = rgb_to_gray_output->output_buf[row];

        for (JDIMENSION col = 0; col < rgb_to_gray_config->num_cols; col++) {
            /* Y */
            outptr[col] = (JSAMPLE)((
                                        (uint32_t)(F_0_298 * (uint16_t)inptr[0]) +
                                        (uint32_t)(F_0_587 * (uint16_t)inptr[1]) +
                                        (uint32_t)(F_0_113 * (uint16_t)inptr[2])) >>
                                    16);
            inptr += RGB_PIXELSIZE;
        }
    }
}