#include "downsample.hpp"
#include "scalar_kernels.hpp"
#include <cstdio>

/* Downsample pixel values of a single component.
 * This version handles the standard case of 2:1 horizontal and 2:1 vertical,
 * without smoothing.
 */

void downsample_scalar(int LANE_NUM,
                       config_t *config,
                       input_t *input,
                       output_t *output) {
    downsample_config_t *downsample_config = (downsample_config_t *)config;
    downsample_input_t *downsample_input = (downsample_input_t *)input;
    downsample_output_t *downsample_output = (downsample_output_t *)output;
    JSAMPROW inptr0;
    JSAMPROW inptr1;
    JSAMPROW outptr;
    int bias;

    for (JDIMENSION row = 0; row < downsample_config->num_rows; row++) {
        outptr = downsample_output->output_buf[row];
        inptr0 = downsample_input->input_buf[2 * row];
        inptr1 = downsample_input->input_buf[2 * row + 1];
        bias = 1; /* bias = 1,2,1,2,... for successive samples */
        for (JDIMENSION col = 0; col < downsample_config->num_cols; col++) {
            *outptr++ = (JSAMPLE)((inptr0[0] + inptr0[1] + inptr1[0] + inptr1[1] + bias) >> 2);
            bias ^= 3; /* 1=>2, 2=>1 */
            inptr0 += 2;
            inptr1 += 2;
        }
    }
}