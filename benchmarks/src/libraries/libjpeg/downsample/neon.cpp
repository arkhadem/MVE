#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "downsample.hpp"
#include "libjpeg.hpp"

/* Downsample pixel values of a single component.
 * This version handles the standard case of 2:1 horizontal and 2:1 vertical,
 * without smoothing.
 */

void downsample_neon(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    downsample_config_t *downsample_config = (downsample_config_t *)config;
    downsample_input_t *downsample_input = (downsample_input_t *)input;
    downsample_output_t *downsample_output = (downsample_output_t *)output;

    JSAMPROW inptr0;
    JSAMPROW inptr1;
    JSAMPROW outptr;
    const uint16x8_t bias = vreinterpretq_u16_u32(vdupq_n_u32(0x00020001));

    for (JDIMENSION row = 0; row < downsample_config->num_rows; row++) {
        outptr = downsample_output->output_buf[row];
        inptr0 = downsample_input->input_buf[2 * row];
        inptr1 = downsample_input->input_buf[2 * row + 1];
        for (JDIMENSION col = 0; col < downsample_config->num_cols; col += 8) {
            uint8x16_t pixels_r0 = vld1q_u8(inptr0);
            uint8x16_t pixels_r1 = vld1q_u8(inptr1);
            /* Add adjacent pixel values in row 0, widen to 16-bit, and add bias. */
            uint16x8_t samples_u16 = vpadalq_u8(bias, pixels_r0);
            /* Add adjacent pixel values in row 1, widen to 16-bit, and accumulate. */
            samples_u16 = vpadalq_u8(samples_u16, pixels_r1);
            /* Divide total by 4 and narrow to 8-bit. */
            uint8x8_t samples_u8 = vshrn_n_u16(samples_u16, 2);
            /* Store samples to memory and increment pointers. */
            vst1_u8(outptr, samples_u8);

            inptr0 += 16;
            inptr1 += 16;
            outptr += 8;
        }
    }
}