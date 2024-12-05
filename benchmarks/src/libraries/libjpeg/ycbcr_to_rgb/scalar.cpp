/* Notes on safe memory access for YCbCr -> RGB conversion routines:
 *
 * Input memory buffers can be safely overread up to the next multiple of
 * ALIGN_SIZE bytes, since they are always allocated by alloc_sarray() in
 * jmemmgr.c.
 *
 * The output buffer cannot safely be written beyond output_width, since
 * output_buf points to a possibly unpadded row in the decompressed image
 * buffer allocated by the calling program.
 */

#include "scalar_kernels.hpp"
#include "ycbcr_to_rgb.hpp"

void ycbcr_to_rgb_scalar(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    ycbcr_to_rgb_config_t *ycbcr_to_rgb_config = (ycbcr_to_rgb_config_t *)config;
    ycbcr_to_rgb_input_t *ycbcr_to_rgb_input = (ycbcr_to_rgb_input_t *)input;
    ycbcr_to_rgb_output_t *ycbcr_to_rgb_output = (ycbcr_to_rgb_output_t *)output;
    JSAMPROW outptr;
    JSAMPROW inptr0;
    JSAMPROW inptr1;
    JSAMPROW inptr2;

    for (JDIMENSION row = 0; row < ycbcr_to_rgb_config->num_rows; row++) {
        inptr0 = ycbcr_to_rgb_input->input_buf[0][row];
        inptr1 = ycbcr_to_rgb_input->input_buf[1][row];
        inptr2 = ycbcr_to_rgb_input->input_buf[2][row];
        outptr = ycbcr_to_rgb_output->output_buf[row];

        for (JDIMENSION col = 0; col < ycbcr_to_rgb_config->num_cols; col++) {
            JSAMPLE Y = inptr0[col];
            JSAMPLE CB = inptr1[col];
            JSAMPLE CR = inptr2[col];
            int16_t CR_128 = CR - 128;
            int16_t CB_128 = CB - 128;
            // R
            outptr[0] = (JSAMPLE)((int32_t)Y + ((int32_t)((int16_t)F_1_402 * CR_128) >> 15));
            // G
            outptr[1] = (JSAMPLE)((int32_t)Y + (((int32_t)((int16_t)F_0_344 * -CB_128) + (int32_t)((int16_t)F_0_714 * -CR_128)) >> 14));
            // B
            outptr[2] = (JSAMPLE)((int32_t)Y + ((int32_t)((int16_t)F_1_772 * CB_128) >> 15));
            outptr += RGB_PIXELSIZE;
        }
    }
}