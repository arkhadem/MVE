#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "convolve_horizontally.hpp"
#include "skia.hpp"

void convolve_horizontally_neon(int LANE_NUM,
                                config_t *config,
                                input_t *input,
                                output_t *output) {
    convolve_horizontally_config_t *convolve_horizontally_config = (convolve_horizontally_config_t *)config;
    convolve_horizontally_input_t *convolve_horizontally_input = (convolve_horizontally_input_t *)input;
    convolve_horizontally_output_t *convolve_horizontally_output = (convolve_horizontally_output_t *)output;

    // Loop over each pixel on this row in the output image.
    int num_cols = convolve_horizontally_config->num_cols;
    int num_rows = convolve_horizontally_config->num_rows;
    int filter_length = convolve_horizontally_config->filter_length;

    uint8_t *src_data = convolve_horizontally_input->src_data;
    int16_t *filter_values = convolve_horizontally_input->filter_values;
    uint8_t *out_row = convolve_horizontally_output->out_row;

    for (int out_y = 0; out_y < num_rows; out_y++) {
        uint8_t *out_row_addr = out_row;
        for (int out_x = 0; out_x < num_cols; out_x++) {

            // Compute the first pixel in this row that the filter affects. It will
            // touch |filter_length| pixels (4 bytes each) after this.
            const unsigned char *row_to_filter = &src_data[out_x * 4];

            int16_t *filter_values_addr = filter_values;

            // Apply the filter to the row to get the destination pixel in |accum|.
            int32x4_t accum = vdupq_n_s32(0);
            for (int filter_x = 0; filter_x < (filter_length / 4); filter_x++) {
                // Load 4 coefficients.
                int16x4_t coeffs = vld1_s16(filter_values_addr);
                // Load 4 pixels into a q-register.
                uint8x16_t pixels = vld1q_u8(row_to_filter);

                // Expand to 16-bit channels split across two q-registers.
                int16x8_t p01_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels)));
                int16x8_t p23_16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pixels)));

                // Scale each pixel (each d-register) by its filter coefficients,
                // accumulating into 32-bit.
                accum = vmlal_lane_s16(accum, vget_low_s16(p01_16), coeffs, 0);
                accum = vmlal_lane_s16(accum, vget_high_s16(p01_16), coeffs, 1);
                accum = vmlal_lane_s16(accum, vget_low_s16(p23_16), coeffs, 2);
                accum = vmlal_lane_s16(accum, vget_high_s16(p23_16), coeffs, 3);

                // Advance to next elements.
                row_to_filter += 16;
                filter_values_addr += 4;
            }

            // Bring this value back in range. All of the filter scaling factors
            // are in fixed point with kShiftBits bits of fractional part.
            int16x4_t accum16 = vqshrn_n_s32(accum, 2);

            // Pack and store the new pixel.
            uint8x8_t accum8 = vqmovun_s16(vcombine_s16(accum16, accum16));
            vst1_lane_u32(reinterpret_cast<uint32_t *>(out_row_addr),
                          vreinterpret_u32_u8(accum8), 0);
            out_row_addr += 4;
        }
        src_data += (num_cols + filter_length) * 4;
        out_row += num_cols * 4;
    }
}
