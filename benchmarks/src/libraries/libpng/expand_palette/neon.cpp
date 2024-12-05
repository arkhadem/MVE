#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <stdint.h>

#include "expand_palette.hpp"
#include "libpng.hpp"

void expand_palette_neon(int LANE_NUM,
                         config_t *config,
                         input_t *input,
                         output_t *output) {
    expand_palette_config_t *expand_palette_config = (expand_palette_config_t *)config;
    expand_palette_input_t *expand_palette_input = (expand_palette_input_t *)input;
    expand_palette_output_t *expand_palette_output = (expand_palette_output_t *)output;

    int num_rows = expand_palette_config->num_rows;
    int row_width = expand_palette_config->num_cols;
    png_uint_32 *riffled_palette = expand_palette_input->riffled_palette;

    for (int row = 0; row < num_rows; row++) {
        png_bytep dp = expand_palette_output->output_buf[row];
        png_bytep sp = expand_palette_input->input_buf[row];
        for (int i = 0; i < row_width; i += 4) {
            uint32x4_t cur;
            cur = vld1q_dup_u32(riffled_palette + *sp);
            sp++;
            cur = vld1q_lane_u32(riffled_palette + *sp, cur, 1);
            sp++;
            cur = vld1q_lane_u32(riffled_palette + *sp, cur, 2);
            sp++;
            cur = vld1q_lane_u32(riffled_palette + *sp, cur, 3);
            sp++;
            vst1q_u32((unsigned int *)dp, cur);
            dp += 16;
        }
    }
}
