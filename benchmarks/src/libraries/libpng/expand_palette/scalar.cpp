#include "expand_palette.hpp"
#include "libpng.hpp"
#include "scalar_kernels.hpp"

void expand_palette_scalar(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    expand_palette_config_t *expand_palette_config = (expand_palette_config_t *)config;
    expand_palette_input_t *expand_palette_input = (expand_palette_input_t *)input;
    expand_palette_output_t *expand_palette_output = (expand_palette_output_t *)output;

    int num_rows = expand_palette_config->num_rows;
    int row_width = expand_palette_config->num_cols;
    png_bytep trans_alpha = expand_palette_input->a_palette;
    png_bytep palette = expand_palette_input->rgb_palette;

    for (int row = 0; row < num_rows; row++) {
        png_bytep dp = expand_palette_output->output_buf[row];
        png_bytep sp = expand_palette_input->input_buf[row];
        for (int i = 0; i < row_width; i++) {
            *dp++ = trans_alpha[*sp];
            int sp3 = *sp * 3;
            *dp++ = palette[sp3];
            *dp++ = palette[sp3 + 1];
            *dp++ = palette[sp3 + 2];
            sp++;
        }
    }
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         printf("%d ", expand_palette_output->output_buf[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}