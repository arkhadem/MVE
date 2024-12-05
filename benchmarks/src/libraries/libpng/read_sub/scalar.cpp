#include "read_sub.hpp"
#include "scalar_kernels.hpp"

void read_sub_scalar(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    read_sub_config_t *read_sub_config = (read_sub_config_t *)config;
    read_sub_input_t *read_sub_input = (read_sub_input_t *)input;
    read_sub_output_t *read_sub_output = (read_sub_output_t *)output;

    for (int row = 0; row < read_sub_config->num_rows; row++) {
        size_t i;
        size_t istop = read_sub_config->num_cols;
        png_bytep rp = read_sub_input->input_buf[row];
        png_bytep rp_out = read_sub_output->output_buf[row];

        rp_out[0] = rp[0];
        rp_out[1] = rp[1];
        rp_out[2] = rp[2];
        rp_out[3] = rp[3];

        rp += 4;
        rp_out += 4;

        for (i = 4; i < istop; i++) {
            *rp_out = (png_byte)(((int)(*rp) + (int)(*(rp_out - 4))) & 0xff);
            rp++;
            rp_out++;
        }
    }
}