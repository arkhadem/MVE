#include "read_up.hpp"
#include "scalar_kernels.hpp"

void read_up_scalar(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    read_up_config_t *read_up_config = (read_up_config_t *)config;
    read_up_input_t *read_up_input = (read_up_input_t *)input;
    read_up_output_t *read_up_output = (read_up_output_t *)output;

    for (int row = 0; row < read_up_config->num_rows; row++) {
        size_t i;
        size_t istop = read_up_config->num_cols;
        png_bytep rp = read_up_input->input_buf[row];
        png_bytep pp = read_up_input->prev_input_buf[row];
        png_bytep rp_out = read_up_output->output_buf[row];

        for (i = 0; i < istop; i++) {
            *rp_out = (png_byte)(((int)(*rp) + (int)(*pp++)) & 0xff);
            rp++;
            rp_out++;
        }
    }
}