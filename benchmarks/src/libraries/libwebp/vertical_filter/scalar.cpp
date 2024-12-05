#include "scalar_kernels.hpp"
#include "vertical_filter.hpp"

void vertical_filter_scalar(int LANE_NUM,
                            config_t *config,
                            input_t *input,
                            output_t *output) {
    vertical_filter_config_t *vertical_filter_config = (vertical_filter_config_t *)config;
    vertical_filter_input_t *vertical_filter_input = (vertical_filter_input_t *)input;
    vertical_filter_output_t *vertical_filter_output = (vertical_filter_output_t *)output;

    int num_rows = vertical_filter_config->num_rows;
    int num_cols = vertical_filter_config->num_cols;
    int stride = vertical_filter_config->stride;
    uint8_t *out = vertical_filter_output->out;
    uint8_t *preds = vertical_filter_input->in;
    uint8_t *in = preds + stride;
    // Filter line-by-line.
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col)
            out[col] = (uint8_t)(in[col] - preds[col]);
        preds += stride;
        in += stride;
        out += stride;
    }
}