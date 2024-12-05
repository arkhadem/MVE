#include "gradient_filter.hpp"
#include "scalar_kernels.hpp"

void gradient_filter_scalar(int LANE_NUM,
                            config_t *config,
                            input_t *input,
                            output_t *output) {
    gradient_filter_config_t *gradient_filter_config = (gradient_filter_config_t *)config;
    gradient_filter_input_t *gradient_filter_input = (gradient_filter_input_t *)input;
    gradient_filter_output_t *gradient_filter_output = (gradient_filter_output_t *)output;

    int num_rows = gradient_filter_config->num_rows;
    int num_cols = gradient_filter_config->num_cols;
    int stride = gradient_filter_config->stride;
    uint8_t *in = gradient_filter_input->in + stride + 1;
    uint8_t *preds = gradient_filter_input->in + 1;
    uint8_t *out = gradient_filter_output->out;
    // Filter line-by-line.
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            uint8_t a = in[col - 1];
            uint8_t b = preds[col];
            uint8_t c = preds[col - 1];
            int16_t g = a + b - c;
            int16_t pred = ((g & ~0xff) == 0) ? g : (g < 0) ? 0
                                                            : 255;
            out[col] = (uint8_t)(in[col] - pred);
        }
        preds += stride;
        in += stride;
        out += stride;
    }
}