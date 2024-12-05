#include "scalar_kernels.hpp"
#include "sharp_filter_row.hpp"

static uint16_t clip(int v, int max) {
    return (v < 0) ? 0 : (v > max) ? max
                                   : (uint16_t)v;
}

void sharp_filter_row_scalar(int LANE_NUM,
                             config_t *config,
                             input_t *input,
                             output_t *output) {
    sharp_filter_row_config_t *sharp_filter_row_config = (sharp_filter_row_config_t *)config;
    sharp_filter_row_input_t *sharp_filter_row_input = (sharp_filter_row_input_t *)input;
    sharp_filter_row_output_t *sharp_filter_row_output = (sharp_filter_row_output_t *)output;

    const int max_y = (1 << sharp_filter_row_config->bit_depth) - 1;

    int16_t *A = sharp_filter_row_input->A;
    int16_t *B = sharp_filter_row_input->B;
    uint16_t *best_y = sharp_filter_row_input->best_y;
    uint16_t *out = sharp_filter_row_output->out;

    for (int row = 0; row < sharp_filter_row_config->num_rows; row++) {
        for (int i = 0; i < sharp_filter_row_config->num_cols; ++i, ++A, ++B) {
            const int v0 = (A[0] * 9 + A[1] * 3 + B[0] * 3 + B[1] + 8) >> 4;
            const int v1 = (A[1] * 9 + A[0] * 3 + B[1] * 3 + B[0] + 8) >> 4;
            out[2 * i + 0] = clip(best_y[2 * i + 0] + v0, max_y);
            out[2 * i + 1] = clip(best_y[2 * i + 1] + v1, max_y);
        }
        best_y += 2 * sharp_filter_row_config->num_cols;
        out += 2 * sharp_filter_row_config->num_cols;
    }
}