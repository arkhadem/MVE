#include "cstdint"
#include "intra.hpp"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void kvz_intra_pred_planar_generic(const int_fast8_t log2_width,
                                   const int_fast8_t width,
                                   const kvz_pixel *const ref_top,
                                   const kvz_pixel *const ref_left,
                                   kvz_pixel *const dst);

void intra_scalar(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {

    intra_config_t *intra_config = (intra_config_t *)config;
    intra_input_t *intra_input = (intra_input_t *)input;
    intra_output_t *intra_output = (intra_output_t *)output;

    int count = intra_config->count;
    const int_fast8_t log2_width = intra_config->log2_width;
    const int_fast8_t width = intra_config->width;
    kvz_pixel *ref_top = intra_input->ref_top;
    kvz_pixel *ref_left = intra_input->ref_left;
    kvz_pixel *dst = intra_output->dst;

    for (int i = 0; i < count; i++)
        kvz_intra_pred_planar_generic(log2_width, width, ref_top, ref_left, dst);
}

void kvz_intra_pred_planar_generic(const int_fast8_t log2_width,
                                   const int_fast8_t width,
                                   const kvz_pixel *const ref_top,
                                   const kvz_pixel *const ref_left,
                                   kvz_pixel *const dst) {
    assert(log2_width >= 2 && log2_width <= 5);

    const kvz_pixel top_right = ref_top[width + 1];
    const kvz_pixel bottom_left = ref_left[width + 1];

    int_fast16_t top[32];
    for (int i = 0; i < width; ++i) {
        top[i] = ref_top[i + 1] << log2_width;
    }

    for (int y = 0; y < width; ++y) {
        int_fast16_t hor = (ref_left[y + 1] << log2_width) + width;
        for (int x = 0; x < width; ++x) {
            hor += top_right - ref_left[y + 1];
            top[x] += bottom_left - ref_top[x + 1];
            dst[y * width + x] = (hor + top[x]) >> (log2_width + 1);
        }
    }
}