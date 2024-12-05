#include "intra.hpp"
#include "kvazaar.hpp"
#include "neon_common_functions.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>

void intra_neon(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {

    intra_config_t *intra_config = (intra_config_t *)config;
    intra_input_t *intra_input = (intra_input_t *)input;
    intra_output_t *intra_output = (intra_output_t *)output;

    int count = intra_config->count;
    kvz_pixel *ref_top = intra_input->ref_top;
    kvz_pixel *ref_left = intra_input->ref_left;
    kvz_pixel *dst = intra_output->dst;
    const int_fast8_t log2_width = intra_config->log2_width;
    const int_fast8_t width = intra_config->width;
    assert(log2_width >= 2 && log2_width <= 5);
    const uint8_t top_right = ref_top[width + 1];
    const uint8_t bottom_left = ref_left[width + 1];
    int16x8_t v_width = vdupq_n_s16(width);
    for (int __i = 0; __i < count; __i++) {
        uint16x8_t v_top_right = vdupq_n_s16(top_right);
        uint16x8_t v_bottom_left = vdupq_n_s16(bottom_left);

        for (int y = 0; y < width; ++y) {
            int16x8_t x_plus_1 = {-7, -6, -5, -4, -3, -2, -1, 0};
            // tmp vars for valid address or type consistency
            uint16x8_t v_ref_left = vdupq_n_s16(ref_left[y + 1]);
            int16x8_t y_plus_1 = vdupq_n_s16(y + 1);

            for (int x = 0; x < width; x += 8) {
                x_plus_1 = vaddq_s16(x_plus_1, vdupq_n_s16(8));

                // NOTE: since the upper 64 bit is not used in the next instrinsic,
                // we can use load dup which loads from a memory address two both
                // 64-bit lanes of the vector register
                int64x2_t v_ref_top = vld1q_dup_s64((int64_t *)&(ref_top[x + 1]));
                int16x8_t v_ref_top_ext = vcvtepq_u8(v_ref_top);

                int16x8_t hor = vaddq_s16(vmulq_s16(vsubq_s16(v_width, x_plus_1), v_ref_left),
                                          vmulq_s16(x_plus_1, v_top_right));
                int16x8_t ver = vaddq_s16(vmulq_s16(vsubq_s16(v_width, y_plus_1), v_ref_top_ext),
                                          vmulq_s16(y_plus_1, v_bottom_left));

                int16x8_t chunk; // right shift val has to be compile-time literal
                switch (log2_width) {
                case 3: {
                    chunk = vshrq_n_s16(vaddq_s16(vaddq_s16(ver, hor), v_width), 4);
                } break;
                case 4: {
                    chunk = vshrq_n_s16(vaddq_s16(vaddq_s16(ver, hor), v_width), 5);
                } break;
                case 5: {
                    chunk = vshrq_n_s16(vaddq_s16(vaddq_s16(ver, hor), v_width), 6);
                } break;
                default: {
                    printf("unexpected log2_width: %d, program exiting...", log2_width);
                    exit(-1);
                }
                }
                chunk = vpackusq_s16(chunk, chunk);
                vst1q_lane_s64((int64_t *)&(dst[y * width + x]), chunk, 0);
            }
        }
    }
}