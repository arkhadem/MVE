#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

typedef struct fir_config_s : config_t {
    int sample_count;
    int coeff_count;
} fir_config_t;

typedef struct fir_input_s : input_t {
    int32_t *src;
    int32_t *coeff;
} fir_input_t;

typedef struct fir_output_s : output_t {
    int32_t *dst;
} fir_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
