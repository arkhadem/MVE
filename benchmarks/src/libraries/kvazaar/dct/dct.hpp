#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

typedef struct dct_config_s : config_t {
    int count;
    int8_t *bitdepth;
} dct_config_t;

typedef struct dct_input_s : input_t {
    int16_t *input;
} dct_input_t;

typedef struct dct_output_s : output_t {
    int16_t *output;
} dct_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
