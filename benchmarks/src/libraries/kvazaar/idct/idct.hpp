#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

typedef struct idct_config_s : config_t {
    int count;
    int8_t *bitdepth;
} idct_config_t;

typedef struct idct_input_s : input_t {
    int16_t *input;
} idct_input_t;

typedef struct idct_output_s : output_t {
    int16_t *output;
} idct_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
