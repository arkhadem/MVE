#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include "kvazaar.hpp"

typedef struct intra_config_s : config_t {
    int count;
    int_fast8_t log2_width;
    int_fast8_t width;
} intra_config_t;

typedef struct intra_input_s : input_t {
    kvz_pixel *ref_top;
    kvz_pixel *ref_left;
} intra_input_t;

typedef struct intra_output_s : output_t {
    kvz_pixel *dst;
} intra_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
