#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include "kvazaar.hpp"

typedef struct lpack_config_s : config_t {
    int n;
} lpack_config_t;

typedef struct lpack_input_s : input_t {
    int32_t *da;
    int32_t *dx;
    int32_t *dyin;
} lpack_input_t;

typedef struct lpack_output_s : output_t {
    int32_t *dyout;
} lpack_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
