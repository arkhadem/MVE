#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include "kvazaar.hpp"

typedef struct satd_config_s : config_t {
    int count;
} satd_config_t;

typedef struct satd_input_s : input_t {
    uint8_t *piOrg;
    uint8_t *piCur;
} satd_input_t;

typedef struct satd_output_s : output_t {
    int32_t *result;
} satd_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
