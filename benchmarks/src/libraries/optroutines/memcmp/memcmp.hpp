#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "optroutines.hpp"

typedef struct memcmp_config_s : config_t {
    int size;
} memcmp_config_t;

typedef struct memcmp_input_s : input_t {
    char *src1;
    char *src2;
    char **src1_addr;
    char **src2_addr;
} memcmp_input_t;

typedef struct memcmp_output_s : output_t {
    int *return_val;
} memcmp_output_t;

extern char *cmp_mve_const;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
