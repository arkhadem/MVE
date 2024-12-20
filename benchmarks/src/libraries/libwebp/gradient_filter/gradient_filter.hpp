#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libwebp.hpp"

typedef struct gradient_filter_config_s : config_t {
    int num_rows;
    int num_cols;
    int stride;
} gradient_filter_config_t;

typedef struct gradient_filter_input_s : input_t {
    uint8_t *in;
} gradient_filter_input_t;

typedef struct gradient_filter_output_s : output_t {
    uint8_t *out;
} gradient_filter_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
