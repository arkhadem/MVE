#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdlib.h>

#include "libwebp.hpp"

typedef struct apply_alpha_multiply_config_s : config_t {
    int num_rows;
    int num_cols;
} apply_alpha_multiply_config_t;

typedef struct apply_alpha_multiply_input_s : input_t {
    uint8_t *rgba;
} apply_alpha_multiply_input_t;

typedef struct apply_alpha_multiply_output_s : output_t {
    int16_t dummy[1];
} apply_alpha_multiply_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
