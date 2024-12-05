#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libwebp.hpp"

extern uint8_t clip1[255 + 511 + 1];

typedef struct tm_prediction_config_s : config_t {
    int num_blocks;
    int pic_size;
} tm_prediction_config_t;

typedef struct tm_prediction_input_s : input_t {
    uint8_t *dst;
} tm_prediction_input_t;

typedef struct tm_prediction_output_s : output_t {
    uint8_t **block_dst;
} tm_prediction_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
