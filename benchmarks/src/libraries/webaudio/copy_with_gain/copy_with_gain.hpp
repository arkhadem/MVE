#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdlib.h>

#include "webaudio.hpp"

typedef struct copy_with_gain_config_s : config_t {
    uint32_t data_size;
    uint32_t number_of_channels;
    float gain;
} copy_with_gain_config_t;

typedef struct copy_with_gain_input_s : input_t {
    float **sources;
} copy_with_gain_input_t;

typedef struct copy_with_gain_output_s : output_t {
    float **destinations;
} copy_with_gain_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
