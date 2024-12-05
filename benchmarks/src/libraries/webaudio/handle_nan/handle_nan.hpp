#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdlib.h>

#include "webaudio.hpp"

typedef struct handle_nan_config_s : config_t {
    unsigned number_of_values;
    float default_value;
} handle_nan_config_t;

typedef struct handle_nan_input_s : input_t {
    float *values;
} handle_nan_input_t;

typedef struct handle_nan_output_s : output_t {
    float dummy[1];
} handle_nan_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
