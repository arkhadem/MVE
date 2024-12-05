#ifndef BCD754D4_9604_4C5E_A1C0_5B4EE0416E80
#define BCD754D4_9604_4C5E_A1C0_5B4EE0416E80

#include <stdint.h>
#include <stdlib.h>

#include "libpng.hpp"

typedef struct expand_palette_config_s : config_t {
    int num_rows;
    int num_cols;
} expand_palette_config_t;

typedef struct expand_palette_input_s : input_t {
    png_byte **input_buf;
    // For NEON
    png_uint_32 *riffled_palette;
    // For Scalar + a_palette
    png_byte *rgb_palette;
    // For MVE
    png_byte *r_palette;
    png_byte *g_palette;
    png_byte *b_palette;
    png_byte *a_palette;
} expand_palette_input_t;

typedef struct expand_palette_output_s : output_t {
    png_byte **output_buf;
} expand_palette_output_t;

#endif /* BCD754D4_9604_4C5E_A1C0_5B4EE0416E80 */
