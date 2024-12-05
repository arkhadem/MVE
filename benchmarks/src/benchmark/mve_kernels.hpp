#ifndef C64E03B0_2C82_4453_8369_5FFFBC214EF0
#define C64E03B0_2C82_4453_8369_5FFFBC214EF0

#include "benchmark.hpp"
#include "init.hpp"

void downsample_mve(int, config_t *, input_t *, output_t *);
void ycbcr_to_rgb_mve(int, config_t *, input_t *, output_t *);
void upsample_mve(int, config_t *, input_t *, output_t *);
void rgb_to_gray_mve(int, config_t *, input_t *, output_t *);
void huffman_encode_mve(int, config_t *, input_t *, output_t *);

void read_sub_mve(int, config_t *, input_t *, output_t *);
void read_up_mve(int, config_t *, input_t *, output_t *);
void expand_palette_mve(int, config_t *, input_t *, output_t *);

void sharp_update_rgb_mve(int, config_t *, input_t *, output_t *);
void sharp_filter_row_mve(int, config_t *, input_t *, output_t *);
void apply_alpha_multiply_mve(int, config_t *, input_t *, output_t *);
void dispatch_alpha_mve(int, config_t *, input_t *, output_t *);
void tm_prediction_mve(int, config_t *, input_t *, output_t *);
void vertical_filter_mve(int, config_t *, input_t *, output_t *);
void gradient_filter_mve(int, config_t *, input_t *, output_t *);

void aes_mve(int, config_t *, input_t *, output_t *);
void des_mve(int, config_t *, input_t *, output_t *);
void chacha20_mve(int, config_t *, input_t *, output_t *);

void adler32_mve(int, config_t *, input_t *, output_t *);
void crc32_mve(int, config_t *, input_t *, output_t *);

void convolve_horizontally_mve(int, config_t *, input_t *, output_t *);
void convolve_vertically_mve(int, config_t *, input_t *, output_t *);
void row_blend_mve(int, config_t *, input_t *, output_t *);
void row_opaque_mve(int, config_t *, input_t *, output_t *);

void is_audible_mve(int, config_t *, input_t *, output_t *);
void copy_with_gain_mve(int, config_t *, input_t *, output_t *);
void copy_with_sample_mve(int, config_t *, input_t *, output_t *);
void sum_from_mve(int, config_t *, input_t *, output_t *);
void handle_nan_mve(int, config_t *, input_t *, output_t *);

void memchr_mve(int, config_t *, input_t *, output_t *);
void memcmp_mve(int, config_t *, input_t *, output_t *);
void memset_mve(int, config_t *, input_t *, output_t *);
void strlen_mve(int, config_t *, input_t *, output_t *);

void fir_mve(int, config_t *, input_t *, output_t *);
void fir_lattice_mve(int, config_t *, input_t *, output_t *);
void fir_sparse_mve(int, config_t *, input_t *, output_t *);

void dct_mve(int, config_t *, input_t *, output_t *);
void idct_mve(int, config_t *, input_t *, output_t *);
void intra_mve(int, config_t *, input_t *, output_t *);
void satd_mve(int, config_t *, input_t *, output_t *);

void lpack_mve(int, config_t *, input_t *, output_t *);

#endif /* C64E03B0_2C82_4453_8369_5FFFBC214EF0 */
