#ifndef FC597AA3_5A1C_4BCC_B0A0_18F269E78F27
#define FC597AA3_5A1C_4BCC_B0A0_18F269E78F27

#include "benchmark.hpp"

void downsample_neon(int, config_t *, input_t *, output_t *);
void ycbcr_to_rgb_neon(int, config_t *, input_t *, output_t *);
void upsample_neon(int, config_t *, input_t *, output_t *);
void rgb_to_gray_neon(int, config_t *, input_t *, output_t *);
void huffman_encode_neon(int, config_t *, input_t *, output_t *);

void read_sub_neon(int, config_t *, input_t *, output_t *);
void read_up_neon(int, config_t *, input_t *, output_t *);
void expand_palette_neon(int, config_t *, input_t *, output_t *);

void sharp_update_rgb_neon(int, config_t *, input_t *, output_t *);
void sharp_filter_row_neon(int, config_t *, input_t *, output_t *);
void apply_alpha_multiply_neon(int, config_t *, input_t *, output_t *);
void dispatch_alpha_neon(int, config_t *, input_t *, output_t *);
void tm_prediction_neon(int, config_t *, input_t *, output_t *);
void vertical_filter_neon(int, config_t *, input_t *, output_t *);
void gradient_filter_neon(int, config_t *, input_t *, output_t *);

void aes_neon(int, config_t *, input_t *, output_t *);
void des_neon(int, config_t *, input_t *, output_t *);
void chacha20_neon(int, config_t *, input_t *, output_t *);

void adler32_neon(int, config_t *, input_t *, output_t *);
void crc32_neon(int, config_t *, input_t *, output_t *);

void convolve_horizontally_neon(int, config_t *, input_t *, output_t *);
void convolve_vertically_neon(int, config_t *, input_t *, output_t *);
void row_blend_neon(int, config_t *, input_t *, output_t *);
void row_opaque_neon(int, config_t *, input_t *, output_t *);

void is_audible_neon(int, config_t *, input_t *, output_t *);
void copy_with_gain_neon(int, config_t *, input_t *, output_t *);
void copy_with_sample_neon(int, config_t *, input_t *, output_t *);
void sum_from_neon(int, config_t *, input_t *, output_t *);
void handle_nan_neon(int, config_t *, input_t *, output_t *);

void memchr_neon(int, config_t *, input_t *, output_t *);
void memcmp_neon(int, config_t *, input_t *, output_t *);
void memset_neon(int, config_t *, input_t *, output_t *);
void strlen_neon(int, config_t *, input_t *, output_t *);

void fir_neon(int, config_t *, input_t *, output_t *);
void fir_lattice_neon(int, config_t *, input_t *, output_t *);
void fir_sparse_neon(int, config_t *, input_t *, output_t *);

void dct_neon(int, config_t *, input_t *, output_t *);
void idct_neon(int, config_t *, input_t *, output_t *);
void intra_neon(int, config_t *, input_t *, output_t *);
void satd_neon(int, config_t *, input_t *, output_t *);

void lpack_neon(int, config_t *, input_t *, output_t *);

#endif /* FC597AA3_5A1C_4BCC_B0A0_18F269E78F27 */
