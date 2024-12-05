#ifndef FC597AA3_5A1C_4BCC_B0A0_18F269E78F27
#define FC597AA3_5A1C_4BCC_B0A0_18F269E78F27

#include "benchmark.hpp"

void downsample_scalar(int, config_t *, input_t *, output_t *);
void ycbcr_to_rgb_scalar(int, config_t *, input_t *, output_t *);
void upsample_scalar(int, config_t *, input_t *, output_t *);
void rgb_to_gray_scalar(int, config_t *, input_t *, output_t *);
void huffman_encode_scalar(int, config_t *, input_t *, output_t *);

void read_sub_scalar(int, config_t *, input_t *, output_t *);
void read_up_scalar(int, config_t *, input_t *, output_t *);
void expand_palette_scalar(int, config_t *, input_t *, output_t *);

void sharp_update_rgb_scalar(int, config_t *, input_t *, output_t *);
void sharp_filter_row_scalar(int, config_t *, input_t *, output_t *);
void apply_alpha_multiply_scalar(int, config_t *, input_t *, output_t *);
void dispatch_alpha_scalar(int, config_t *, input_t *, output_t *);
void tm_prediction_scalar(int, config_t *, input_t *, output_t *);
void vertical_filter_scalar(int, config_t *, input_t *, output_t *);
void gradient_filter_scalar(int, config_t *, input_t *, output_t *);

void aes_scalar(int, config_t *, input_t *, output_t *);
void des_scalar(int, config_t *, input_t *, output_t *);
void chacha20_scalar(int, config_t *, input_t *, output_t *);

void adler32_scalar(int, config_t *, input_t *, output_t *);
void crc32_scalar(int, config_t *, input_t *, output_t *);

void convolve_horizontally_scalar(int, config_t *, input_t *, output_t *);
void convolve_vertically_scalar(int, config_t *, input_t *, output_t *);
void row_blend_scalar(int, config_t *, input_t *, output_t *);
void row_opaque_scalar(int, config_t *, input_t *, output_t *);

void is_audible_scalar(int, config_t *, input_t *, output_t *);
void copy_with_gain_scalar(int, config_t *, input_t *, output_t *);
void copy_with_sample_scalar(int, config_t *, input_t *, output_t *);
void sum_from_scalar(int, config_t *, input_t *, output_t *);
void handle_nan_scalar(int, config_t *, input_t *, output_t *);

void memchr_scalar(int, config_t *, input_t *, output_t *);
void memcmp_scalar(int, config_t *, input_t *, output_t *);
void memset_scalar(int, config_t *, input_t *, output_t *);
void strlen_scalar(int, config_t *, input_t *, output_t *);

void fir_scalar(int, config_t *, input_t *, output_t *);
void fir_lattice_scalar(int, config_t *, input_t *, output_t *);
void fir_sparse_scalar(int, config_t *, input_t *, output_t *);

void dct_scalar(int, config_t *, input_t *, output_t *);
void idct_scalar(int, config_t *, input_t *, output_t *);
void intra_scalar(int, config_t *, input_t *, output_t *);
void satd_scalar(int, config_t *, input_t *, output_t *);

void lpack_scalar(int, config_t *, input_t *, output_t *);

#endif /* FC597AA3_5A1C_4BCC_B0A0_18F269E78F27 */
