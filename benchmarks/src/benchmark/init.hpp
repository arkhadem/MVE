#ifndef A01AC121_98C5_4E77_85B1_4D93BFDB2DBA
#define A01AC121_98C5_4E77_85B1_4D93BFDB2DBA
#include "benchmark.hpp"
#include "stdio.h"

#include <iostream>
#include <map>
#include <string>

extern std::map<std::string, std::map<std::string, initfunc>> init_functions;

template <typename T>
void init_1D(int count0, T *&data) {
    data = (T *)malloc(count0 * sizeof(T));
}

template <typename T>
void init_2D(int count1, int count0, T **&data) {
    init_1D<T *>(count1, data);
    for (int i = 0; i < count1; i++)
        init_1D<T>(count0, data[i]);
}

template <typename T>
void init_3D(int count2, int count1, int count0, T ***&data) {
    init_1D<T **>(count2, data);
    for (int i = 0; i < count2; i++)
        init_2D<T>(count1, count0, data[i]);
}

template <typename T>
void random_init_1D(int count0, T *&data) {
    init_1D(count0, data);
    for (int i = 0; i < count0; i++)
        data[i] = rand() % 256;
}

template <typename T>
void random_init_2D(int count1, int count0, T **&data) {
    init_1D<T *>(count1, data);
    for (int i = 0; i < count1; i++)
        random_init_1D<T>(count0, data[i]);
}

template <typename T>
void random_init_3D(int count2, int count1, int count0, T ***&data) {
    init_1D<T **>(count2, data);
    for (int i = 0; i < count2; i++)
        random_init_2D<T>(count1, count0, data[i]);
}

void register_inits();

int downsample_init(size_t, int, config_t *&, input_t **&, output_t **&);
int ycbcr_to_rgb_init(size_t, int, config_t *&, input_t **&, output_t **&);
int upsample_init(size_t, int, config_t *&, input_t **&, output_t **&);
int rgb_to_gray_init(size_t, int, config_t *&, input_t **&, output_t **&);
int huffman_encode_init(size_t, int, config_t *&, input_t **&, output_t **&);

int read_sub_init(size_t, int, config_t *&, input_t **&, output_t **&);
int read_up_init(size_t, int, config_t *&, input_t **&, output_t **&);
int expand_palette_init(size_t, int, config_t *&, input_t **&, output_t **&);

int sharp_update_rgb_init(size_t, int, config_t *&, input_t **&, output_t **&);
int sharp_filter_row_init(size_t, int, config_t *&, input_t **&, output_t **&);
int apply_alpha_multiply_init(size_t, int, config_t *&, input_t **&, output_t **&);
int dispatch_alpha_init(size_t, int, config_t *&, input_t **&, output_t **&);
int tm_prediction_init(size_t, int, config_t *&, input_t **&, output_t **&);
int vertical_filter_init(size_t, int, config_t *&, input_t **&, output_t **&);
int gradient_filter_init(size_t, int, config_t *&, input_t **&, output_t **&);

int aes_init(size_t, int, config_t *&, input_t **&, output_t **&);
int des_init(size_t, int, config_t *&, input_t **&, output_t **&);
int chacha20_init(size_t, int, config_t *&, input_t **&, output_t **&);

int adler32_init(size_t, int, config_t *&, input_t **&, output_t **&);
int crc32_init(size_t, int, config_t *&, input_t **&, output_t **&);

int convolve_horizontally_init(size_t, int, config_t *&, input_t **&, output_t **&);
int convolve_vertically_init(size_t, int, config_t *&, input_t **&, output_t **&);
int row_blend_init(size_t, int, config_t *&, input_t **&, output_t **&);
int row_opaque_init(size_t, int, config_t *&, input_t **&, output_t **&);

int is_audible_init(size_t, int, config_t *&, input_t **&, output_t **&);
int copy_with_gain_init(size_t, int, config_t *&, input_t **&, output_t **&);
int copy_with_sample_init(size_t, int, config_t *&, input_t **&, output_t **&);
int sum_from_init(size_t, int, config_t *&, input_t **&, output_t **&);
int handle_nan_init(size_t, int, config_t *&, input_t **&, output_t **&);

int memchr_init(size_t, int, config_t *&, input_t **&, output_t **&);
int memcmp_init(size_t, int, config_t *&, input_t **&, output_t **&);
int memset_init(size_t, int, config_t *&, input_t **&, output_t **&);
int strlen_init(size_t, int, config_t *&, input_t **&, output_t **&);

int fir_init(size_t, int, config_t *&, input_t **&, output_t **&);
int fir_sparse_init(size_t, int, config_t *&, input_t **&, output_t **&);
int fir_lattice_init(size_t, int, config_t *&, input_t **&, output_t **&);

int dct_init(size_t, int, config_t *&, input_t **&, output_t **&);
int idct_init(size_t, int, config_t *&, input_t **&, output_t **&);
int intra_init(size_t, int, config_t *&, input_t **&, output_t **&);
int satd_init(size_t, int, config_t *&, input_t **&, output_t **&);

int lpack_init(size_t, int, config_t *&, input_t **&, output_t **&);

#endif /* A01AC121_98C5_4E77_85B1_4D93BFDB2DBA */
