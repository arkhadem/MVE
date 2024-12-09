#ifndef A01AC121_98C5_4E77_85B1_4D93BFDB2DBA
#define A01AC121_98C5_4E77_85B1_4D93BFDB2DBA
#include "benchmark.hpp"
#include "stdio.h"

#include <functional>
#include <iostream>
#include <map>
#include <random>
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

template <typename T>
T zero_remover(int count0, float sparsity, T inp, T min_val, T max_val) {
    if (inp != 0)
        return inp;

    // negative sparsity shows sparsity is not important to the caller
    if (sparsity < 0)
        return inp;

    // It's a dummy call, output not used at all
    if (count0 == 0)
        return inp;

    // make a value close to zero (works for floats)
    T return_val = (max_val - min_val) / (T)(count0 + 1);

    // make a value close to zero (works for integers)
    if (return_val == (T)0) {
        return_val = 1;
    }

    // if it's signed, return a positive number
    if (min_val >= 0)
        return return_val;

    // Otherwise, 50-50 chance return positive or negative
    if (rand() % 2)
        return return_val;
    return -1 * return_val;
}

template <typename T>
void sparse_init_1D(int count0, float sparsity, T *data, T min_val, T max_val) {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());

    std::vector<T> temp(count0);

    // Filling out the required zeros
    size_t sparse_end = 0;
    if (sparsity > 0)
        sparse_end = std::max(std::min(size_t(float(count0) * sparsity), size_t(count0)), size_t(0));
    std::fill(temp.begin(), temp.begin() + sparse_end, 0);

    // Filling out the rest of the non-zero values
    if ((std::is_same<T, float>::value) || ((std::is_same<T, double>::value))) {
        auto f32_rng = std::bind(std::uniform_real_distribution<float>(min_val, max_val), std::ref(rng));
        auto my_rng = std::bind(zero_remover<T>, count0, sparsity, f32_rng, min_val, max_val);
        std::generate(temp.begin() + sparse_end, temp.end(), std::ref(my_rng));
    } else {
        auto other_rng = std::bind(std::uniform_int_distribution<T>(min_val, max_val), std::ref(rng));
        auto my_rng = std::bind(zero_remover<T>, count0, sparsity, other_rng, min_val, max_val);
        std::generate(temp.begin() + sparse_end, temp.end(), std::ref(my_rng));
    }

    // Shuffling non-zeros
    std::shuffle(temp.begin(), temp.end(), rng);

    // Coppying data back to the array
    memcpy(data, temp.data(), count0 * sizeof(T));
}

template <typename T>
void sparse_init_1D(int count0, float sparsity, T *data) {
    if ((std::is_same<T, float>::value) || ((std::is_same<T, double>::value))) {
        sparse_init_1D(count0, sparsity, data, (T)-100, (T)100);
    } else if (std::is_signed<T>::value) {
        sparse_init_1D(count0, sparsity, data, (T)-50, (T)50);
    } else {
        sparse_init_1D(count0, sparsity, data, (T)0, (T)100);
    }
}

template <typename T>
void sparse_random_init_1D(int count0, float sparsity, T *&data) {
    init_1D<T>(count0, data);
    sparse_init_1D<T>(count0, sparsity, data);
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
int csum_init(size_t, int, config_t *&, input_t **&, output_t **&);

int fir_init(size_t, int, config_t *&, input_t **&, output_t **&);
int fir_sparse_init(size_t, int, config_t *&, input_t **&, output_t **&);
int fir_lattice_init(size_t, int, config_t *&, input_t **&, output_t **&);

int dct_init(size_t, int, config_t *&, input_t **&, output_t **&);
int idct_init(size_t, int, config_t *&, input_t **&, output_t **&);
int intra_init(size_t, int, config_t *&, input_t **&, output_t **&);
int satd_init(size_t, int, config_t *&, input_t **&, output_t **&);

int lpack_init(size_t, int, config_t *&, input_t **&, output_t **&);

int gemm_init(size_t, int, config_t *&, input_t **&, output_t **&);
int spmm_init(size_t, int, config_t *&, input_t **&, output_t **&);

#endif /* A01AC121_98C5_4E77_85B1_4D93BFDB2DBA */
