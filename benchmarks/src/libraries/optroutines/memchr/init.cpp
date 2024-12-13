#include "memchr.hpp"

#include "optroutines.hpp"

#include "benchmark.hpp"

#include "init.hpp"

char *chr_mve_const;
int rand_index[128] = {0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120, 4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122, 6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65, 33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127};

int memchr_init(size_t cache_size,
                int LANE_NUM,
                config_t *&config,
                input_t **&input,
                output_t **&output) {

    memchr_config_t *memchr_config = (memchr_config_t *)config;
    memchr_input_t **memchr_input = (memchr_input_t **)input;
    memchr_output_t **memchr_output = (memchr_output_t **)output;

    // configuration
    int size = 128 * 1024;

    init_1D<memchr_config_t>(1, memchr_config);
    memchr_config->size = size;

    // in/output versions
    size_t input_size = size * sizeof(char);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<memchr_input_t *>(count, memchr_input);
    init_1D<memchr_output_t *>(count, memchr_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<memchr_input_t>(1, memchr_input[i]);
        init_1D<memchr_output_t>(1, memchr_output[i]);

        random_init_1D<char>(1, memchr_input[i]->value);
        memchr_input[i]->value[0] = (char)129;
        init_1D<char>(size, memchr_input[i]->src);
        for (int j = 0; j < size; j++) {
            memchr_input[i]->src[j] = (char)128;
        }
        memchr_input[i]->src[size - 1] = (char)129;
        init_1D<char *>(size, memchr_input[i]->src_addr);
        for (int j = 0; j < 128; j++) {
            memchr_input[i]->src_addr[j] = memchr_input[i]->src + rand_index[j];
        }
        init_1D<char>(1, memchr_output[i]->return_value);
    }

    init_1D<char>(LANE_NUM * 256, chr_mve_const);
    for (int i = 0; i < LANE_NUM * 2; i++) {
        for (int j = 0; j < 128; j++) {
            chr_mve_const[i * 128 + j] = j;
        }
    }

    config = (config_t *)memchr_config;
    input = (input_t **)memchr_input;
    output = (output_t **)memchr_output;

    return count;
}