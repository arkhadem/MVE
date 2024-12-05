#include "strlen.hpp"

#include "optroutines.hpp"

#include "benchmark.hpp"

#include "init.hpp"

char *len_mve_const;

int strlen_init(size_t cache_size,
                int LANE_NUM,
                config_t *&config,
                input_t **&input,
                output_t **&output) {

    strlen_config_t *strlen_config = (strlen_config_t *)config;
    strlen_input_t **strlen_input = (strlen_input_t **)input;
    strlen_output_t **strlen_output = (strlen_output_t **)output;

    // configuration
    int size = 65536;

    init_1D<strlen_config_t>(1, strlen_config);
    strlen_config->size = size;

    // in/output versions
    size_t input_size = size * sizeof(char);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<strlen_input_t *>(count, strlen_input);
    init_1D<strlen_output_t *>(count, strlen_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<strlen_input_t>(1, strlen_input[i]);
        init_1D<strlen_output_t>(1, strlen_output[i]);

        init_1D<char>(size, strlen_input[i]->src);
        for (int j = 0; j < size; j++) {
            strlen_input[i]->src[j] = 'a';
        }
        strlen_input[i]->src[size - 1] = (char)'\0';
        init_1D<char *>(size, strlen_input[i]->src_addr);
        for (int j = 0; j < 128; j++) {
            strlen_input[i]->src_addr[j] = strlen_input[i]->src + rand_index[j];
        }
        init_1D<int>(1, strlen_output[i]->return_value);
    }

    init_1D<char>(LANE_NUM, len_mve_const);
    for (int i = 0; i < LANE_NUM / 128; i++) {
        for (int j = 0; j < 128; j++) {
            len_mve_const[i * 128 + j] = j;
        }
    }

    config = (config_t *)strlen_config;
    input = (input_t **)strlen_input;
    output = (output_t **)strlen_output;

    return count;
}