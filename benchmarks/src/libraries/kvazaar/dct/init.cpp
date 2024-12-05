#include "dct.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int dct_init(size_t cache_size,
             int LANE_NUM,
             config_t *&config,
             input_t **&input,
             output_t **&output) {

    dct_config_t *dct_config = (dct_config_t *)config;
    dct_input_t **dct_input = (dct_input_t **)input;
    dct_output_t **dct_output = (dct_output_t **)output;

    // configuration
    init_1D<dct_config_t>(1, dct_config);
    dct_config->count = 8192;
    init_1D<int8_t>(dct_config->count, dct_config->bitdepth);
    for (int i = 0; i < dct_config->count; i++) {
        dct_config->bitdepth[i] = 12;
    }
    int count = cache_size / (dct_config->count * 8 * 8 * 2 * sizeof(int16_t)) + 1;

    // initializing in/output
    init_1D<dct_input_t *>(count, dct_input);
    init_1D<dct_output_t *>(count, dct_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<dct_input_t>(count, dct_input[i]);
        init_1D<dct_output_t>(count, dct_output[i]);

        random_init_1D<int16_t>(dct_config->count * 8 * 8, dct_input[i]->input);
        random_init_1D<int16_t>(dct_config->count * 8 * 8, dct_output[i]->output);
    }

    config = (config_t *)dct_config;
    input = (input_t **)dct_input;
    output = (output_t **)dct_output;

    return count;
}

const int8_t kvz_g_convert_to_bit[LCU_WIDTH + 1] = {
    -1, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 2,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4};

const int16_t kvz_g_dct_8_s16_2D[8][8] = {
    {64, 64, 64, 64, 64, 64, 64, 64},
    {89, 75, 50, 18, -18, -50, -75, -89},
    {83, 36, -36, -83, -83, -36, 36, 83},
    {75, -18, -89, -50, 50, 89, 18, -75},
    {64, -64, -64, 64, 64, -64, -64, 64},
    {50, -89, 18, 75, -75, -18, 89, -50},
    {36, -83, 83, -36, -36, 83, -83, 36},
    {18, -50, 75, -89, 89, -75, 50, -18}};

const int16_t kvz_g_dct_8_s16_1D[8 * 8] = {64, 64, 64, 64, 64, 64, 64, 64, 89, 75, 50, 18, -18, -50, -75, -89, 83, 36, -36, -83, -83, -36, 36, 83, 75, -18, -89, -50, 50, 89, 18, -75, 64, -64, -64, 64, 64, -64, -64, 64, 50, -89, 18, 75, -75, -18, 89, -50, 36, -83, 83, -36, -36, 83, -83, 36, 18, -50, 75, -89, 89, -75, 50, -18};

const int32_t kvz_g_dct_4_s32_2D[8][4] = {
    {64, 64, 64, 64},
    {89, 75, 50, 18},
    {83, 36, -36, -83},
    {75, -18, -89, -50},
    {64, -64, -64, 64},
    {50, -89, 18, 75},
    {36, -83, 83, -36},
    {18, -50, 75, -89}};
