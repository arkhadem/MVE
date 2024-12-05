#include "mve.hpp"
#include "mve_kernels.hpp"

#include "sum_from.hpp"

#ifndef MIN_BLOCK
#define MIN_BLOCK 64
#endif

void sum_from_mve(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {
    sum_from_config_t *sum_from_config = (sum_from_config_t *)config;
    sum_from_input_t *sum_from_input = (sum_from_input_t *)input;
    sum_from_output_t *sum_from_output = (sum_from_output_t *)output;

    uint32_t data_size = sum_from_config->data_size;
    uint32_t number_of_channels = sum_from_config->number_of_channels;
    float **source1 = sum_from_input->source1;
    float **source2 = sum_from_input->source2;
    float **destination = sum_from_output->destination;

    // DIM0: a channel
    // DIM1: channels
    _mve_set_dim_count(2);

    // Loading and storing every other 32-bit
    // DIM0: sequentially
    // DIM1: random
    __vidx_var stride = {1, 0, 0, 0};

    int DIM0_TILE = data_size > LANE_NUM ? LANE_NUM : data_size;
    int DIM1_TILE = LANE_NUM / DIM0_TILE;

    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);

    int channel = 0;
    while (channel < number_of_channels) {
        int remaining_channels = number_of_channels - channel;
        remaining_channels = remaining_channels > DIM1_TILE ? DIM1_TILE : remaining_channels;
        if (remaining_channels != DIM1_TILE) {
            _mve_set_dim_length(1, remaining_channels);
        }

        int element = 0;
        while (element < data_size) {
            int remaining_elements = data_size - element;
            remaining_elements = remaining_elements > DIM0_TILE ? DIM0_TILE : remaining_elements;
            if (remaining_elements != DIM0_TILE) {
                _mve_set_dim_length(0, remaining_elements);
            }
            __mdvf src1_f = _mve_loadro_f((const float **)source1, element, stride);
            __mdvf src2_f = _mve_loadro_f((const float **)source2, element, stride);
            __mdvf dst_f = _mve_add_f(src1_f, src2_f);
            // free src1_f and src2_f
            _mve_free_f();
            _mve_free_f();
            _mve_storero_f(destination, element, dst_f, stride);
            // free dst_f
            _mve_free_f();

            element += DIM0_TILE;
        }

        source1 += DIM1_TILE;
        source2 += DIM1_TILE;
        destination += DIM1_TILE;
        channel += DIM1_TILE;
    }
}