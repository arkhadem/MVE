#include "mve.hpp"
#include "mve_kernels.hpp"

#include "copy_with_sample.hpp"

#ifndef MIN_BLOCK
#define MIN_BLOCK 64
#endif

void copy_with_sample_mve(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    copy_with_sample_config_t *copy_with_sample_config = (copy_with_sample_config_t *)config;
    copy_with_sample_input_t *copy_with_sample_input = (copy_with_sample_input_t *)input;
    copy_with_sample_output_t *copy_with_sample_output = (copy_with_sample_output_t *)output;

    uint32_t data_size = copy_with_sample_config->data_size;
    uint32_t number_of_channels = copy_with_sample_config->number_of_channels;
    float *gain = copy_with_sample_input->gain;
    float **sources = copy_with_sample_input->sources;
    float **destinations = copy_with_sample_output->destinations;

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

    __mdvf gain_f = _mve_load_f(gain, stride);

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
            __mdvf src_f = _mve_loadro_f((const float **)sources, element, stride);
            __mdvf dst_f = _mve_mul_f(gain_f, src_f);
            // free src_f
            _mve_free_f();
            _mve_storero_f(destinations, element, dst_f, stride);
            // free dst_f
            _mve_free_f();

            element += DIM0_TILE;
        }

        sources += DIM1_TILE;
        destinations += DIM1_TILE;
        channel += DIM1_TILE;
    }

    // free gain_f
    _mve_free_f();
}