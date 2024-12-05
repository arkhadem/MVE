#include "mve.hpp"
#include "mve_kernels.hpp"

#include "is_audible.hpp"

#ifndef MIN_BLOCK
#define MIN_BLOCK 64
#endif

void is_audible_mve(int LANE_NUM,
                    config_t *config,
                    input_t *input,
                    output_t *output) {
    is_audible_config_t *is_audible_config = (is_audible_config_t *)config;
    is_audible_input_t *is_audible_input = (is_audible_input_t *)input;
    is_audible_output_t *is_audible_output = (is_audible_output_t *)output;

    uint32_t data_size = is_audible_config->data_size;
    uint32_t number_of_channels = is_audible_config->number_of_channels;
    float **data = is_audible_input->data;

    // DIM0: a channel
    // DIM1: channels
    _mve_set_dim_count(2);

    // Loading and storing every other 32-bit
    // DIM0: sequentially
    // DIM1: random
    __vidx_var stride = {2, 2, 0, 0};

    int DIM1_TILE = number_of_channels;
    int DIM0_TILE = LANE_NUM / number_of_channels;

    __mdvf energy_f = _mve_set1_f(0);

    _mve_set_dim_length(0, DIM0_TILE);
    _mve_set_dim_length(1, DIM1_TILE);

    float memory[8192];

    int element = 0;
    while (element < data_size) {
        int remaining_elements = data_size - element;
        remaining_elements = remaining_elements > DIM0_TILE ? DIM0_TILE : remaining_elements;
        if (remaining_elements != DIM0_TILE) {
            _mve_set_dim_length(0, remaining_elements);
        }
        __mdvf data_f = _mve_loadro_f((const float **)data, element, stride);
        energy_f = _mve_add_f(energy_f, data_f);
        // free energy_f and data_f
        _mve_free_f();
        _mve_free_f();

        element += DIM0_TILE;
    }

    int current_elements = LANE_NUM >> 1;
    _mve_set_dim_length(1, 1);
    _mve_set_dim_length(0, current_elements);
    _mve_set_dim_length(1, 2);

    _mve_unset_only_element(1, 1);
    _mve_store_f(memory, energy_f, stride);
    _mve_set_all_elements(1);

    while (true) {
        _mve_set_dim_length(0, current_elements >> 1);
        __mdvf temp_energy_f = _mve_load_f(memory + current_elements, stride);
        energy_f = _mve_add_f(energy_f, temp_energy_f);
        // free energy_f and temp_energy_f
        _mve_free_f();
        _mve_free_f();
        if (current_elements == MIN_BLOCK) {
            _mve_store_f(memory, energy_f, stride);
            // free energy_f
            _mve_free_f();
            break;
        } else {
            _mve_unset_only_element(1, 1);
            _mve_store_f(memory, energy_f, stride);
            _mve_set_all_elements(1);
        }
        current_elements >>= 1;
    }

    float energy = 0;
#pragma unroll 16
    for (int i = 0; i < MIN_BLOCK; i++) {
        energy += memory[i];
    }

    is_audible_output->return_value[0] = energy > 0;
}