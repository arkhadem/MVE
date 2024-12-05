#include "copy_with_gain.hpp"
#include "scalar_kernels.hpp"

static inline void Vsmul(const float *source_p,
                         int source_stride,
                         const float *scale,
                         float *dest_p,
                         int dest_stride,
                         uint32_t frames_to_process) {
    const float k = *scale;
    while (frames_to_process > 0u) {
        *dest_p = k * *source_p;
        source_p += source_stride;
        dest_p += dest_stride;
        --frames_to_process;
    }
}

void copy_with_gain_scalar(int LANE_NUM,
                           config_t *config,
                           input_t *input,
                           output_t *output) {
    copy_with_gain_config_t *copy_with_gain_config = (copy_with_gain_config_t *)config;
    copy_with_gain_input_t *copy_with_gain_input = (copy_with_gain_input_t *)input;
    copy_with_gain_output_t *copy_with_gain_output = (copy_with_gain_output_t *)output;

    uint32_t data_size = copy_with_gain_config->data_size;
    uint32_t number_of_channels = copy_with_gain_config->number_of_channels;
    float gain = copy_with_gain_config->gain;
    float **sources = copy_with_gain_input->sources;
    float **destinations = copy_with_gain_output->destinations;

    for (unsigned channel_index = 0; channel_index < number_of_channels; ++channel_index) {
        Vsmul(sources[channel_index], 1, &gain, destinations[channel_index], 1, data_size);
    }
}