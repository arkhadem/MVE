#include "copy_with_sample.hpp"
#include "scalar_kernels.hpp"

static inline void Vmul(const float *source1p,
                        int source_stride1,
                        const float *source2p,
                        int source_stride2,
                        float *dest_p,
                        int dest_stride,
                        uint32_t frames_to_process) {
    while (frames_to_process > 0u) {
        *dest_p = *source1p * *source2p;
        source1p += source_stride1;
        source2p += source_stride2;
        dest_p += dest_stride;
        --frames_to_process;
    }
}

void copy_with_sample_scalar(int LANE_NUM,
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

    for (unsigned channel_index = 0; channel_index < number_of_channels; ++channel_index) {
        Vmul(sources[channel_index], 1, gain, 1, destinations[channel_index], 1, data_size);
    }
}