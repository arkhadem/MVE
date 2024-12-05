#include "scalar_kernels.hpp"
#include "sum_from.hpp"

static inline void Vadd(const float *source1p,
                        int source_stride1,
                        const float *source2p,
                        int source_stride2,
                        float *dest_p,
                        int dest_stride,
                        uint32_t frames_to_process) {
    while (frames_to_process > 0u) {
        *dest_p = *source1p + *source2p;
        source1p += source_stride1;
        source2p += source_stride2;
        dest_p += dest_stride;
        --frames_to_process;
    }
}

void sum_from_scalar(int LANE_NUM,
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

    for (unsigned channel_index = 0; channel_index < number_of_channels; ++channel_index) {
        Vadd(source1[channel_index], 1, source2[channel_index], 1, destination[channel_index], 1, data_size);
    }
}