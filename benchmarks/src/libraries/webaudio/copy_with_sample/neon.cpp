#include "copy_with_sample.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>

static inline void Vmul(const float *source1p,
                        int source_stride1,
                        const float *source2p,
                        int source_stride2,
                        float *dest_p,
                        int dest_stride,
                        uint32_t frames_to_process) {
    int n = frames_to_process;

    if (source_stride1 == 1 && source_stride2 == 1 && dest_stride == 1) {
        int tail_frames = n % 4;
        const float *end_p = dest_p + n - tail_frames;

        while (dest_p < end_p) {
            float32x4_t source1 = vld1q_f32(source1p);
            float32x4_t source2 = vld1q_f32(source2p);
            vst1q_f32(dest_p, vmulq_f32(source1, source2));

            source1p += 4;
            source2p += 4;
            dest_p += 4;
        }
        n = tail_frames;
    }

    while (n > 0u) {
        *dest_p = *source1p * *source2p;
        source1p += source_stride1;
        source2p += source_stride2;
        dest_p += dest_stride;
        --n;
    }
}

void copy_with_sample_neon(int LANE_NUM,
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