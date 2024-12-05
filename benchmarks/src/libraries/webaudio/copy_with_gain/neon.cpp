#include "copy_with_gain.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>

static inline void Vsmul(const float *source_p,
                         int source_stride,
                         const float *scale,
                         float *dest_p,
                         int dest_stride,
                         uint32_t frames_to_process) {
    int n = frames_to_process;

    if (source_stride == 1 && dest_stride == 1) {
        float k = *scale;
        int tail_frames = n % 4;
        const float *end_p = dest_p + n - tail_frames;

        while (dest_p < end_p) {
            float32x4_t source = vld1q_f32(source_p);
            vst1q_f32(dest_p, vmulq_n_f32(source, k));

            source_p += 4;
            dest_p += 4;
        }
        n = tail_frames;
    }

    const float k = *scale;
    while (n > 0u) {
        *dest_p = k * *source_p;
        source_p += source_stride;
        dest_p += dest_stride;
        --n;
    }
}

void copy_with_gain_neon(int LANE_NUM,
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