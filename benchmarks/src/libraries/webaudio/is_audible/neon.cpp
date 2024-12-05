#include "is_audible.hpp"
#include "neon_kernels.hpp"
#include <arm_neon.h>

static inline void Vsvesq(const float *source_p,
                          int source_stride,
                          float *sum_p,
                          uint32_t frames_to_process) {
    int n = frames_to_process;

    if (source_stride == 1) {
        int tail_frames = n % 4;
        const float *end_p = source_p + n - tail_frames;

        float32x4_t four_sum = vdupq_n_f32(0);
        while (source_p < end_p) {
            float32x4_t source = vld1q_f32(source_p);
            four_sum = vmlaq_f32(four_sum, source, source);
            source_p += 4;
        }
        float32x2_t two_sum =
            vadd_f32(vget_low_f32(four_sum), vget_high_f32(four_sum));

        float group_sum[2];
        vst1_f32(group_sum, two_sum);
        *sum_p += group_sum[0] + group_sum[1];

        n = tail_frames;
    }

    while (n > 0u) {
        const float sample = *source_p;
        *sum_p += sample * sample;
        source_p += source_stride;
        --n;
    }
}

void is_audible_neon(int LANE_NUM,
                     config_t *config,
                     input_t *input,
                     output_t *output) {
    is_audible_config_t *is_audible_config = (is_audible_config_t *)config;
    is_audible_input_t *is_audible_input = (is_audible_input_t *)input;
    is_audible_output_t *is_audible_output = (is_audible_output_t *)output;
    // Compute the energy in each channel and sum up the energy in each channel
    // for the total energy.
    float energy = 0;

    uint32_t data_size = is_audible_config->data_size;
    uint32_t number_of_channels = is_audible_config->number_of_channels;
    float **data = is_audible_input->data;

    for (uint32_t k = 0; k < number_of_channels; ++k) {
        const float *my_data = data[k];
        float channel_energy;
        Vsvesq(my_data, 1, &channel_energy, data_size);
        energy += channel_energy;
    }

    is_audible_output->return_value[0] = energy > 0;
}