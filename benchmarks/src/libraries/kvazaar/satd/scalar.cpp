#include "cstdint"
#include "kvazaar.hpp"
#include "satd.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void satd_scalar(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {

    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t *satd_input = (satd_input_t *)input;
    satd_output_t *satd_output = (satd_output_t *)output;

    int count = satd_config->count;
    uint8_t *piOrg = satd_input->piOrg;
    uint8_t *piCur = satd_input->piCur;
    int32_t *result = satd_output->result;

    int32_t diff[64], m1[64], m2[64], m3[64];

    int32_t *my_result = result;

    for (int __i = 0; __i < count; __i++) {
        for (int i = 0; i < 64; i += 1) {
            diff[i] = piOrg[i] - piCur[i];
        }

        // horizontal
        for (int i = 0; i < 64; i += 8) {
            m2[i + 0] = diff[i + 0] + diff[i + 4];
            m2[i + 1] = diff[i + 1] + diff[i + 5];
            m2[i + 2] = diff[i + 2] + diff[i + 6];
            m2[i + 3] = diff[i + 3] + diff[i + 7];
            m2[i + 4] = diff[i + 0] - diff[i + 4];
            m2[i + 5] = diff[i + 1] - diff[i + 5];
            m2[i + 6] = diff[i + 2] - diff[i + 6];
            m2[i + 7] = diff[i + 3] - diff[i + 7];

            m1[i + 0] = m2[i + 0] + m2[i + 2];
            m1[i + 1] = m2[i + 1] + m2[i + 3];
            m1[i + 2] = m2[i + 0] - m2[i + 2];
            m1[i + 3] = m2[i + 1] - m2[i + 3];
            m1[i + 4] = m2[i + 4] + m2[i + 6];
            m1[i + 5] = m2[i + 5] + m2[i + 7];
            m1[i + 6] = m2[i + 4] - m2[i + 6];
            m1[i + 7] = m2[i + 5] - m2[i + 7];

            m2[i + 0] = m1[i + 0] + m1[i + 1];
            m2[i + 1] = m1[i + 0] - m1[i + 1];
            m2[i + 2] = m1[i + 2] + m1[i + 3];
            m2[i + 3] = m1[i + 2] - m1[i + 3];
            m2[i + 4] = m1[i + 4] + m1[i + 5];
            m2[i + 5] = m1[i + 4] - m1[i + 5];
            m2[i + 6] = m1[i + 6] + m1[i + 7];
            m2[i + 7] = m1[i + 6] - m1[i + 7];
        }

        // vertical
        for (int i = 0; i < 8; i++) {
            m3[0 + i] = m2[0 + i] + m2[32 + i];
            m3[8 + i] = m2[8 + i] + m2[40 + i];
            m3[16 + i] = m2[16 + i] + m2[48 + i];
            m3[24 + i] = m2[24 + i] + m2[56 + i];
            m3[32 + i] = m2[0 + i] - m2[32 + i];
            m3[40 + i] = m2[8 + i] - m2[40 + i];
            m3[48 + i] = m2[16 + i] - m2[48 + i];
            m3[56 + i] = m2[24 + i] - m2[56 + i];

            m1[0 + i] = m3[0 + i] + m3[16 + i];
            m1[8 + i] = m3[8 + i] + m3[24 + i];
            m1[16 + i] = m3[0 + i] - m3[16 + i];
            m1[24 + i] = m3[8 + i] - m3[24 + i];
            m1[32 + i] = m3[32 + i] + m3[48 + i];
            m1[40 + i] = m3[40 + i] + m3[56 + i];
            m1[48 + i] = m3[32 + i] - m3[48 + i];
            m1[56 + i] = m3[40 + i] - m3[56 + i];

            my_result[0 + i] = m1[0 + i] + m1[8 + i];
            my_result[8 + i] = m1[0 + i] - m1[8 + i];
            my_result[16 + i] = m1[16 + i] + m1[24 + i];
            my_result[24 + i] = m1[16 + i] - m1[24 + i];
            my_result[32 + i] = m1[32 + i] + m1[40 + i];
            my_result[40 + i] = m1[32 + i] - m1[40 + i];
            my_result[48 + i] = m1[48 + i] + m1[56 + i];
            my_result[56 + i] = m1[48 + i] - m1[56 + i];
        }

        piOrg += 64;
        piCur += 64;
        my_result += 64;
    }
}