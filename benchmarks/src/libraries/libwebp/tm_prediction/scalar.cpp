#include "scalar_kernels.hpp"
#include "tm_prediction.hpp"

void tm_prediction_scalar(int LANE_NUM,
                          config_t *config,
                          input_t *input,
                          output_t *output) {
    tm_prediction_config_t *tm_prediction_config = (tm_prediction_config_t *)config;
    tm_prediction_output_t *tm_prediction_output = (tm_prediction_output_t *)output;
    // tm_prediction_input_t *tm_prediction_input = (tm_prediction_input_t *)input;

    int num_blocks = tm_prediction_config->num_blocks;
    int BPS = tm_prediction_config->pic_size;
    uint8_t *VP8kclip1 = &clip1[255];

    for (int i = 0; i < num_blocks; i++) {
        uint8_t *dst = tm_prediction_output->block_dst[i];
        uint8_t *top = dst - BPS;
        uint8_t *clip0 = VP8kclip1 - top[-1];
        for (int y = 0; y < 16; ++y) {
            uint8_t *clip = clip0 + dst[-1];
            for (int x = 0; x < 16; ++x) {
                dst[x] = clip[top[x]];
            }
            dst += BPS;
        }
    }

    // for (int i = 0; i < num_blocks; i++) {
    //     uint8_t *src = tm_prediction_output->block_dst[i];
    //     int idx = src - tm_prediction_input->dst;
    //     printf("block[%d][%d] %d:\n", idx / BPS, idx % BPS, i);
    //     for (int j = 0; j < 16; j++) {
    //         for (int k = 0; k < 16; k++) {
    //             printf("%d ", src[j * BPS + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}