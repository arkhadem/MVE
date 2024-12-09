#include "cstdint"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include "spmm.hpp"
#include <cstdint>

#define MR 32

void spmm_scalar(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {

    spmm_config_t *spmm_config = (spmm_config_t *)config;
    spmm_input_t *spmm_input = (spmm_input_t *)input;
    spmm_output_t *spmm_output = (spmm_output_t *)output;

    int M = spmm_config->M;
    int N = spmm_config->N;
    int K = spmm_config->K;
    int32_t min = spmm_config->min;
    int32_t max = spmm_config->max;
    int32_t *in = spmm_input->input;
    int32_t *bias = spmm_input->bias;
    int32_t *weights = spmm_input->weights;
    int32_t *IDX = spmm_input->IDX;
    uint32_t *NNZ = spmm_input->NNZ;
    int32_t *out = spmm_output->output;

    for (int m = 0; m < M; m += MR) {
        int r_MR = M - m > MR ? MR : M - m;
        for (int n = 0; n < N; n++) {
            int32_t acc[MR];
            int32_t bias_data = bias[n];
            for (int mr = 0; mr < r_MR; mr++)
                acc[mr] = bias_data;
            for (int k = NNZ[n]; k < NNZ[n + 1]; k++) {
                int32_t weight_data = weights[k];
                int32_t *intput_addr = in + IDX[k] * M;
                for (int mr = 0; mr < r_MR; mr++) {
                    acc[mr] += (weight_data * (*intput_addr));
                    intput_addr++;
                }
            }
            int32_t *output_addr = out + n * M;
            for (int mr = 0; mr < r_MR; mr++) {
                acc[mr] = acc[mr] > max ? max : acc[mr];
                acc[mr] = acc[mr] < min ? min : acc[mr];
                *output_addr = acc[mr];
                output_addr++;
            }
        }
        out += MR;
        in += MR;
    }
}