#include "cstdint"
#include "gemm.hpp"
#include "kvazaar.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

#define MR 32

void gemm_scalar(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {

    gemm_config_t *gemm_config = (gemm_config_t *)config;
    gemm_input_t *gemm_input = (gemm_input_t *)input;
    gemm_output_t *gemm_output = (gemm_output_t *)output;

    int M = gemm_config->M;
    int N = gemm_config->N;
    int K = gemm_config->K;
    int32_t min = gemm_config->min;
    int32_t max = gemm_config->max;
    int32_t *in = gemm_input->input;
    int32_t *bias = gemm_input->bias;
    int32_t *weights = gemm_input->weights;
    int32_t *out = gemm_output->output;

    for (int m = 0; m < M; m += MR) {
        int r_MR = M - m > MR ? MR : M - m;
        int32_t *weight_addr = weights;
        for (int n = 0; n < N; n++) {
            int32_t acc[MR];
            int32_t bias_data = bias[n];
            for (int mr = 0; mr < r_MR; mr++)
                acc[mr] = bias_data;
            int32_t *intput_addr = in;
            for (int k = 0; k < K; k++) {
                int32_t *input_addr_temp = intput_addr;
                for (int mr = 0; mr < r_MR; mr++) {
                    acc[mr] += (*(weight_addr) * (*input_addr_temp));
                    input_addr_temp++;
                }
                weight_addr++;
                intput_addr += M;
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