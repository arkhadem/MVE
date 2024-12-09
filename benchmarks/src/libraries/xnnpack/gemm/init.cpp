#include "gemm.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int gemm_init(size_t cache_size,
              int LANE_NUM,
              config_t *&config,
              input_t **&input,
              output_t **&output) {

    gemm_config_t *gemm_config = (gemm_config_t *)config;
    gemm_input_t **gemm_input = (gemm_input_t **)input;
    gemm_output_t **gemm_output = (gemm_output_t **)output;

    // configuration
    init_1D<gemm_config_t>(1, gemm_config);
    gemm_config->M = XNNPACK_M;
    gemm_config->N = XNNPACK_N;
    gemm_config->K = XNNPACK_K;
    gemm_config->min = -32768;
    gemm_config->max = 32768;
    const int weight_elements = gemm_config->N * gemm_config->K;
    const int bias_elements = gemm_config->N;
    const int input_elements = gemm_config->K * gemm_config->M;
    const int output_elements = gemm_config->N * gemm_config->M;
    int count = cache_size / ((weight_elements + bias_elements + input_elements + output_elements) * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<gemm_input_t *>(count, gemm_input);
    init_1D<gemm_output_t *>(count, gemm_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<gemm_input_t>(1, gemm_input[i]);
        init_1D<gemm_output_t>(1, gemm_output[i]);

        random_init_1D<int32_t>(weight_elements, gemm_input[i]->weights);
        random_init_1D<int32_t>(bias_elements, gemm_input[i]->bias);
        random_init_1D<int32_t>(input_elements, gemm_input[i]->input);
        random_init_1D<int32_t>(output_elements, gemm_output[i]->output);
    }

    config = (config_t *)gemm_config;
    input = (input_t **)gemm_input;
    output = (output_t **)gemm_output;

    return count;
}