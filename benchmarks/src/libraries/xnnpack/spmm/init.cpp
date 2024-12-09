#include "spmm.hpp"

#include "kvazaar.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int spmm_init(size_t cache_size,
              int LANE_NUM,
              config_t *&config,
              input_t **&input,
              output_t **&output) {

    spmm_config_t *spmm_config = (spmm_config_t *)config;
    spmm_input_t **spmm_input = (spmm_input_t **)input;
    spmm_output_t **spmm_output = (spmm_output_t **)output;

    // configuration
    init_1D<spmm_config_t>(1, spmm_config);
    spmm_config->sparsity = 0.8;
    spmm_config->M = XNNPACK_M;
    spmm_config->N = XNNPACK_N;
    spmm_config->K = XNNPACK_K;
    spmm_config->min = -32768;
    spmm_config->max = 32768;
    const int weight_elements = (float)(spmm_config->N * spmm_config->K) * (1.0 - spmm_config->sparsity) + 1;
    const int bias_elements = spmm_config->N;
    const int input_elements = spmm_config->K * spmm_config->M;
    const int output_elements = spmm_config->N * spmm_config->M;
    int count = cache_size / ((weight_elements + bias_elements + input_elements + output_elements) * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<spmm_input_t *>(count, spmm_input);
    init_1D<spmm_output_t *>(count, spmm_output);

    // initializing individual versions
    int32_t *weight_matrix;
    for (int i = 0; i < count; i++) {
        init_1D<spmm_input_t>(1, spmm_input[i]);
        init_1D<spmm_output_t>(1, spmm_output[i]);

        random_init_1D<int32_t>(bias_elements, spmm_input[i]->bias);
        random_init_1D<int32_t>(input_elements, spmm_input[i]->input);
        random_init_1D<int32_t>(output_elements, spmm_output[i]->output);

        sparse_random_init_1D<int32_t>(spmm_config->N * spmm_config->K, spmm_config->sparsity, weight_matrix);
        init_1D<int32_t>(weight_elements, spmm_input[i]->weights);
        init_1D<int32_t>(weight_elements, spmm_input[i]->IDX);
        init_1D<uint32_t>(spmm_config->N + 1, spmm_input[i]->NNZ);

        uint32_t nnz = 0;
        for (uint32_t n = 0; n < spmm_config->N; n++) {
            spmm_input[i]->NNZ[n] = nnz;
            for (uint32_t k = 0; k < spmm_config->K; k++) {
                if (weight_matrix[n * spmm_config->K + k] != 0) {
                    spmm_input[i]->weights[nnz] = weight_matrix[n * spmm_config->K + k];
                    spmm_input[i]->IDX[nnz] = k;
                    nnz++;
                }
            }
        }
        spmm_input[i]->NNZ[spmm_config->N] = nnz;

        if (nnz != weight_elements) {
            fprintf(stderr, "Wrong calculation (M=%d,N=%d,K=%d) in CSR factory. Actual NNZ (%d) != Calculated NNZ (%d)!\n", spmm_config->M, spmm_config->N, spmm_config->K, nnz, weight_elements);
            exit(-1);
        }
    }

    config = (config_t *)spmm_config;
    input = (input_t **)spmm_input;
    output = (output_t **)spmm_output;

    return count;
}