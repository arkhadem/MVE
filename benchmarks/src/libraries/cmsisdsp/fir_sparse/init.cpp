#include "fir_sparse.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int fir_sparse_init(size_t cache_size,
                    int LANE_NUM,
                    config_t *&config,
                    input_t **&input,
                    output_t **&output) {

    fir_sparse_config_t *fir_sparse_config = (fir_sparse_config_t *)config;
    fir_sparse_input_t **fir_sparse_input = (fir_sparse_input_t **)input;
    fir_sparse_output_t **fir_sparse_output = (fir_sparse_output_t **)output;

    // configuration
    init_1D<fir_sparse_config_t>(1, fir_sparse_config);
    fir_sparse_config->sample_count = 192 * 1024;
    fir_sparse_config->coeff_count = 32;
    fir_sparse_config->sparsity = 0.8;
    fir_sparse_config->input_count = fir_sparse_config->sample_count + fir_sparse_config->coeff_count - 1;
    fir_sparse_config->effective_coeff_count = (int)(float(fir_sparse_config->coeff_count) * (1.00 - fir_sparse_config->sparsity));
    int count = cache_size / ((fir_sparse_config->input_count + fir_sparse_config->effective_coeff_count * 2 + fir_sparse_config->sample_count) * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<fir_sparse_input_t *>(count, fir_sparse_input);
    init_1D<fir_sparse_output_t *>(count, fir_sparse_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<fir_sparse_input_t>(1, fir_sparse_input[i]);
        init_1D<fir_sparse_output_t>(1, fir_sparse_output[i]);

        random_init_1D<int32_t>(fir_sparse_config->effective_coeff_count, fir_sparse_input[i]->delay);
        for (int j = 0; j < fir_sparse_config->effective_coeff_count; j++) {
            fir_sparse_input[i]->delay[j] %= fir_sparse_config->coeff_count;
        }
        random_init_1D<int32_t>(fir_sparse_config->input_count, fir_sparse_input[i]->src);
        random_init_1D<int32_t>(fir_sparse_config->coeff_count, fir_sparse_input[i]->coeff);
        random_init_1D<int32_t>(fir_sparse_config->sample_count, fir_sparse_output[i]->dst);
    }

    config = (config_t *)fir_sparse_config;
    input = (input_t **)fir_sparse_input;
    output = (output_t **)fir_sparse_output;

    printf("initialized %d versions\n", count);

    return count;
}