#include "fir.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int fir_init(size_t cache_size,
             int LANE_NUM,
             config_t *&config,
             input_t **&input,
             output_t **&output) {

    fir_config_t *fir_config = (fir_config_t *)config;
    fir_input_t **fir_input = (fir_input_t **)input;
    fir_output_t **fir_output = (fir_output_t **)output;

    // configuration
    init_1D<fir_config_t>(1, fir_config);
    fir_config->sample_count = 192 * 1024 * 1024;
    fir_config->coeff_count = 32;
    int input_count = fir_config->sample_count + fir_config->coeff_count - 1;
    int count = cache_size / ((input_count + fir_config->coeff_count + fir_config->sample_count) * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<fir_input_t *>(count, fir_input);
    init_1D<fir_output_t *>(count, fir_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<fir_input_t>(1, fir_input[i]);
        init_1D<fir_output_t>(1, fir_output[i]);

        random_init_1D<int32_t>(input_count, fir_input[i]->src);
        random_init_1D<int32_t>(fir_config->coeff_count, fir_input[i]->coeff);
        random_init_1D<int32_t>(fir_config->sample_count, fir_output[i]->dst);
    }

    config = (config_t *)fir_config;
    input = (input_t **)fir_input;
    output = (output_t **)fir_output;

    printf("initialized %d versions\n", count);

    return count;
}