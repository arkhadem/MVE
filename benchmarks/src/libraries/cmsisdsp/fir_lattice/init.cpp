#include "fir_lattice.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int fir_lattice_init(size_t cache_size,
                     int LANE_NUM,
                     config_t *&config,
                     input_t **&input,
                     output_t **&output) {

    fir_lattice_config_t *fir_lattice_config = (fir_lattice_config_t *)config;
    fir_lattice_input_t **fir_lattice_input = (fir_lattice_input_t **)input;
    fir_lattice_output_t **fir_lattice_output = (fir_lattice_output_t **)output;

    // configuration
    init_1D<fir_lattice_config_t>(1, fir_lattice_config);
    fir_lattice_config->sample_count = 192 * 1024 * 1024;
    fir_lattice_config->coeff_count = 32;
    int input_count = fir_lattice_config->sample_count + fir_lattice_config->coeff_count - 1;
    int count = cache_size / ((input_count + fir_lattice_config->coeff_count + fir_lattice_config->sample_count) * sizeof(int32_t)) + 1;

    // initializing in/output
    init_1D<fir_lattice_input_t *>(count, fir_lattice_input);
    init_1D<fir_lattice_output_t *>(count, fir_lattice_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<fir_lattice_input_t>(1, fir_lattice_input[i]);
        init_1D<fir_lattice_output_t>(1, fir_lattice_output[i]);

        random_init_1D<int32_t>(input_count, fir_lattice_input[i]->src);
        random_init_1D<int32_t>(fir_lattice_config->coeff_count, fir_lattice_input[i]->coeff);
        random_init_1D<int32_t>(fir_lattice_config->sample_count, fir_lattice_output[i]->dst);
    }

    config = (config_t *)fir_lattice_config;
    input = (input_t **)fir_lattice_input;
    output = (output_t **)fir_lattice_output;

    return count;
}