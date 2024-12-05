#include "adler32.hpp"

#include "zlib.hpp"

#include "benchmark.hpp"

#include "init.hpp"

uint16_t mve_adler_coeff[65536];

int adler32_init(size_t cache_size,
                 int LANE_NUM,
                 config_t *&config,
                 input_t **&input,
                 output_t **&output) {

    adler32_config_t *adler32_config = (adler32_config_t *)config;
    adler32_input_t **adler32_input = (adler32_input_t **)input;
    adler32_output_t **adler32_output = (adler32_output_t **)output;

    // configuration
    int length = 65536;

    init_1D<adler32_config_t>(1, adler32_config);
    adler32_config->len = length;
    adler32_config->adler = 734276;

    // in/output versions
    size_t input_size = length * sizeof(Bytef);
    size_t output_size = 0;
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<adler32_input_t *>(count, adler32_input);
    init_1D<adler32_output_t *>(count, adler32_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<adler32_input_t>(1, adler32_input[i]);
        init_1D<adler32_output_t>(1, adler32_output[i]);

        random_init_1D<Bytef>(length, adler32_input[i]->buf);
        random_init_1D<uLong>(1, adler32_output[i]->return_value);
    }

    int sample_per_block = LANE_NUM / MIN_BLOCK;
    for (int i = 0; i < 65536; i++) {
        mve_adler_coeff[i] = i % sample_per_block;
    }

    config = (config_t *)adler32_config;
    input = (input_t **)adler32_input;
    output = (output_t **)adler32_output;

    return count;
}