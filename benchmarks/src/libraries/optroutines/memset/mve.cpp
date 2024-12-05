#include "mve.hpp"
#include "mve_kernels.hpp"

#include "memset.hpp"

void memset_mve(int LANE_NUM,
                config_t *config,
                input_t *input,
                output_t *output) {
    memset_config_t *memset_config = (memset_config_t *)config;
    memset_input_t *memset_input = (memset_input_t *)input;
    memset_output_t *memset_output = (memset_output_t *)output;

    int size = memset_config->size;

    char value = memset_input->value[0];
    __uint8_t *dst = (__uint8_t *)memset_output->dst;

    _mve_set_dim_count(1);

    __vidx_var stride = {1, 0, 0, 0};

    _mve_set_dim_length(0, LANE_NUM);

    __mdvb value_b = _mve_set1_b(value);

    while (size > 0) {
        if (size < LANE_NUM)
            _mve_set_dim_length(0, size);
        _mve_store_b(dst, value_b, stride);
        size -= LANE_NUM;
    }

    _mve_free_b();
}