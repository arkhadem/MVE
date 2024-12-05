#include "memset.hpp"
#include "scalar_kernels.hpp"
#include <stdio.h>
#include <string.h>

void memset_scalar(int LANE_NUM,
                   config_t *config,
                   input_t *input,
                   output_t *output) {
    memset_config_t *memset_config = (memset_config_t *)config;
    memset_input_t *memset_input = (memset_input_t *)input;
    memset_output_t *memset_output = (memset_output_t *)output;

    int size = memset_config->size;

    char value = memset_input->value[0];
    char *dst = memset_output->dst;

    memset(dst, value, size);
}