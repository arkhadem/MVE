#include "neon_kernels.hpp"
#include "strlen.hpp"
#include <stdio.h>
#include <string.h>

void strlen_neon(int LANE_NUM,
                 config_t *config,
                 input_t *input,
                 output_t *output) {
    strlen_input_t *strlen_input = (strlen_input_t *)input;
    strlen_output_t *strlen_output = (strlen_output_t *)output;

    char *src = strlen_input->src;

    strlen_output->return_value[0] = strlen(src);
}