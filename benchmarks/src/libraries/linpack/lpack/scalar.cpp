#include "cstdint"
#include "kvazaar.hpp"
#include "lpack.hpp"
#include "scalar_kernels.hpp"
#include <cstdint>

void lpack_scalar(int LANE_NUM,
                  config_t *config,
                  input_t *input,
                  output_t *output) {

    lpack_config_t *lpack_config = (lpack_config_t *)config;
    lpack_input_t *lpack_input = (lpack_input_t *)input;
    lpack_output_t *lpack_output = (lpack_output_t *)output;

    int n = lpack_config->n;
    int32_t *da = lpack_input->da;
    int32_t *dx = lpack_input->dx;
    int32_t *dyin = lpack_input->dyin;
    int32_t *dyout = lpack_output->dyout;

    for (int i = 0; i < n; i++) {
        dyout[i] = dyin[i] + da[0] * dx[i];
    }
}