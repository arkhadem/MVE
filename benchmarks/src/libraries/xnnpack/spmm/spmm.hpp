#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include "kvazaar.hpp"

typedef struct spmm_config_s : config_t {
    float sparsity;
    int M;
    int N;
    int K;
    int32_t min;
    int32_t max;
} spmm_config_t;

typedef struct spmm_input_s : input_t {
    int32_t *input;
    int32_t *bias;
    int32_t *weights;
    int32_t *IDX;
    uint32_t *NNZ;
} spmm_input_t;

typedef struct spmm_output_s : output_t {
    int32_t *output;
} spmm_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
