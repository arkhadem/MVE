#ifndef B54AD8F1_0C7E_45FB_B967_B373BEE1E681
#define B54AD8F1_0C7E_45FB_B967_B373BEE1E681

#include <stdint.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include "kvazaar.hpp"

#define BLOCK_16K 16384
#define BLOCK_8K 8192
#define BLOCK_4K 4096
#define BLOCK_2K 2048
#define BLOCK_1K 1024
#define BLOCK_512 512
#define BLOCK_256 256
#define BLOCK_128 128
#define BLOCK_64 64
#define BLOCK_32 32
#define BLOCK_16 16
#define BLOCK_8 8
#define BLOCK_4 4
#define BLOCK_2 2
#define BLOCK_1 1

typedef struct csum_config_s : config_t {
    int count;
} csum_config_t;

typedef struct csum_input_s : input_t {
    int32_t *ptr;
} csum_input_t;

typedef struct csum_output_s : output_t {
    int16_t *sum;
} csum_output_t;

#endif /* B54AD8F1_0C7E_45FB_B967_B373BEE1E681 */
