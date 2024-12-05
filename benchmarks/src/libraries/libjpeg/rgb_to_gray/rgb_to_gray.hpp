#ifndef D197F8E1_C256_4D28_8065_BB3FD4132843
#define D197F8E1_C256_4D28_8065_BB3FD4132843

#include <stdint.h>

#include "libjpeg.hpp"

typedef struct rgb_to_gray_config_s : config_t {
    JDIMENSION num_rows;
    JDIMENSION num_cols;
} rgb_to_gray_config_t;

typedef struct rgb_to_gray_input_s : input_t {
    JSAMPARRAY input_buf;
} rgb_to_gray_input_t;

typedef struct rgb_to_gray_output_s : output_t {
    JSAMPARRAY output_buf;
} rgb_to_gray_output_t;

#endif /* D197F8E1_C256_4D28_8065_BB3FD4132843 */
