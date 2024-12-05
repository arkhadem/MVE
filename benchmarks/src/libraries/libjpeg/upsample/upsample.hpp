#ifndef CDEF0667_0942_43D0_95E6_090E75EDEE29
#define CDEF0667_0942_43D0_95E6_090E75EDEE29

#include <stdint.h>

#include "libjpeg.hpp"

extern const int16_t upsample_consts[4];

typedef struct upsample_config_s : config_t {
    // This is input size, output size is twice these
    JDIMENSION num_rows;
    JDIMENSION num_cols;
} upsample_config_t;

typedef struct upsample_input_s : input_t {
    // 3 channels: Y, CB, CR

    // 16 input rows
    // (for Y, but it is 8 for CB and CR which is not used)

    // 1024 input columns
    // (for Y, but it is 512 for CB and CR which is not used)

    JSAMPIMAGE input_buf;
} upsample_input_t;

typedef struct upsample_output_s : output_t {
    JSAMPARRAY output_buf;
} upsample_output_t;

#endif /* CDEF0667_0942_43D0_95E6_090E75EDEE29 */
