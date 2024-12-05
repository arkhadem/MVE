#ifndef CF2D2AD5_CAD9_4839_888A_ED759CD32BB2
#define CF2D2AD5_CAD9_4839_888A_ED759CD32BB2

#include "libjpeg.hpp"

extern const int16_t ycbcr_to_rgb_const[4];

typedef struct ycbcr_to_rgb_config_s : config_t {
    JDIMENSION num_rows;
    JDIMENSION num_cols;
} ycbcr_to_rgb_config_t;

typedef struct ycbcr_to_rgb_input_s : input_t {
    JSAMPIMAGE input_buf;
} ycbcr_to_rgb_input_t;

typedef struct ycbcr_to_rgb_output_s : output_t {
    JSAMPARRAY output_buf;
} ycbcr_to_rgb_output_t;

#endif /* CF2D2AD5_CAD9_4839_888A_ED759CD32BB2 */
