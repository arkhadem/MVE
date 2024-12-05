#ifndef __MVE_VARIABLE_TYPES_HPP__
#define __MVE_VARIABLE_TYPES_HPP__

#include <cstdint>

typedef int16_t hfloat;

typedef union {
    bool c;
    uint8_t b;
    int16_t w;
    int32_t dw;
    int64_t qw;
    hfloat hf;
    float f;
    double df;
} __scalar_var;

typedef struct __dim_var {
    int length = 0;
    bool *mask;
    int load_stride = 0;
    int store_stride = 0;
} __dim_var;

typedef int __vidx_var[4];

class __mdv_var {
public:
    __mdv_var(__dim_var dims[4]);
    __mdv_var(const __mdv_var &other);
    __mdv_var &operator=(const __mdv_var other);
    ~__mdv_var();
    int idx_cvt(int dim3, int dim2, int dim1, int dim0);
    int total_length();
    __scalar_var *val;
    __dim_var my_dims[4];
};

#endif