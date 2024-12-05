
#ifndef __MVE_DATA_TYPES_H__
#define __MVE_DATA_TYPES_H__
#include <cstdint>

// 256 bits - for 256 x 1-bit c
typedef struct __mdvc {
    int64_t id;
} __mdvc;

// 1024 bits - for 256 x 8-bit b
typedef struct __mdvb {
    int64_t id;
} __mdvb;

// 2048 bits - for 256 x 16-bit w
typedef struct __mdvw {
    int64_t id;
} __mdvw;

// 4096 bits - for 256 x 32-bit dw
typedef struct __mdvdw {
    int64_t id;
} __mdvdw;

// 8192 bits - for 256 x 64-bit qw
typedef struct __mdvqw {
    int64_t id;
} __mdvqw;

// 2048 bits - for 256 x 16-bit f
typedef struct __mdvhf {
    int64_t id;
} __mdvhf;

// 4096 bits - for 256 x 32-bit f
typedef struct __mdvf {
    int64_t id;
} __mdvf;

// 8192 bits - for 256 x 64-bit df
typedef struct __mdvdf {
    int64_t id;
} __mdvdf;

#endif
