#ifndef EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F
#define EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F

#include <stdint.h>
#include <stdio.h>

#define CACHE_SIZE 2097152

typedef struct timing_t {
    double create_buffer;
    double map_buffer;
    double memcpy;
    double kernel_launch;
    double kernel_execute;
} timing_t;

void benchmark_runner(const char *library, const char *kernel, int rounds, bool execute, int LANE_NUM);
long *pollute_cache(size_t size);

typedef struct config_s {
    int dummy[1];
} config_t;

typedef struct input_s {
    int dummy[1];
} input_t;

typedef struct output_s {
    uint32_t dummy[1];
} output_t;

typedef void (*kernelfunc)(int, config_t *, input_t *, output_t *);
typedef timing_t (*adrenokernelfunc)(config_t *, input_t *, output_t *);
typedef int (*initfunc)(size_t, int, config_t *&, input_t **&, output_t **&);
typedef void (*initGPUfunc)(config_t *);
typedef void (*destroyGPUfunc)(config_t *);

extern int XNNPACK_M;
extern int XNNPACK_N;
extern int XNNPACK_K;

#endif /* EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F */
