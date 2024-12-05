#ifndef EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F
#define EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F

#include "clutil.hpp"
#include <stdint.h>
#include <stdio.h>

#define CACHE_SIZE 2097152

void benchmark_runner(char *library, char *kernel, int rounds, bool execute, int LANE_NUM);
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

#endif /* EB6E52BD_79AF_40E4_9EF0_FE0FBEAE172F */
