#ifndef C64E03B0_2C82_4453_8369_5FFFBC214EF0
#define C64E03B0_2C82_4453_8369_5FFFBC214EF0

#include "benchmark.hpp"
#include "init.hpp"

void fir_rvv(int, config_t *, input_t *, output_t *);
void fir_lattice_rvv(int, config_t *, input_t *, output_t *);
void fir_sparse_rvv(int, config_t *, input_t *, output_t *);

void dct_rvv(int, config_t *, input_t *, output_t *);
void idct_rvv(int, config_t *, input_t *, output_t *);
void intra_rvv(int, config_t *, input_t *, output_t *);
void satd_rvv(int, config_t *, input_t *, output_t *);

void lpack_rvv(int, config_t *, input_t *, output_t *);

#endif /* C64E03B0_2C82_4453_8369_5FFFBC214EF0 */
