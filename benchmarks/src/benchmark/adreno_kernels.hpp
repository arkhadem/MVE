#ifndef __ADRENO_KERNELS_HPP__
#define __ADRENO_KERNELS_HPP__

#include "benchmark.hpp"
#include "clutil.hpp"
#include "init.hpp"

timing_t fir_adreno(config_t *, input_t *, output_t *);
timing_t fir_lattice_adreno(config_t *, input_t *, output_t *);
timing_t fir_sparse_adreno(config_t *, input_t *, output_t *);

timing_t dct_adreno(config_t *, input_t *, output_t *);
timing_t idct_adreno(config_t *, input_t *, output_t *);
timing_t intra_adreno(config_t *, input_t *, output_t *);
timing_t satd_adreno(config_t *, input_t *, output_t *);

timing_t lpack_adreno(config_t *, input_t *, output_t *);

void fir_InitGPU(config_t *);
void fir_lattice_InitGPU(config_t *);
void fir_sparse_InitGPU(config_t *);
void dct_InitGPU(config_t *);
void idct_InitGPU(config_t *);
void intra_InitGPU(config_t *);
void satd_InitGPU(config_t *);
void lpack_InitGPU(config_t *);

void fir_DestroyGPU(config_t *);
void fir_lattice_DestroyGPU(config_t *);
void fir_sparse_DestroyGPU(config_t *);
void dct_DestroyGPU(config_t *);
void idct_DestroyGPU(config_t *);
void intra_DestroyGPU(config_t *);
void satd_DestroyGPU(config_t *);
void lpack_DestroyGPU(config_t *);

#endif /* __ADRENO_KERNELS_HPP__ */
