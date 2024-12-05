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

#endif /* __ADRENO_KERNELS_HPP__ */
