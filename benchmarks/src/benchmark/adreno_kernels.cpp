#include "benchmark.hpp"

#include "adreno_kernels.hpp"
#include "init.hpp"
#include "runner.hpp"

std::map<std::string, std::map<std::string, adrenokernelfunc>> adreno_kernel_functions;

void register_kernels() {
    adreno_kernel_functions["cmsisdsp"]["fir"] = fir_adreno;
    adreno_kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_adreno;
    adreno_kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_adreno;

    adreno_kernel_functions["kvazaar"]["dct"] = dct_adreno;
    adreno_kernel_functions["kvazaar"]["idct"] = idct_adreno;
    adreno_kernel_functions["kvazaar"]["intra"] = intra_adreno;
    adreno_kernel_functions["kvazaar"]["satd"] = satd_adreno;

    adreno_kernel_functions["linpack"]["lpack"] = lpack_adreno;
}
