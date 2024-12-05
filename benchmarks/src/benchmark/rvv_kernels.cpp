#include "benchmark.hpp"

#include "init.hpp"
#include "runner.hpp"
#include "rvv_kernels.hpp"

std::map<std::string, std::map<std::string, kernelfunc>> kernel_functions;

void register_kernels() {
    kernel_functions["cmsisdsp"]["fir"] = fir_rvv;
    kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_rvv;
    kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_rvv;

    kernel_functions["kvazaar"]["dct"] = dct_rvv;
    kernel_functions["kvazaar"]["idct"] = idct_rvv;
    kernel_functions["kvazaar"]["intra"] = intra_rvv;
    kernel_functions["kvazaar"]["satd"] = satd_rvv;

    kernel_functions["linpack"]["lpack"] = lpack_rvv;
}
