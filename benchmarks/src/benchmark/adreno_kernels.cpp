#include "benchmark.hpp"

#include "adreno_kernels.hpp"
#include "init.hpp"
#include "runner.hpp"

std::map<std::string, std::map<std::string, adrenokernelfunc>> adreno_kernel_functions;
std::map<std::string, std::map<std::string, initGPUfunc>> adreno_init_functions;
std::map<std::string, std::map<std::string, destroyGPUfunc>> adreno_destroy_functions;

void register_kernels() {
    adreno_kernel_functions["cmsisdsp"]["fir"] = fir_adreno;
    adreno_kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_adreno;
    adreno_kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_adreno;

    adreno_kernel_functions["kvazaar"]["dct"] = dct_adreno;
    adreno_kernel_functions["kvazaar"]["idct"] = idct_adreno;
    adreno_kernel_functions["kvazaar"]["intra"] = intra_adreno;
    adreno_kernel_functions["kvazaar"]["satd"] = satd_adreno;

    adreno_kernel_functions["linpack"]["lpack"] = lpack_adreno;

    adreno_init_functions["cmsisdsp"]["fir"] = fir_InitGPU;
    adreno_init_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_InitGPU;
    adreno_init_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_InitGPU;

    adreno_init_functions["kvazaar"]["dct"] = dct_InitGPU;
    adreno_init_functions["kvazaar"]["idct"] = idct_InitGPU;
    adreno_init_functions["kvazaar"]["intra"] = intra_InitGPU;
    adreno_init_functions["kvazaar"]["satd"] = satd_InitGPU;

    adreno_init_functions["linpack"]["lpack"] = lpack_InitGPU;

    adreno_destroy_functions["cmsisdsp"]["fir"] = fir_DestroyGPU;
    adreno_destroy_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_DestroyGPU;
    adreno_destroy_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_DestroyGPU;

    adreno_destroy_functions["kvazaar"]["dct"] = dct_DestroyGPU;
    adreno_destroy_functions["kvazaar"]["idct"] = idct_DestroyGPU;
    adreno_destroy_functions["kvazaar"]["intra"] = intra_DestroyGPU;
    adreno_destroy_functions["kvazaar"]["satd"] = satd_DestroyGPU;

    adreno_destroy_functions["linpack"]["lpack"] = lpack_DestroyGPU;
}
