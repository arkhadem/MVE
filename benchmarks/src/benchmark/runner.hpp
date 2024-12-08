#ifndef B4A1AF31_89E5_42D4_9A07_334E337B747B
#define B4A1AF31_89E5_42D4_9A07_334E337B747B

#include "benchmark.hpp"
#include <map>
#include <string>

extern std::map<std::string, std::map<std::string, kernelfunc>> kernel_functions;
extern std::map<std::string, std::map<std::string, adrenokernelfunc>> adreno_kernel_functions;
extern std::map<std::string, std::map<std::string, initGPUfunc>> adreno_init_functions;
extern std::map<std::string, std::map<std::string, destroyGPUfunc>> adreno_destroy_functions;

#endif /* B4A1AF31_89E5_42D4_9A07_334E337B747B */
