#include "chacha20.hpp"

#include "boringssl.hpp"

#include "benchmark.hpp"

#include "init.hpp"

#include "arm_arch.h"
#include <cstdint>

const uint8_t sigma[16] = {'e', 'x', 'p', 'a', 'n', 'd', ' ', '3',
                           '2', '-', 'b', 'y', 't', 'e', ' ', 'k'};

uint32_t block_counter[65536];

uint32_t OPENSSL_armcap_P;

int chacha20_init(size_t cache_size,
                  int LANE_NUM,
                  config_t *&config,
                  input_t **&input,
                  output_t **&output) {

    chacha20_config_t *chacha20_config = (chacha20_config_t *)config;
    chacha20_input_t **chacha20_input = (chacha20_input_t **)input;
    chacha20_output_t **chacha20_output = (chacha20_output_t **)output;

    // configuration
    int length = 65536;

    init_1D<chacha20_config_t>(1, chacha20_config);
    chacha20_config->in_len = length;
    chacha20_config->counter = 4291;
    for (int i = 0; i < LANE_NUM; i++) {
        block_counter[i] = i;
    }

    // in/output versions
    size_t input_size = length * sizeof(uint8_t);
    input_size += 32 * sizeof(uint8_t);
    input_size += 12 * sizeof(uint8_t);
    size_t output_size = length * sizeof(uint8_t);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<chacha20_input_t *>(count, chacha20_input);
    init_1D<chacha20_output_t *>(count, chacha20_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<chacha20_input_t>(1, chacha20_input[i]);
        init_1D<chacha20_output_t>(1, chacha20_output[i]);

        random_init_1D<uint8_t>(length, chacha20_input[i]->in);
        random_init_1D<uint8_t>(32, chacha20_input[i]->key);
        random_init_1D<uint8_t>(12, chacha20_input[i]->nonce);
        random_init_1D<uint8_t>(length, chacha20_output[i]->out);
    }

#if defined(__ARMEL__) || defined(_M_ARM) || defined(__AARCH64EL__) || defined(_M_ARM64)
    OPENSSL_armcap_P = 0;
    OPENSSL_armcap_P |= ARMV7_NEON;
    OPENSSL_armcap_P |= ARMV8_AES;
    OPENSSL_armcap_P |= ARMV8_PMULL;
    OPENSSL_armcap_P |= ARMV8_SHA1;
    OPENSSL_armcap_P |= ARMV8_SHA256;
    OPENSSL_armcap_P |= ARMV8_SHA512;
#endif

    config = (config_t *)chacha20_config;
    input = (input_t **)chacha20_input;
    output = (output_t **)chacha20_output;

    return count;
}