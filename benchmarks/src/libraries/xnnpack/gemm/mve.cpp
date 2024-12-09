#include "mve.hpp"
#include "cstdint"
#include "gemm.hpp"
#include "kvazaar.hpp"
#include <cstdint>
#include <cstdio>

void gemm_mve(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    gemm_config_t *gemm_config = (gemm_config_t *)config;
    gemm_input_t *gemm_input = (gemm_input_t *)input;
    gemm_output_t *gemm_output = (gemm_output_t *)output;

    int M = gemm_config->M;
    int N = gemm_config->N;
    int K = gemm_config->K;
    int32_t min = gemm_config->min;
    int32_t max = gemm_config->max;
    int32_t *in = gemm_input->input;
    int32_t *bias = gemm_input->bias;
    int32_t *weights = gemm_input->weights;
    int32_t *out = gemm_output->output;

    // Dim0: M, Dim1: N
    _mve_set_dim_count(2);

    // Dim0: writing output to consequetive cells in M direction (Handled by stride mode)
    // Dim1: Writing output rows to rows of N direction
    _mve_set_store_stride(1, M);
    __vidx_var output_stride = {1, 3, 0, 0};

    // Dim0: Reading input from consequetive cells in M direction (Handled by stride mode)
    // Dim1: Reading the same input rows from rows of N direction (Handled by stride mode)
    __vidx_var input_stride = {1, 0, 0, 0};

    // Dim0: Reading the same weight for all cells of M direction (Handled by stride mode)
    // Dim1: Reading weights column wise for cells in N direction (Handled by stride mode)
    _mve_set_load_stride(1, K);
    __vidx_var weight_stride = {0, 3, 0, 0};

    // Bias is loaded the same for all cells of M direction
    // and consequetively for rows in N direction
    __vidx_var bias_stride = {0, 1, 0, 0};

    // R0
    __mdvdw min_v = _mve_set1_dw(min);
    // R1
    __mdvdw max_v = _mve_set1_dw(max);

    int32_t *bias_addr;
    int32_t *weight_addr;
    int32_t *input_addr;
    int32_t *output_addr;

    int m = 0;

    if (LANE_NUM == 8192) {
        if (m + 8192 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 8192);
        }
        while (m + 8192 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 8192, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 8192, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 8192, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 8192;
            out += 8192;
            m += 8192;
        }
        if (m + 4096 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 4096);
        }
        while (m + 4096 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 4096, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 4096, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 4096, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 4096, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 4096, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 4096, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 4096;
            out += 4096;
            m += 4096;
        }
        if (m + 2048 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 2048);
        }
        while (m + 2048 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 2048;
            out += 2048;
            m += 2048;
        }
        if (m + 1024 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 1024);
        }
        while (m + 1024 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 1024;
            out += 1024;
            m += 1024;
        }
        if (m + 512 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 512);
        }
        while (m + 512 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 512;
            out += 512;
            m += 512;
        }
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 128 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 128);
            }

            while (n + 128 <= N) {

                // R2 - Loading bias - Dim1.length = 128 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 128;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 128 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 128 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 128 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 128 * M;

                mve_flusher();
                n += 128;
            }

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 256 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 256);
            }

            while (n + 256 <= N) {

                // R2 - Loading bias - Dim1.length = 256 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 256;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 256 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 256 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 256 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 256 * M;

                mve_flusher();
                n += 256;
            }

            if (n + 128 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 128);
            }

            while (n + 128 <= N) {

                // R2 - Loading bias - Dim1.length = 128 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 128;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 128 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 128 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 128 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 128 * M;

                mve_flusher();
                n += 128;
            }

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 32 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else if (LANE_NUM == 4096) {
        if (m + 4096 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 4096);
        }
        while (m + 4096 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 4096, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 4096, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 4096, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 4096;
            out += 4096;
            m += 4096;
        }
        if (m + 2048 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 2048);
        }
        while (m + 2048 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 2048;
            out += 2048;
            m += 2048;
        }
        if (m + 1024 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 1024);
        }
        while (m + 1024 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 1024;
            out += 1024;
            m += 1024;
        }
        if (m + 512 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 512);
        }
        while (m + 512 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 512;
            out += 512;
            m += 512;
        }
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 128 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 128);
            }

            while (n + 128 <= N) {

                // R2 - Loading bias - Dim1.length = 128 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 128;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 128 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 128 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 128 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 128 * M;

                mve_flusher();
                n += 128;
            }

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 16 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else if (LANE_NUM == 2048) {
        if (m + 2048 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 2048);
        }
        while (m + 2048 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 2048, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 2048, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 2048;
            out += 2048;
            m += 2048;
        }
        if (m + 1024 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 1024);
        }
        while (m + 1024 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 1024;
            out += 1024;
            m += 1024;
        }
        if (m + 512 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 512);
        }
        while (m + 512 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 512;
            out += 512;
            m += 512;
        }
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 64 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 64);
            }

            while (n + 64 <= N) {

                // R2 - Loading bias - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 64;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 64 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 64 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 64 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 64 * M;

                mve_flusher();
                n += 64;
            }

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 8 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else if (LANE_NUM == 1024) {
        if (m + 1024 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 1024);
        }
        while (m + 1024 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 1024, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 1024, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 1024;
            out += 1024;
            m += 1024;
        }
        if (m + 512 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 512);
        }
        while (m + 512 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 512;
            out += 512;
            m += 512;
        }
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 32 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 32);
            }

            while (n + 32 <= N) {

                // R2 - Loading bias - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 32;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 32 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 32 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 32 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 32 * M;

                mve_flusher();
                n += 32;
            }

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 4 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else if (LANE_NUM == 512) {
        if (m + 512 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 512);
        }
        while (m + 512 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 512, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 512, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 512;
            out += 512;
            m += 512;
        }
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 16 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 16);
            }

            while (n + 16 <= N) {

                // R2 - Loading bias - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 16;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 16 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 16 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 16 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 16 * M;

                mve_flusher();
                n += 16;
            }

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 2 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else if (LANE_NUM == 256) {
        if (m + 256 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 256);
        }
        while (m + 256 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 256, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 256, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 256;
            out += 256;
            m += 256;
        }
        if (m + 128 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 128);
        }
        while (m + 128 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 128, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 128, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 128;
            out += 128;
            m += 128;
        }
        if (m + 64 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 64);
        }
        while (m + 64 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 64, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 64, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 64;
            out += 64;
            m += 64;
        }
        if (m + 32 <= M) {
            // First Dim: M
            _mve_set_dim_length(0, 32);
        }
        while (m + 32 <= M) {

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            if (n + 8 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 8);
            }

            while (n + 8 <= N) {

                // R2 - Loading bias - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 8;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 8 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 8 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 8 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 8 * M;

                mve_flusher();
                n += 8;
            }

            if (n + 4 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 4);
            }

            while (n + 4 <= N) {

                // R2 - Loading bias - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 4;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 4 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 4 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 4 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 4 * M;

                mve_flusher();
                n += 4;
            }

            if (n + 2 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 2);
            }

            while (n + 2 <= N) {

                // R2 - Loading bias - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 2;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 2 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 2 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 2 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 2 * M;

                mve_flusher();
                n += 2;
            }

            if (n + 1 <= N) {
                // Second Dim: N
                _mve_set_dim_length(1, 1);
            }

            while (n + 1 <= N) {

                // R2 - Loading bias - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += 1;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += 1 * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = 1 - Dim0.length = 32, Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = 1 - Dim0.length = 32, Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0) and 3 (M) (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free R5
                _mve_free_dw();

                output_addr += 1 * M;

                mve_flusher();
                n += 1;
            }

            in += 32;
            out += 32;
            m += 32;
        }
        if (M > m) {
            _mve_set_dim_length(0, M - m);

            int n_per_iter = 1 * (256 / (M - m));

            bias_addr = bias;
            weight_addr = weights;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                input_addr = in;
                int32_t *weight_addr_tmp = weight_addr;
                weight_addr += curr_n_per_iter * K;

                for (int k = 0; k < K; k++) {

                    // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 1 (dim0) and 0 (dim1)
                    __mdvdw input_v = _mve_load_dw(input_addr, input_stride);

                    input_addr += M;

                    // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 3 (K) (dim1)
                    __mdvdw weight_v = _mve_load_dw(weight_addr_tmp, weight_stride);
                    weight_addr_tmp += 1;

                    // R5
                    __mdvdw mult_v = _mve_mul_dw(input_v, weight_v);
                    // free input_v (R3) and weight_v (R4)
                    _mve_free_dw();
                    _mve_free_dw();

                    // R3
                    acc_v = _mve_add_dw(acc_v, mult_v);
                    // free acc_v (R2) and mult_v (R5)
                    _mve_free_dw();
                    _mve_free_dw();

                    // replace R3 and R2 for acc_v
                }

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free R2
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free R4
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // free acc_v (R4)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }
        }
    } else {
        printf("Error: unsupported LANE_NUM = %d\n", LANE_NUM);
        exit(-1);
    }

    // free min_v (R0) and max_v (R1)
    _mve_free_dw();
    _mve_free_dw();
}