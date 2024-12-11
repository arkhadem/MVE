#include "mve.hpp"
#include "cstdint"
#include "kvazaar.hpp"
#include "spmm.hpp"
#include <cstdint>
#include <cstdio>

void spmm_mve(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    spmm_config_t *spmm_config = (spmm_config_t *)config;
    spmm_input_t *spmm_input = (spmm_input_t *)input;
    spmm_output_t *spmm_output = (spmm_output_t *)output;

    int M = spmm_config->M;
    int N = spmm_config->N;
    int K = spmm_config->K;
    int32_t min = spmm_config->min;
    int32_t max = spmm_config->max;
    int32_t *in = spmm_input->input;
    int32_t *bias = spmm_input->bias;
    int32_t *weights = spmm_input->weights;
    int32_t *IDX = spmm_input->IDX;
    uint32_t *NNZ = spmm_input->NNZ;
    int32_t *out = spmm_output->output;

    // Dim0: M, Dim1: N
    _mve_set_dim_count(2);

    // Dim0: writing to consequetive cells in M direction (Handled by stride mode)
    // Dim1: Writing rows to rows of N direction
    _mve_set_store_stride(1, M);

    // Bias is loaded the same for all cells of M direction
    // and consequetively for rows in N direction
    __vidx_var bias_stride = {0, 1, 0, 0};

    // Input is loaded consequetively for cells of M direction
    // Second dim stride is random for rows in N direction
    __vidx_var input_stride = {1, 0, 0, 0};

    // Weight is loaded the same for all cells of M direction
    // Second dim stride is random for rows in N direction
    __vidx_var weight_stride = {0, 0, 0, 0};

    // Output is stored consequetively for cells of M direction
    // and with stride of M in N direction
    __vidx_var output_stride = {1, 3, 0, 0};

    // R0
    __mdvdw min_v = _mve_set1_dw(min);
    // R1
    __mdvdw max_v = _mve_set1_dw(max);

    int32_t *bias_addr;
    int32_t *output_addr;

    int m = 0;

    if (LANE_NUM == 8192) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 8192 ? (M - m) : 8192;

            int V_per_SA = curr_m_per_iter < 256 ? (256 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 256 ? (((curr_m_per_iter - 1) / 256) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[8192];
                int32_t *weight_addr[8192];
                int32_t *input_addr[8192];
                uint32_t k[8192];
                uint32_t max_k[8192];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else if (LANE_NUM == 4096) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 4096 ? (M - m) : 4096;

            int V_per_SA = curr_m_per_iter < 128 ? (128 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 128 ? (((curr_m_per_iter - 1) / 128) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[4096];
                int32_t *weight_addr[4096];
                int32_t *input_addr[4096];
                uint32_t k[4096];
                uint32_t max_k[4096];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else if (LANE_NUM == 2048) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 2048 ? (M - m) : 2048;

            int V_per_SA = curr_m_per_iter < 64 ? (64 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 64 ? (((curr_m_per_iter - 1) / 64) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[2048];
                int32_t *weight_addr[2048];
                int32_t *input_addr[2048];
                uint32_t k[2048];
                uint32_t max_k[2048];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else if (LANE_NUM == 1024) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 1024 ? (M - m) : 1024;

            int V_per_SA = curr_m_per_iter < 32 ? (32 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 32 ? (((curr_m_per_iter - 1) / 32) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[1024];
                int32_t *weight_addr[1024];
                int32_t *input_addr[1024];
                uint32_t k[1024];
                uint32_t max_k[1024];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else if (LANE_NUM == 512) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 512 ? (M - m) : 512;

            int V_per_SA = curr_m_per_iter < 16 ? (16 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 16 ? (((curr_m_per_iter - 1) / 16) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[512];
                int32_t *weight_addr[512];
                int32_t *input_addr[512];
                uint32_t k[512];
                uint32_t max_k[512];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else if (LANE_NUM == 256) {
        while (m < M) {

            int curr_m_per_iter = (M - m) < 256 ? (M - m) : 256;

            int V_per_SA = curr_m_per_iter < 8 ? (8 / curr_m_per_iter) : 1;
            int SA_per_V = curr_m_per_iter > 8 ? (((curr_m_per_iter - 1) / 8) + 1) : 1;
            int n_per_iter = 32 * V_per_SA / SA_per_V;

            _mve_set_dim_length(0, curr_m_per_iter);

            bias_addr = bias;
            output_addr = out;

            int n = 0;

            while (n < N) {

                int curr_n_per_iter = (N - n) < n_per_iter ? (N - n) : n_per_iter;

                _mve_set_dim_length(1, curr_n_per_iter);

                // R2 - Loading bias - Dim1.length = curr_n_per_iter - Dim0.length = (M - m), Stride: 0 (dim0) and 1 (dim1)
                __mdvdw acc_v = _mve_load_dw(bias_addr, bias_stride);

                bias_addr += curr_n_per_iter;

                int32_t *weight_addr_next[256];
                int32_t *weight_addr[256];
                int32_t *input_addr[256];
                uint32_t k[256];
                uint32_t max_k[256];
                k[0] = NNZ[n];
                weight_addr_next[0] = (int32_t *)(weights + k[0]);
                for (int i = 1; i < curr_n_per_iter; i++) {
                    k[i] = NNZ[n + i];
                    max_k[i - 1] = k[i];
                    weight_addr_next[i] = (int32_t *)(weights + k[i]);
                }
                max_k[curr_n_per_iter - 1] = NNZ[n + curr_n_per_iter];

                bool finished;
                __mdvdw input_v, weight_v, mult_v, add_acc_mult_v;

                do {
                    finished = true;

                    _mve_unset_all_elements(1);

                    for (int i = 0; i < curr_n_per_iter; i++) {
                        if (k[i] < max_k[i]) {
                            input_addr[i] = (int32_t *)(in + IDX[k[i]] * M);
                            weight_addr[i] = weight_addr_next[i];
                            weight_addr_next[i]++;
                            k[i]++;
                            finished = false;
                            _mve_set_active_element(1, i);
                        }
                    }

                    if (finished == false) {
                        // R3 - Loading input - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 1 (dim0)
                        input_v = _mve_loadr_dw((const int32_t **)input_addr, input_stride);

                        // R4 - Loading weight - Dim1.length = curr_n_per_iter - Dim0.length = (M - m) - Dim0.length = (M - m), Stride: 0 (dim0)
                        weight_v = _mve_loadr_dw((const int32_t **)weight_addr, weight_stride);

                        // R5 = R3 * R4
                        mult_v = _mve_mul_dw(input_v, weight_v);

                        // Free input_v (R3) and weight_v(R4)
                        _mve_free_dw();
                        _mve_free_dw();

                        // R3 = R2 + R5
                        add_acc_mult_v = _mve_add_dw(acc_v, mult_v);

                        // Free mult_v (R5)
                        _mve_free_dw();

                        // R4
                        acc_v = _mve_assign_dw(acc_v, add_acc_mult_v);

                        // Free acc_v (R2) and add_acc_mult_v (R3)
                        _mve_free_dw();
                        _mve_free_dw();

                        // Swap R2 and R4 for acc_v
                    }

                } while (finished == false);

                _mve_set_all_elements(1);

                // Caculating Min and Max

                // R4 = R2 min R1
                acc_v = _mve_min_dw(acc_v, max_v);

                // Free acc_v (R2)
                _mve_free_dw();

                // R5 = R4 min R0
                acc_v = _mve_max_dw(acc_v, min_v);

                // Free acc_v (R4)
                _mve_free_dw();

                // Storing the results, Stride: 1 (dim0), 3 or M (dim1)
                _mve_store_dw(output_addr, acc_v, output_stride);

                // Free acc_v (R5)
                _mve_free_dw();

                output_addr += curr_n_per_iter * M;

                mve_flusher();
                n += curr_n_per_iter;
            }

            in += curr_m_per_iter;
            out += curr_m_per_iter;
            m += curr_m_per_iter;
        }
    } else {
        printf("Error: unsupported LANE_NUM = %d\n", LANE_NUM);
        exit(-1);
    }

    // free min_v (R0) and max_v (R1)
    _mve_free_dw();
    _mve_free_dw();
}