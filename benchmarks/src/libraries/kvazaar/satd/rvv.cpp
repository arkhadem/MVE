#include "kvazaar.hpp"
#include "mve.hpp"
#include "satd.hpp"
#include <cstdint>
#include <cstdio>

void satd_rvv(int LANE_NUM,
              config_t *config,
              input_t *input,
              output_t *output) {

    satd_config_t *satd_config = (satd_config_t *)config;
    satd_input_t *satd_input = (satd_input_t *)input;
    satd_output_t *satd_output = (satd_output_t *)output;

    int count = satd_config->count;
    uint8_t *piOrg = satd_input->piOrg;
    uint8_t *piCur = satd_input->piCur;
    int32_t *result = satd_output->result;
    int32_t *m2_mem = (int32_t *)malloc(LANE_NUM * sizeof(int32_t));

    _mve_set_dim_count(3);

    _mve_set_dim_length(0, 8);

    _mve_set_dim_length(2, 8);

    int count_per_iter = LANE_NUM / 64;

    _mve_set_load_stride(0, 8);
    _mve_set_load_stride(1, 64);

    // Load sources {8, 64, 0}
    __vidx_var src_stride = {3, 3, 0, 0};

    // Load and transpose {1, 64, 0}
    __vidx_var transpose_stride = {1, 3, 0, 0};

    _mve_set_store_stride(0, 8);
    _mve_set_store_stride(1, 64);
    _mve_set_store_stride(2, 8);

    // Store sequentially to the memory {8, 64, 1}
    __vidx_var sequential_stride = {3, 3, 1, 0};

    // Store destinations {1, 64, 8}
    __vidx_var dst_stride = {1, 3, 3, 0};

    while (count > 0) {

        _mve_set_dim_length(1, count > count_per_iter ? count_per_iter : count);
        count -= count_per_iter;

        ////////////////  HORIZONTAL  ////////////////

        // Column 0 (+, +, +, +, +, +, +, +)

        __mdvb src1_0_b = _mve_set1_b(0);
        __mdvb src2_0_b = _mve_set1_b(0);
        __mdvb src1_1_b = _mve_set1_b(0);
        __mdvb src2_1_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_0_b = _mve_assign_b(src1_0_b, _mve_load_b(piOrg + 0, src_stride));
            src2_0_b = _mve_assign_b(src2_0_b, _mve_load_b(piCur + 0, src_stride));
            src1_1_b = _mve_assign_b(src1_1_b, _mve_load_b(piOrg + 1, src_stride));
            src2_1_b = _mve_assign_b(src2_1_b, _mve_load_b(piCur + 1, src_stride));
        }
        _mve_set_all_elements(2);

        __mdvdw src_1 = _mve_cvts_btodw(src1_0_b);
        __mdvdw src_2 = _mve_cvts_btodw(src2_0_b);
        __mdvdw m2 = _mve_sub_dw(src_1, src_2);

        // Column 1 (+, -, +, -, +, -, +, -)

        src_1 = _mve_cvts_btodw(src1_1_b);
        src_2 = _mve_cvts_btodw(src2_1_b);
        __mdvdw diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        __mdvdw add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // [R0 R1 R2 R3]

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        __mdvdw sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // [R0 R1 R2 R4 R5]

        // Column 2 (+, +, -, -, +, +, -, -)

        __mdvb src1_2_b = _mve_set1_b(0);
        __mdvb src2_2_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_2_b = _mve_assign_b(src1_2_b, _mve_load_b(piOrg + 2, src_stride));
            src2_2_b = _mve_assign_b(src2_2_b, _mve_load_b(piCur + 2, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_2_b);
        src_2 = _mve_cvts_btodw(src2_2_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // Column 3 (+, -, -, +, +, -, -, +)

        __mdvb src1_3_b = _mve_set1_b(0);
        __mdvb src2_3_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_3_b = _mve_assign_b(src1_3_b, _mve_load_b(piOrg + 3, src_stride));
            src2_3_b = _mve_assign_b(src2_3_b, _mve_load_b(piCur + 3, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_3_b);
        src_2 = _mve_cvts_btodw(src2_3_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // Column 4 (+, +, +, +, -, -, -, -)

        __mdvb src1_4_b = _mve_set1_b(0);
        __mdvb src2_4_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_4_b = _mve_assign_b(src1_4_b, _mve_load_b(piOrg + 4, src_stride));
            src2_4_b = _mve_assign_b(src2_4_b, _mve_load_b(piCur + 4, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_4_b);
        src_2 = _mve_cvts_btodw(src2_4_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 3);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 4);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // Column 5 (+, -, +, -, -, +, -, +)

        __mdvb src1_5_b = _mve_set1_b(0);
        __mdvb src2_5_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_5_b = _mve_assign_b(src1_5_b, _mve_load_b(piOrg + 5, src_stride));
            src2_5_b = _mve_assign_b(src2_5_b, _mve_load_b(piCur + 5, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_5_b);
        src_2 = _mve_cvts_btodw(src2_5_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // Column 6 (+, +, -, -, -, -, +, +)

        __mdvb src1_6_b = _mve_set1_b(0);
        __mdvb src2_6_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_6_b = _mve_assign_b(src1_6_b, _mve_load_b(piOrg + 6, src_stride));
            src2_6_b = _mve_assign_b(src2_6_b, _mve_load_b(piCur + 6, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_6_b);
        src_2 = _mve_cvts_btodw(src2_6_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        // Column 7 (+, -, -, +, -, +, +, -)

        __mdvb src1_7_b = _mve_set1_b(0);
        __mdvb src2_7_b = _mve_set1_b(0);
        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            src1_7_b = _mve_assign_b(src1_7_b, _mve_load_b(piOrg + 7, src_stride));
            src2_7_b = _mve_assign_b(src2_7_b, _mve_load_b(piCur + 7, src_stride));
        }
        _mve_set_all_elements(2);

        src_1 = _mve_cvts_btodw(src1_7_b);
        src_2 = _mve_cvts_btodw(src2_7_b);
        diff = _mve_sub_dw(src_1, src_2);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        add = _mve_add_dw(m2, diff);

        m2 = _mve_assign_dw(m2, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m2, diff);

        m2 = _mve_assign_dw(m2, sub);

        for (int element_idx = 0; element_idx < 8; element_idx += 1) {
            _mve_set_only_element(2, element_idx);
            _mve_store_dw(m2_mem, m2, sequential_stride);
        }
        _mve_set_all_elements(2);

        ////////////////  VERTICAL  ////////////////

        // Column 0 (+, +, +, +, +, +, +, +)
        __mdvdw m3 = _mve_set1_dw(0);

        // Column 1 (+, -, +, -, +, -, +, -)
        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m3 = _mve_assign_dw(m3, _mve_load_dw(m2_mem, transpose_stride));
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 8, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 2 (+, +, -, -, +, +, -, -)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 16, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 3 (+, -, -, +, +, -, -, +)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 24, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 4 (+, +, +, +, -, -, -, -)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 32, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 3);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 4);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 5 (+, -, +, -, -, +, -, +)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 40, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 6 (+, +, -, -, -, -, +, +)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 48, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        // Column 7 (+, -, -, +, -, +, +, -)

        _mve_set_all_elements(2);

        m2 = _mve_set1_dw(0);
        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                m2 = _mve_assign_dw(m2, _mve_load_dw(m2_mem + 56, transpose_stride));
            }
        }
        _mve_set_all_elements(0);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        add = _mve_add_dw(m3, m2);

        m3 = _mve_assign_dw(m3, add);

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        sub = _mve_sub_dw(m3, m2);

        m3 = _mve_assign_dw(m3, sub);

        _mve_set_all_elements(2);

        for (int i = 0; i < 8; i++) {
            _mve_set_only_element(2, i);
            for (int j = 0; j < 8; j++) {
                _mve_set_only_element(0, j);
                _mve_store_dw(result, m3, dst_stride);
            }
        }
        _mve_set_all_elements(0);

        piOrg += LANE_NUM;
        piCur += LANE_NUM;
        result += LANE_NUM;

        mve_flusher();
    }
}