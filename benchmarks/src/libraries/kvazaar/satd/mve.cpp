#include "mve.hpp"
#include "cstdint"
#include "kvazaar.hpp"
#include "satd.hpp"
#include <cstdint>
#include <cstdio>

void satd_mve(int LANE_NUM,
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

    __mdvb temp;

    while (count > 0) {

        _mve_set_dim_length(1, count > count_per_iter ? count_per_iter : count);
        count -= count_per_iter;

        ////////////////  HORIZONTAL  ////////////////

        // Column 0 (+, +, +, +, +, +, +, +)

        // R0-L
        temp = _mve_load_b(piOrg + 0, src_stride);
        // R1
        __mdvdw src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 0, src_stride);
        // R2
        __mdvdw src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R3
        __mdvdw m2 = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // Column 1 (+, -, +, -, +, -, +, -)

        // R0-L
        temp = _mve_load_b(piOrg + 1, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 1, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        __mdvdw diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        // R0
        __mdvdw add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // [R0 R1 R2 R3]

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        // R0
        __mdvdw sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // [R0 R1 R2 R4 R5]

        // Column 2 (+, +, -, -, +, +, -, -)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 2, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 2, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 3 (+, -, -, +, +, -, -, +)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 3, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 3, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 4 (+, +, +, +, -, -, -, -)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 4, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 4, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 3);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 4);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 5 (+, -, +, -, -, +, -, +)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 5, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 5, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 6 (+, +, -, -, -, -, +, +)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 6, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 6, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 7 (+, -, -, +, -, +, +, -)

        _mve_set_all_elements(2);

        // R0-L
        temp = _mve_load_b(piOrg + 7, src_stride);
        // R1
        src_1 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R0-L
        temp = _mve_load_b(piCur + 7, src_stride);
        // R2
        src_2 = _mve_cvts_btodw(temp);
        // free temp (R0-L)
        _mve_free_b();

        // R4
        diff = _mve_sub_dw(src_1, src_2);
        // free src_1 (R1) and src_2 (R2)
        _mve_free_dw();
        _mve_free_dw();

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        // R0
        add = _mve_add_dw(m2, diff);

        // R5
        m2 = _mve_assign_dw(m2, add);
        // free m2 (R3) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m2, diff);
        // free diff (R4)
        _mve_free_dw();

        // R3
        m2 = _mve_assign_dw(m2, sub);
        // free m2 (R5) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        _mve_set_all_elements(2);

        _mve_store_dw(m2_mem, m2, sequential_stride);
        // free m2 (R3)
        _mve_free_dw();

        ////////////////  VERTICAL  ////////////////

        // Column 0 (+, +, +, +, +, +, +, +)

        // R5
        __mdvdw m3 = _mve_load_dw(m2_mem, transpose_stride);

        // Column 1 (+, -, +, -, +, -, +, -)

        // R2
        m2 = _mve_load_dw(m2_mem + 8, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 2 (+, +, -, -, +, +, -, -)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 16, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 3 (+, -, -, +, +, -, -, +)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 24, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 4 (+, +, +, +, -, -, -, -)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 32, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 3);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 4);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 5 (+, -, +, -, -, +, -, +)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 40, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 6);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 6 (+, +, -, -, -, -, +, +)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 48, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 1);
        _mve_set_active_element(2, 6);
        _mve_set_active_element(2, 7);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 2);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 5);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        // Column 7 (+, -, -, +, -, +, +, -)

        _mve_set_all_elements(2);

        // R2
        m2 = _mve_load_dw(m2_mem + 56, transpose_stride);

        // +
        _mve_set_only_element(2, 0);
        _mve_set_active_element(2, 3);
        _mve_set_active_element(2, 5);
        _mve_set_active_element(2, 6);

        // R0
        add = _mve_add_dw(m3, m2);

        // R4
        m3 = _mve_assign_dw(m3, add);
        // free m3 (R5) and add (R0)
        _mve_free_dw();
        _mve_free_dw();

        // -
        _mve_set_only_element(2, 1);
        _mve_set_active_element(2, 2);
        _mve_set_active_element(2, 4);
        _mve_set_active_element(2, 7);

        // R0
        sub = _mve_sub_dw(m3, m2);
        // free m2 (R2)
        _mve_free_dw();

        // R5
        m3 = _mve_assign_dw(m3, sub);
        // free m3 (R4) and sub (R0)
        _mve_free_dw();
        _mve_free_dw();

        _mve_set_all_elements(2);

        _mve_store_dw(result, m3, dst_stride);
        // free m3 (R5)
        _mve_free_dw();

        piOrg += LANE_NUM;
        piCur += LANE_NUM;
        result += LANE_NUM;

        mve_flusher();
    }
}