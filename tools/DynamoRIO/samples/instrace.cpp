/* **********************************************************
 * Copyright (c) 2131 Google, Inc.  All rights reserved.
 * **********************************************************/

/*
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of Google, Inc. nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL GOOGLE, INC. OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

/* Finds the instruction trace of a specific function
 *
 */

#include "dr_api.h"
#include "drmgr.h"
#include "droption.h"
#include "drreg.h"
#include "drsyms.h"
#include "drutil.h"
#include "utils.h"
#include <fstream>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <string>

#define CONST_FLAGS 4
#define BB_NONE 0
#define BB_FUNC_START 1
#define BB_FUNC_MIDDLE 2
#define BB_FUNC_FINISH 3

#define MVE_FINISHER_FUNC_NAME "mve_finisher"

// #define DEBUG

#ifndef DEBUG
#define dr_hint(...)
#else
#define dr_hint(...)         \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)
#endif

#define drs_fprintf(...)                  \
    do {                                  \
        fprintf(data->logf, __VA_ARGS__); \
    } while (0)

enum {
    REF_TYPE_READ = 0,
    REF_TYPE_WRITE = 1,
};

typedef struct _instr_ref_t {
    // ushort type; /* r(0), w(1), or opcode (assuming 0/1 are invalid opcode) */
    // ushort size; /* mem ref size or instr length */
    app_pc addr; /* mem ref addr or instr pc */
} instr_ref_t;

/* Max number of mem_ref a buffer can have. It should be big enough
 * to hold all entries between clean calls.
 */
#define MAX_NUM_MEM_REFS 1024
/* The maximum size of buffer for holding mem_refs. */
#define MEM_BUF_SIZE (sizeof(instr_ref_t) * MAX_NUM_MEM_REFS)

#define MINSERT instrlist_meta_preinsert

/* thread private counter */
typedef struct {
    byte *seg_base;
    instr_ref_t *buf_base;
    file_t log;
    FILE *logf;
} per_thread_t;

/* Allocated TLS slot offsets */
enum {
    MEMTRACE_TLS_OFFS_BUF_PTR,
    MEMTRACE_TLS_COUNT, /* total number of TLS slots allocated */
};

static reg_id_t tls_seg;
static uint tls_offs;
static int tls_idx;
#define TLS_SLOT(tls_base, enum_val) (void **)((byte *)(tls_base) + tls_offs + (enum_val))
#define BUF_PTR(tls_base) *(instr_ref_t **)TLS_SLOT(tls_base, MEMTRACE_TLS_OFFS_BUF_PTR)

int intrinsics_num = 213;
const char *mve_intrinsics_name[213] = {
    "mve_init_dims",
    "mve_flusher",
    "_mve_set_load_stride",
    "_mve_set_store_stride",
    "_mve_set_dim_count",
    "_mve_set_dim_length",
    "_mve_set_mask",
    "_mve_set_active_element",
    "_mve_unset_active_element",
    "_mve_set_only_element",
    "_mve_unset_only_element",
    "_mve_set_all_elements",
    "_mve_unset_all_elements",
    "_mve_shirs_b",
    "_mve_shirs_w",
    "_mve_shirs_dw",
    "_mve_shirs_qw",
    "_mve_shiru_b",
    "_mve_shiru_w",
    "_mve_shiru_dw",
    "_mve_shiru_qw",
    "_mve_shil_b",
    "_mve_shil_w",
    "_mve_shil_dw",
    "_mve_shil_qw",
    "_mve_rotir_b",
    "_mve_rotir_w",
    "_mve_rotir_dw",
    "_mve_rotir_qw",
    "_mve_rotil_b",
    "_mve_rotil_w",
    "_mve_rotil_dw",
    "_mve_rotil_qw",
    "_mve_shrrs_b",
    "_mve_shrrs_w",
    "_mve_shrrs_dw",
    "_mve_shrrs_qw",
    "_mve_shrru_b",
    "_mve_shrru_w",
    "_mve_shrru_dw",
    "_mve_shrru_qw",
    "_mve_shrl_b",
    "_mve_shrl_w",
    "_mve_shrl_dw",
    "_mve_shrl_qw",
    "_mve_set1_b",
    "_mve_set1_w",
    "_mve_set1_dw",
    "_mve_set1_qw",
    "_mve_set1_hf",
    "_mve_set1_f",
    "_mve_set1_df",
    "_mve_load_b",
    "_mve_load_w",
    "_mve_load_dw",
    "_mve_load_qw",
    "_mve_load_hf",
    "_mve_load_f",
    "_mve_load_df",
    "_mve_dict_b",
    "_mve_dict_w",
    "_mve_dict_dw",
    "_mve_dict_qw",
    "_mve_store_b",
    "_mve_store_w",
    "_mve_store_dw",
    "_mve_store_qw",
    "_mve_store_hf",
    "_mve_store_f",
    "_mve_store_df",
    "_mve_loadr_b",
    "_mve_loadr_w",
    "_mve_loadr_dw",
    "_mve_loadr_qw",
    "_mve_loadr_hf",
    "_mve_loadr_f",
    "_mve_loadr_df",
    "_mve_storer_b",
    "_mve_storer_w",
    "_mve_storer_dw",
    "_mve_storer_qw",
    "_mve_storer_hf",
    "_mve_storer_f",
    "_mve_storer_df",
    "_mve_loadro_b",
    "_mve_loadro_w",
    "_mve_loadro_dw",
    "_mve_loadro_qw",
    "_mve_loadro_hf",
    "_mve_loadro_f",
    "_mve_loadro_df",
    "_mve_storero_b",
    "_mve_storero_w",
    "_mve_storero_dw",
    "_mve_storero_qw",
    "_mve_storero_hf",
    "_mve_storero_f",
    "_mve_storero_df",
    "_mve_add_b",
    "_mve_add_w",
    "_mve_add_dw",
    "_mve_add_qw",
    "_mve_add_hf",
    "_mve_add_f",
    "_mve_add_df",
    "_mve_sub_b",
    "_mve_sub_w",
    "_mve_sub_dw",
    "_mve_sub_qw",
    "_mve_sub_hf",
    "_mve_sub_f",
    "_mve_sub_df",
    "_mve_mul_b",
    "_mve_mul_w",
    "_mve_mul_dw",
    "_mve_mul_qw",
    "_mve_mul_hf",
    "_mve_mul_f",
    "_mve_mul_df",
    "_mve_mulmodp_dw",
    "_mve_min_b",
    "_mve_min_w",
    "_mve_min_dw",
    "_mve_min_qw",
    "_mve_min_hf",
    "_mve_min_f",
    "_mve_min_df",
    "_mve_max_b",
    "_mve_max_w",
    "_mve_max_dw",
    "_mve_max_qw",
    "_mve_max_hf",
    "_mve_max_f",
    "_mve_max_df",
    "_mve_xor_b",
    "_mve_xor_w",
    "_mve_xor_dw",
    "_mve_xor_qw",
    "_mve_and_b",
    "_mve_and_w",
    "_mve_and_dw",
    "_mve_and_qw",
    "_mve_or_b",
    "_mve_or_w",
    "_mve_or_dw",
    "_mve_or_qw",
    "_mve_cmpeq_b",
    "_mve_cmpeq_w",
    "_mve_cmpeq_dw",
    "_mve_cmpeq_qw",
    "_mve_cmpeq_hf",
    "_mve_cmpeq_f",
    "_mve_cmpeq_df",
    "_mve_cmpneq_b",
    "_mve_cmpneq_w",
    "_mve_cmpneq_dw",
    "_mve_cmpneq_qw",
    "_mve_cmpneq_hf",
    "_mve_cmpneq_f",
    "_mve_cmpneq_df",
    "_mve_cmpgte_b",
    "_mve_cmpgte_w",
    "_mve_cmpgte_dw",
    "_mve_cmpgte_qw",
    "_mve_cmpgte_hf",
    "_mve_cmpgte_f",
    "_mve_cmpgte_df",
    "_mve_cmpgt_b",
    "_mve_cmpgt_w",
    "_mve_cmpgt_dw",
    "_mve_cmpgt_qw",
    "_mve_cmpgt_hf",
    "_mve_cmpgt_f",
    "_mve_cmpgt_df",
    "_mve_cvt_wtob",
    "_mve_cvt_dwtow",
    "_mve_cvt_dwtob",
    "_mve_cvtu_btow",
    "_mve_cvts_btow",
    "_mve_cvtu_btodw",
    "_mve_cvts_btodw",
    "_mve_cvtu_wtodw",
    "_mve_cvts_wtodw",
    "_mve_cvtu_dwtoqw",
    "_mve_cvts_dwtoqw",
    "_mve_assign_b",
    "_mve_assign_w",
    "_mve_assign_dw",
    "_mve_assign_qw",
    "_mve_assign_hf",
    "_mve_assign_f",
    "_mve_assign_df",
    "_mve_free_b",
    "_mve_free_w",
    "_mve_free_dw",
    "_mve_free_qw",
    "_mve_free_hf",
    "_mve_free_f",
    "_mve_free_df",
    "_mve_copy_b",
    "_mve_copy_w",
    "_mve_copy_dw",
    "_mve_copy_qw",
    "_mve_copy_hf",
    "_mve_copy_f",
    "_mve_copy_df",
    "_mve_redsum_b",
    "_mve_redsum_w",
    "_mve_redsum_dw",
    "_mve_redsum_qw",
    "_mve_redsum_hf",
    "_mve_redsum_f",
    "_mve_redsum_df"};

static const module_data_t *mve_intrinsics_mod = NULL;
static app_pc mve_intrinsics_pc[213];

static module_data_t *trace_function_start_mod = NULL;
static app_pc trace_function_start_pc;

static const module_data_t *trace_function_finish_mod = NULL;
static app_pc trace_function_finish_pc;

static bool func_running = false;
static bool func_paused = false;
static uint64 prev_comp = 0;

static client_id_t client_id;

static droption_t<std::string> trace_function(
    DROPTION_SCOPE_CLIENT, "trace_function", "malloc",
    "Name of function to trace",
    "The name of the function to wrap and print callstacks on every call.");

static void insert_load_buf_ptr(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t reg_ptr) {
    DR_ASSERT(drcontext != NULL);
    DR_ASSERT(ilist != NULL);
    DR_ASSERT(where != NULL);
    dr_insert_read_raw_tls(drcontext, ilist, where, tls_seg, tls_offs + MEMTRACE_TLS_OFFS_BUF_PTR, reg_ptr);
}

static void insert_update_buf_ptr(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t reg_ptr, int adjust) {
    DR_ASSERT(drcontext != NULL);
    DR_ASSERT(ilist != NULL);
    DR_ASSERT(where != NULL);

    MINSERT(ilist, where, XINST_CREATE_add(drcontext, opnd_create_reg(reg_ptr), OPND_CREATE_INT16(adjust)));

    dr_insert_write_raw_tls(drcontext, ilist, where, tls_seg, tls_offs + MEMTRACE_TLS_OFFS_BUF_PTR, reg_ptr);
}

// static void insert_save_type(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t base, reg_id_t scratch, ushort type) {
//     scratch = reg_resize_to_opsz(scratch, OPSZ_2);
//     MINSERT(ilist, where, XINST_CREATE_load_int(drcontext, opnd_create_reg(scratch), OPND_CREATE_INT16(type)));
//     MINSERT(ilist, where, XINST_CREATE_store_2bytes(drcontext, OPND_CREATE_MEM16(base, offsetof(instr_ref_t, type)), opnd_create_reg(scratch)));
// }

// static void insert_save_size(void *drcontext, instrlist_t *ilist, instr_t *where, reg_id_t base, reg_id_t scratch, ushort size) {
//     scratch = reg_resize_to_opsz(scratch, OPSZ_2);
//     MINSERT(ilist, where, XINST_CREATE_load_int(drcontext, opnd_create_reg(scratch), OPND_CREATE_INT16(size)));
//     MINSERT(ilist, where, XINST_CREATE_store_2bytes(drcontext, OPND_CREATE_MEM16(base, offsetof(instr_ref_t, size)), opnd_create_reg(scratch)));
// }

static void insert_save_addr(void *drcontext, instrlist_t *ilist, instr_t *where, opnd_t ref, reg_id_t reg_ptr, reg_id_t reg_addr) {
    /* we use reg_ptr as scratch to get addr */
    DR_ASSERT(drutil_insert_get_mem_addr(drcontext, ilist, where, ref, reg_addr, reg_ptr));
    insert_load_buf_ptr(drcontext, ilist, where, reg_ptr);
    MINSERT(ilist, where, XINST_CREATE_store(drcontext, OPND_CREATE_MEMPTR(reg_ptr, offsetof(instr_ref_t, addr)), opnd_create_reg(reg_addr)));
}

/* insert inline code to add a memory reference info entry into the buffer */
static void instrument_mem(void *drcontext, instrlist_t *ilist, instr_t *where, opnd_t ref, bool write) {
    /* We need two scratch registers */
    reg_id_t reg_ptr, reg_tmp;
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg_ptr) != DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &reg_tmp) != DRREG_SUCCESS) {
        DR_ASSERT(false); /* cannot recover */
        return;
    }
    /* save_addr should be called first as reg_ptr or reg_tmp maybe used in ref */
    insert_save_addr(drcontext, ilist, where, ref, reg_ptr, reg_tmp);
    // insert_save_type(drcontext, ilist, where, reg_ptr, reg_tmp, write ? REF_TYPE_WRITE : REF_TYPE_READ);
    // insert_save_size(drcontext, ilist, where, reg_ptr, reg_tmp, (ushort)drutil_opnd_mem_size_in_bytes(ref, where));
    insert_update_buf_ptr(drcontext, ilist, where, reg_ptr, sizeof(instr_ref_t));
    /* Restore scratch registers */
    if (drreg_unreserve_register(drcontext, ilist, where, reg_ptr) != DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, reg_tmp) != DRREG_SUCCESS)
        DR_ASSERT(false);
}

static void mve_print(uint flag) {
    per_thread_t *data = (per_thread_t *)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    DR_ASSERT(func_running == true);
    drs_fprintf("mve %s %lu\n", mve_intrinsics_name[flag], prev_comp);
    prev_comp = 0;
    func_paused = false;
}

static void load_print() {
    per_thread_t *data = (per_thread_t *)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    instr_ref_t *buf_ptr = BUF_PTR(data->seg_base);

    if ((func_running == true) && (func_paused == false)) {
        for (instr_ref_t *mem_ref = (instr_ref_t *)data->buf_base; mem_ref < buf_ptr; mem_ref++) {
            /* We use PIFX to avoid leading zeroes and shrink the resulting file. */
            drs_fprintf("load %p %lu\n", mem_ref->addr, prev_comp);
            // drs_fprintf("load %d %p %lu\n", mem_ref->size, mem_ref->addr, prev_comp);
        }
        prev_comp = 0;
    }
    // else {
    //     drs_fprintf("load ignored due to func_running (%d) or func_paused (%d)\n", func_running, func_paused);
    // }
    BUF_PTR(data->seg_base) = data->buf_base;
}

static void store_print() {
    per_thread_t *data = (per_thread_t *)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    instr_ref_t *buf_ptr = BUF_PTR(data->seg_base);

    if ((func_running == true) && (func_paused == false)) {
        for (instr_ref_t *mem_ref = (instr_ref_t *)data->buf_base; mem_ref < buf_ptr; mem_ref++) {
            /* We use PIFX to avoid leading zeroes and shrink the resulting file. */
            drs_fprintf("store %p %lu\n", mem_ref->addr, prev_comp);
            // drs_fprintf("store %d %p %lu\n", mem_ref->size, mem_ref->addr, prev_comp);
        }
        prev_comp = 0;
    }
    // else {
    //     drs_fprintf("store ignored due to func_running (%d) or func_paused (%d)\n", func_running, func_paused);
    // }

    BUF_PTR(data->seg_base) = data->buf_base;
}

static void compute_instruction() {
    if ((func_running == true) && (func_paused == false)) {
        prev_comp += 1;
    }
}

static void reset_execution() {
    prev_comp = 0;
    func_paused = false;
    dr_hint("execution reseted!\n");
}

static void pause_execution() {
    if ((func_running == true) && (func_paused == false)) {
        func_paused = true;
        dr_hint("execution paused!\n");
    }
}

static void resume_execution() {
    if ((func_running == true) && (func_paused == true)) {
        func_paused = true;
        dr_hint("execution resumed!\n");
    }
}

static void start_execution() {
    DR_ASSERT(func_running == false);
    dr_hint("function started!\n");
    func_running = true;
    prev_comp = 0;
    func_paused = false;
}

static void finish_execution() {
    DR_ASSERT(func_running == true);
    dr_hint("function finished!\n");
    func_running = false;
    prev_comp = 0;
    func_paused = false;
}

#ifdef DEBUG
static void bb_boundry() {
    per_thread_t *data = (per_thread_t *)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    drs_fprintf("----------------------------\n");
}
#endif

static dr_emit_flags_t event_bb_analysis(void *drcontext, void *tag,
                                         instrlist_t *bb, bool for_trace,
                                         bool translating, void **user_data) {
    instr_t *instr;

    /* Check BB's module */
    app_pc pc = dr_fragment_app_pc(tag);
    module_data_t *mod = dr_lookup_module(pc);

    *user_data = (void *)(ptr_uint_t)(BB_NONE);

    if (mod == NULL) {
        return DR_EMIT_DEFAULT;
    }

    if (mve_intrinsics_mod != NULL) {
        if (mod->start == mve_intrinsics_mod->start) {
            // dr_hint("mve BB @%p\n", pc);
            for (int i = 0; i < intrinsics_num; i++) {
                if (pc == mve_intrinsics_pc[i]) {
                    *user_data = (void *)(ptr_uint_t)(i + CONST_FLAGS);
                    dr_hint("mve intrinsic found: %s at this address: %p\n", mve_intrinsics_name[i], pc);
                }
            }
        }
    }

    DR_ASSERT(trace_function_start_mod != NULL);

    if (mod->start == trace_function_start_mod->start) {
        if (pc == trace_function_start_pc) {
            dr_hint("FUNC START BB @%p\n", pc);
            *user_data = (void *)(ptr_uint_t)(BB_FUNC_START);
        } else {
            *user_data = (void *)(ptr_uint_t)(BB_FUNC_MIDDLE);
        }
    }

    if (trace_function_finish_mod != NULL) {
        if (mod->start == trace_function_finish_mod->start) {
            if (pc == trace_function_finish_pc) {
                dr_hint("FUNC FINISH BB @%p\n", pc);
                *user_data = (void *)(ptr_uint_t)(BB_FUNC_FINISH);
            }
        }
    }

    return DR_EMIT_DEFAULT;
}

static dr_emit_flags_t event_app_instruction(void *drcontext, void *tag,
                                             instrlist_t *bb, instr_t *instr,
                                             bool for_trace, bool translating,
                                             void *user_data) {
    /* By default drmgr enables auto-predication, which predicates all
   * instructions with the predicate of the current instruction on ARM. We
   * disable it here because we want to unconditionally execute the following
   * instrumentation. */

    drmgr_disable_auto_predication(drcontext, bb);

    bool first = drmgr_is_first_instr(drcontext, instr);

    uint flag = (uint)(ptr_uint_t)user_data;

    if ((flag == BB_FUNC_START) && (first == true)) {
#ifdef DEBUG
        dr_insert_clean_call(drcontext, bb, instr, (void *)bb_boundry, false, 0);
#endif
        dr_insert_clean_call(drcontext, bb, instrlist_first_app(bb), (void *)start_execution, false, 0);
    }

    if ((flag == BB_FUNC_MIDDLE) || (flag == BB_FUNC_START)) {
        if (first) {
#ifdef DEBUG
            dr_insert_clean_call(drcontext, bb, instr, (void *)bb_boundry, false, 0);
#endif
            dr_insert_clean_call(drcontext, bb, instr, (void *)resume_execution, false, 0);
        }
        if (instr_is_call(instr)) {
            dr_insert_clean_call(drcontext, bb, instr, (void *)pause_execution, false, 0);
        } else if (instr_reads_memory(instr) || instr_writes_memory(instr)) {
            /* Insert code to add an entry for each memory reference opnd. */

            instr_t *instr_operands = drmgr_orig_app_instr_for_operands(drcontext);
            DR_ASSERT(instr_operands != NULL);
            DR_ASSERT(instr_is_app(instr_operands));

            if (instr_reads_memory(instr_operands)) {
                for (int i = 0; i < instr_num_srcs(instr_operands); i++) {
                    opnd_t src_operand = instr_get_src(instr_operands, i);
                    if (opnd_is_memory_reference(src_operand)) {
                        instrument_mem(drcontext, bb, instr, src_operand, false);
                        dr_insert_clean_call(drcontext, bb, instr, (void *)load_print, false, 0);
                    }
                }
            }

            if (instr_writes_memory(instr_operands)) {
                for (int i = 0; i < instr_num_dsts(instr_operands); i++) {
                    opnd_t dst_operand = instr_get_dst(instr_operands, i);
                    if (opnd_is_memory_reference(dst_operand)) {
                        instrument_mem(drcontext, bb, instr, dst_operand, true);
                        dr_insert_clean_call(drcontext, bb, instr, (void *)store_print, false, 0);
                    }
                }
            }

            if ((!instr_reads_memory(instr_operands)) && (!instr_writes_memory(instr_operands)))
                DR_ASSERT(false);
        } else {
            instr_t *instr_operands = drmgr_orig_app_instr_for_operands(drcontext);
            DR_ASSERT(instr_operands != NULL);
            DR_ASSERT(instr_is_app(instr_operands));

            if ((instr_reads_memory(instr_operands)) || (instr_writes_memory(instr_operands)))
                DR_ASSERT(false);

            dr_insert_clean_call(drcontext, bb, instr, (void *)compute_instruction, false, 0);
        }
    }

    if (flag == BB_FUNC_FINISH) {
        if (first)
#ifdef DEBUG
            dr_insert_clean_call(drcontext, bb, instr, (void *)bb_boundry, false, 0);
#endif
        if (first == true)
            dr_insert_clean_call(drcontext, bb, instrlist_first_app(bb), (void *)finish_execution, false, 0);
        else
            dr_insert_clean_call(drcontext, bb, instr, (void *)reset_execution, false, 0);
    }

    if (flag >= CONST_FLAGS) {
        if (first)
#ifdef DEBUG
            dr_insert_clean_call(drcontext, bb, instr, (void *)bb_boundry, false, 0);
#endif
        if (first == true)
            dr_insert_clean_call(drcontext, bb, instr, (void *)mve_print, false, 1, OPND_CREATE_INT32(flag - CONST_FLAGS));
        // else
        //     dr_insert_clean_call(drcontext, bb, instr, (void *)reset_execution, false, 0);
    }

    // if (flag == BB_NONE) {
    //     dr_insert_clean_call(drcontext, bb, instr, (void *)reset_execution, false, 0);
    // }

    return DR_EMIT_DEFAULT;
}

static void module_load_event(void *drcontext, const module_data_t *mod, bool loaded) {
    size_t modoffs;
    dr_hint("module_load_event called for %s | %s\n", mod->full_path, mod->names.module_name);
    for (int i = 0; i < intrinsics_num; i++) {
        drsym_error_t sym_res = drsym_lookup_symbol(mod->full_path, mve_intrinsics_name[i], &modoffs, DRSYM_DEMANGLE);

        if (sym_res == DRSYM_SUCCESS) {
            mve_intrinsics_pc[i] = mod->start + modoffs;
            if (mve_intrinsics_mod == NULL)
                mve_intrinsics_mod = mod;
            else
                DR_ASSERT(mve_intrinsics_mod == mod);
            dr_hint("detected: %s @%p + %p = %p\n", mve_intrinsics_name[i], mod->start, modoffs, mod->start + modoffs);
            if (strcmp("_mve_loadro_b", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_b";
            if (strcmp("_mve_loadro_w", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_w";
            if (strcmp("_mve_loadro_dw", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_dw";
            if (strcmp("_mve_loadro_qw", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_qw";
            if (strcmp("_mve_loadro_hf", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_hf";
            if (strcmp("_mve_loadro_f", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_f";
            if (strcmp("_mve_loadro_df", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_loadr_df";
            if (strcmp("_mve_storero_b", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_b";
            if (strcmp("_mve_storero_w", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_w";
            if (strcmp("_mve_storero_dw", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_dw";
            if (strcmp("_mve_storero_qw", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_qw";
            if (strcmp("_mve_storero_hf", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_hf";
            if (strcmp("_mve_storero_f", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_f";
            if (strcmp("_mve_storero_df", mve_intrinsics_name[i]) == 0)
                mve_intrinsics_name[i] = "_mve_storer_df";
        } else if (mve_intrinsics_mod != NULL) {
            dr_hint("NOT DETECTED: %s\n", mve_intrinsics_name[i]);
        }
    }

    drsym_error_t sym_res = drsym_lookup_symbol(mod->full_path, MVE_FINISHER_FUNC_NAME, &modoffs, DRSYM_DEMANGLE);

    if (sym_res == DRSYM_SUCCESS) {
        trace_function_finish_pc = mod->start + modoffs;
        if (trace_function_finish_mod == NULL)
            trace_function_finish_mod = mod;
        else
            DR_ASSERT(trace_function_finish_mod == mod);
        dr_hint("Finish function %s detected @%p + %p = %p\n", MVE_FINISHER_FUNC_NAME, mod->start, modoffs, mod->start + modoffs);
    }
}

static void event_thread_init(void *drcontext) {
    per_thread_t *data = (per_thread_t *)dr_thread_alloc(drcontext, sizeof(per_thread_t));
    DR_ASSERT(data != NULL);
    drmgr_set_tls_field(drcontext, tls_idx, data);

    /* Keep seg_base in a per-thread data structure so we can get the TLS
     * slot and find where the pointer points to in the buffer.
     */
    data->seg_base = (byte *)dr_get_dr_segment_base(tls_seg);
    data->buf_base = (instr_ref_t *)dr_raw_mem_alloc(MEM_BUF_SIZE, DR_MEMPROT_READ | DR_MEMPROT_WRITE, NULL);
    DR_ASSERT(data->seg_base != NULL && data->buf_base != NULL);
    /* put buf_base to TLS as starting buf_ptr */
    BUF_PTR(data->seg_base) = data->buf_base;

    data->log = log_file_open(client_id, drcontext, "./", "instrace", DR_FILE_ALLOW_LARGE);
    data->logf = log_stream_from_file(data->log);
}

static void event_thread_exit(void *drcontext) {
    per_thread_t *data = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    dr_raw_mem_free(data->buf_base, MEM_BUF_SIZE);
    dr_thread_free(drcontext, data, sizeof(per_thread_t));
    log_stream_close(data->logf); /* closes fd too */
}

DR_EXPORT void dr_client_main(client_id_t id, int argc, const char *argv[]) {
    /* We need 2 reg slots beyond drreg's eflags slots => 3 slots */
    drreg_options_t ops = {sizeof(ops), 3, false};
    dr_set_client_name("DynamoRIO Sample Client 'instrace'",
                       "http://dynamorio.org/issues");

    /* Options */
    if (!droption_parser_t::parse_argv(DROPTION_SCOPE_CLIENT, argc, argv, NULL, NULL))
        DR_ASSERT(false);

    if (!drmgr_init() || drreg_init(&ops) != DRREG_SUCCESS || !drutil_init())
        DR_ASSERT(false);

    /* Corresponding Function (we assume it's located in the main module) */
    trace_function_start_mod = dr_get_main_module();
    DR_ASSERT(trace_function_start_mod != NULL);
    size_t modoffs;
    DR_ASSERT(drsym_init(0) == DRSYM_SUCCESS);
    drsym_error_t sym_res = drsym_lookup_symbol(trace_function_start_mod->full_path, trace_function.get_value().c_str(), &modoffs, DRSYM_DEMANGLE);
    if (sym_res == DRSYM_SUCCESS) {
        trace_function_start_pc = trace_function_start_mod->start + modoffs;
        dr_hint("Function \"%s\" found in the main module\n", trace_function.get_value().c_str());
    } else {
        dr_hint("Couldn't find func \"%s\" in the main module\n", trace_function.get_value().c_str());
        DR_ASSERT(false);
    }

    /* register events */
    if (!drmgr_register_thread_init_event(event_thread_init) ||
        !drmgr_register_thread_exit_event(event_thread_exit))
        DR_ASSERT(false);
    if (!drmgr_register_module_load_event(module_load_event))
        DR_ASSERT(false);
    drmgr_register_bb_instrumentation_event(event_bb_analysis, event_app_instruction, NULL);

    tls_idx = drmgr_register_tls_field();
    DR_ASSERT(tls_idx != -1);

    client_id = id;

    /* The TLS field provided by DR cannot be directly accessed from the code cache.
     * For better performance, we allocate raw TLS so that we can directly
     * access and update it with a single instruction. */
    if (!dr_raw_tls_calloc(&tls_seg, &tls_offs, MEMTRACE_TLS_COUNT, 0))
        DR_ASSERT(false);

    dr_hint("Client instrace is running\n");
}
