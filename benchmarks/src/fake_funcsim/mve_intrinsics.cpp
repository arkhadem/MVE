#include "mve.hpp"
#include <stdio.h>
#include <string.h>
#include <string>

__mdvb temp_b;
__mdvw temp_w;
__mdvdw temp_dw;
__mdvqw temp_qw;
__mdvhf temp_hf;
__mdvf temp_f;
__mdvdf temp_df;

void mve_initializer(char *exp_name, int SA_num) { return; }
void mve_init_dims() { return; }
void mve_finisher() { return; }
void mve_flusher() { return; }
void _mve_set_load_stride(int dim, int stride) { return; }
void _mve_set_store_stride(int dim, int stride) { return; }
void _mve_set_dim_count(int count) { return; }
void _mve_set_dim_length(int dim, int length) { return; }
void _mve_set_mask() { return; }
void _mve_set_active_element(int dim, int element) { return; }
void _mve_unset_active_element(int dim, int element) { return; }
void _mve_set_only_element(int dim, int element) { return; }
void _mve_unset_only_element(int dim, int element) { return; }
void _mve_set_all_elements(int dim) { return; }
void _mve_unset_all_elements(int dim) { return; }
__mdvb _mve_shirs_b(__mdvb a, int imm) {
    return temp_b;
}
__mdvw _mve_shirs_w(__mdvw a, int imm) {
    return temp_w;
}
__mdvdw _mve_shirs_dw(__mdvdw a, int imm) {
    return temp_dw;
}
__mdvqw _mve_shirs_qw(__mdvqw a, int imm) {
    return temp_qw;
}
__mdvb _mve_shiru_b(__mdvb a, int imm) {
    return temp_b;
}
__mdvw _mve_shiru_w(__mdvw a, int imm) {
    return temp_w;
}
__mdvdw _mve_shiru_dw(__mdvdw a, int imm) {
    return temp_dw;
}
__mdvqw _mve_shiru_qw(__mdvqw a, int imm) {
    return temp_qw;
}
__mdvb _mve_shil_b(__mdvb a, int imm) {
    return temp_b;
}
__mdvw _mve_shil_w(__mdvw a, int imm) {
    return temp_w;
}
__mdvdw _mve_shil_dw(__mdvdw a, int imm) {
    return temp_dw;
}
__mdvqw _mve_shil_qw(__mdvqw a, int imm) {
    return temp_qw;
}
__mdvb _mve_rotir_b(__mdvb a, int imm) {
    return temp_b;
}
__mdvw _mve_rotir_w(__mdvw a, int imm) {
    return temp_w;
}
__mdvdw _mve_rotir_dw(__mdvdw a, int imm) {
    return temp_dw;
}
__mdvqw _mve_rotir_qw(__mdvqw a, int imm) {
    return temp_qw;
}
__mdvb _mve_rotil_b(__mdvb a, int imm) {
    return temp_b;
}
__mdvw _mve_rotil_w(__mdvw a, int imm) {
    return temp_w;
}
__mdvdw _mve_rotil_dw(__mdvdw a, int imm) {
    return temp_dw;
}
__mdvqw _mve_rotil_qw(__mdvqw a, int imm) {
    return temp_qw;
}
__mdvb _mve_shrrs_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_shrrs_w(__mdvw a, __mdvb b) {
    return temp_w;
}
__mdvdw _mve_shrrs_dw(__mdvdw a, __mdvb b) {
    return temp_dw;
}
__mdvqw _mve_shrrs_qw(__mdvqw a, __mdvb b) {
    return temp_qw;
}
__mdvb _mve_shrru_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_shrru_w(__mdvw a, __mdvb b) {
    return temp_w;
}
__mdvdw _mve_shrru_dw(__mdvdw a, __mdvb b) {
    return temp_dw;
}
__mdvqw _mve_shrru_qw(__mdvqw a, __mdvb b) {
    return temp_qw;
}
__mdvb _mve_shrl_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_shrl_w(__mdvw a, __mdvb b) {
    return temp_w;
}
__mdvdw _mve_shrl_dw(__mdvdw a, __mdvb b) {
    return temp_dw;
}
__mdvqw _mve_shrl_qw(__mdvqw a, __mdvb b) {
    return temp_qw;
}
__mdvb _mve_set1_b(__uint8_t const a) {
    return temp_b;
}
__mdvw _mve_set1_w(__int16_t const a) {
    return temp_w;
}
__mdvdw _mve_set1_dw(__int32_t const a) {
    return temp_dw;
}
__mdvqw _mve_set1_qw(__int64_t const a) {
    return temp_qw;
}
__mdvhf _mve_set1_hf(hfloat const a) {
    return temp_hf;
}
__mdvf _mve_set1_f(float const a) {
    return temp_f;
}
__mdvdf _mve_set1_df(double const a) {
    return temp_df;
}
__mdvb _mve_load_b(__uint8_t const *mem_addr, __vidx_var stride) {
    return temp_b;
}
__mdvw _mve_load_w(__int16_t const *mem_addr, __vidx_var stride) {
    return temp_w;
}
__mdvdw _mve_load_dw(__int32_t const *mem_addr, __vidx_var stride) {
    return temp_dw;
}
__mdvqw _mve_load_qw(__int64_t const *mem_addr, __vidx_var stride) {
    return temp_qw;
}
__mdvhf _mve_load_hf(hfloat const *mem_addr, __vidx_var stride) {
    return temp_hf;
}
__mdvf _mve_load_f(float const *mem_addr, __vidx_var stride) {
    return temp_f;
}
__mdvdf _mve_load_df(double const *mem_addr, __vidx_var stride) {
    return temp_df;
}
void _mve_store_b(__uint8_t *mem_addr, __mdvb a, __vidx_var stride) { return; }
void _mve_store_w(__int16_t *mem_addr, __mdvw a, __vidx_var stride) { return; }
void _mve_store_dw(__int32_t *mem_addr, __mdvdw a, __vidx_var stride) { return; }
void _mve_store_qw(__int64_t *mem_addr, __mdvqw a, __vidx_var stride) { return; }
void _mve_store_hf(hfloat *mem_addr, __mdvhf a, __vidx_var stride) { return; }
void _mve_store_f(float *mem_addr, __mdvf a, __vidx_var stride) { return; }
void _mve_store_df(double *mem_addr, __mdvdf a, __vidx_var stride) { return; }
__mdvb _mve_loadr_b(__uint8_t const **mem_addr, __vidx_var stride) {
    return temp_b;
}
__mdvw _mve_loadr_w(__int16_t const **mem_addr, __vidx_var stride) {
    return temp_w;
}
__mdvdw _mve_loadr_dw(__int32_t const **mem_addr, __vidx_var stride) {
    return temp_dw;
}
__mdvqw _mve_loadr_qw(__int64_t const **mem_addr, __vidx_var stride) {
    return temp_qw;
}
__mdvhf _mve_loadr_hf(hfloat const **mem_addr, __vidx_var stride) {
    return temp_hf;
}
__mdvf _mve_loadr_f(float const **mem_addr, __vidx_var stride) {
    return temp_f;
}
__mdvdf _mve_loadr_df(double const **mem_addr, __vidx_var stride) {
    return temp_df;
}
void _mve_storer_b(__uint8_t **mem_addr, __mdvb a, __vidx_var stride) { return; }
void _mve_storer_w(__int16_t **mem_addr, __mdvw a, __vidx_var stride) { return; }
void _mve_storer_dw(__int32_t **mem_addr, __mdvdw a, __vidx_var stride) { return; }
void _mve_storer_qw(__int64_t **mem_addr, __mdvqw a, __vidx_var stride) { return; }
void _mve_storer_hf(hfloat **mem_addr, __mdvhf a, __vidx_var stride) { return; }
void _mve_storer_f(float **mem_addr, __mdvf a, __vidx_var stride) { return; }
void _mve_storer_df(double **mem_addr, __mdvdf a, __vidx_var stride) { return; }
__mdvb _mve_loadro_b(__uint8_t const **mem_addr, int offset, __vidx_var stride) {
    return temp_b;
}
__mdvw _mve_loadro_w(__int16_t const **mem_addr, int offset, __vidx_var stride) {
    return temp_w;
}
__mdvdw _mve_loadro_dw(__int32_t const **mem_addr, int offset, __vidx_var stride) {
    return temp_dw;
}
__mdvqw _mve_loadro_qw(__int64_t const **mem_addr, int offset, __vidx_var stride) {
    return temp_qw;
}
__mdvhf _mve_loadro_hf(hfloat const **mem_addr, int offset, __vidx_var stride) {
    return temp_hf;
}
__mdvf _mve_loadro_f(float const **mem_addr, int offset, __vidx_var stride) {
    return temp_f;
}
__mdvdf _mve_loadro_df(double const **mem_addr, int offset, __vidx_var stride) {
    return temp_df;
}
void _mve_storero_b(__uint8_t **mem_addr, int offset, __mdvb a, __vidx_var stride) { return; }
void _mve_storero_w(__int16_t **mem_addr, int offset, __mdvw a, __vidx_var stride) { return; }
void _mve_storero_dw(__int32_t **mem_addr, int offset, __mdvdw a, __vidx_var stride) { return; }
void _mve_storero_qw(__int64_t **mem_addr, int offset, __mdvqw a, __vidx_var stride) { return; }
void _mve_storero_hf(hfloat **mem_addr, int offset, __mdvhf a, __vidx_var stride) { return; }
void _mve_storero_f(float **mem_addr, int offset, __mdvf a, __vidx_var stride) { return; }
void _mve_storero_df(double **mem_addr, int offset, __mdvdf a, __vidx_var stride) { return; }
__mdvb _mve_dict_b(__uint8_t const *mem_addr, __mdvb a) {
    return temp_b;
}
__mdvw _mve_dict_w(__int16_t const *mem_addr, __mdvb a) {
    return temp_w;
}
__mdvdw _mve_dict_dw(__int32_t const *mem_addr, __mdvb a) {
    return temp_dw;
}
__mdvqw _mve_dict_qw(__int64_t const *mem_addr, __mdvb a) {
    return temp_qw;
}
__mdvb _mve_add_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_add_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_add_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_add_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_add_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_add_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_add_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
__mdvb _mve_sub_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_sub_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_sub_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_sub_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_sub_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_sub_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_sub_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
__mdvb _mve_mul_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_mul_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_mul_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_mul_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_mul_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_mul_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_mul_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
__mdvdw _mve_mulmodp_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvb _mve_min_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_min_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_min_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_min_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_min_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_min_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_min_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
__mdvb _mve_max_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_max_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_max_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_max_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_max_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_max_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_max_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
__mdvb _mve_xor_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_xor_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_xor_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_xor_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvb _mve_and_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_and_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_and_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_and_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvb _mve_or_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_or_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_or_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_or_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
void _mve_cmpeq_b(__mdvb a, __mdvb b) { return; }
void _mve_cmpeq_w(__mdvw a, __mdvw b) { return; }
void _mve_cmpeq_dw(__mdvdw a, __mdvdw b) { return; }
void _mve_cmpeq_qw(__mdvqw a, __mdvqw b) { return; }
void _mve_cmpeq_hf(__mdvhf a, __mdvhf b) { return; }
void _mve_cmpeq_f(__mdvf a, __mdvf b) { return; }
void _mve_cmpeq_df(__mdvdf a, __mdvdf b) { return; }
void _mve_cmpneq_b(__mdvb a, __mdvb b) { return; }
void _mve_cmpneq_w(__mdvw a, __mdvw b) { return; }
void _mve_cmpneq_dw(__mdvdw a, __mdvdw b) { return; }
void _mve_cmpneq_qw(__mdvqw a, __mdvqw b) { return; }
void _mve_cmpneq_hf(__mdvhf a, __mdvhf b) { return; }
void _mve_cmpneq_f(__mdvf a, __mdvf b) { return; }
void _mve_cmpneq_df(__mdvdf a, __mdvdf b) { return; }
void _mve_cmpgte_b(__mdvb a, __mdvb b) { return; }
void _mve_cmpgte_w(__mdvw a, __mdvw b) { return; }
void _mve_cmpgte_dw(__mdvdw a, __mdvdw b) { return; }
void _mve_cmpgte_qw(__mdvqw a, __mdvqw b) { return; }
void _mve_cmpgte_hf(__mdvhf a, __mdvhf b) { return; }
void _mve_cmpgte_f(__mdvf a, __mdvf b) { return; }
void _mve_cmpgte_df(__mdvdf a, __mdvdf b) { return; }
void _mve_cmpgt_b(__mdvb a, __mdvb b) { return; }
void _mve_cmpgt_w(__mdvw a, __mdvw b) { return; }
void _mve_cmpgt_dw(__mdvdw a, __mdvdw b) { return; }
void _mve_cmpgt_qw(__mdvqw a, __mdvqw b) { return; }
void _mve_cmpgt_hf(__mdvhf a, __mdvhf b) { return; }
void _mve_cmpgt_f(__mdvf a, __mdvf b) { return; }
void _mve_cmpgt_df(__mdvdf a, __mdvdf b) { return; }
__mdvb _mve_cvt_wtob(__mdvw a) {
    return temp_b;
}
__mdvw _mve_cvt_dwtow(__mdvdw a) {
    return temp_w;
}
__mdvb _mve_cvt_dwtob(__mdvdw a) {
    return temp_b;
}
__mdvw _mve_cvtu_btow(__mdvb a) {
    return temp_w;
}
__mdvw _mve_cvts_btow(__mdvb a) {
    return temp_w;
}
__mdvdw _mve_cvtu_btodw(__mdvb a) {
    return temp_dw;
}
__mdvdw _mve_cvts_btodw(__mdvb a) {
    return temp_dw;
}
__mdvdw _mve_cvtu_wtodw(__mdvw a) {
    return temp_dw;
}
__mdvdw _mve_cvts_wtodw(__mdvw a) {
    return temp_dw;
}
__mdvqw _mve_cvtu_dwtoqw(__mdvdw a) {
    return temp_qw;
}
__mdvqw _mve_cvts_dwtoqw(__mdvdw a) {
    return temp_qw;
}

char *_mve_btos(__mdvb a) { return nullptr; }
char *_mve_wtos(__mdvw a) { return nullptr; }
char *_mve_dwtos(__mdvdw a) { return nullptr; }
char *_mve_qwtos(__mdvqw a) { return nullptr; }
char *_mve_hftos(__mdvhf a) { return nullptr; }
char *_mve_ftos(__mdvf a) { return nullptr; }
char *_mve_dftos(__mdvdf a) { return nullptr; }
__mdvb _mve_assign_b(__mdvb a, __mdvb b) {
    return temp_b;
}
__mdvw _mve_assign_w(__mdvw a, __mdvw b) {
    return temp_w;
}
__mdvdw _mve_assign_dw(__mdvdw a, __mdvdw b) {
    return temp_dw;
}
__mdvqw _mve_assign_qw(__mdvqw a, __mdvqw b) {
    return temp_qw;
}
__mdvhf _mve_assign_hf(__mdvhf a, __mdvhf b) {
    return temp_hf;
}
__mdvf _mve_assign_f(__mdvf a, __mdvf b) {
    return temp_f;
}
__mdvdf _mve_assign_df(__mdvdf a, __mdvdf b) {
    return temp_df;
}
void _mve_free_b() { return; }
void _mve_free_w() { return; }
void _mve_free_dw() { return; }
void _mve_free_qw() { return; }
void _mve_free_hf() { return; }
void _mve_free_f() { return; }
void _mve_free_df() { return; }
__mdvb _mve_copy_b(__mdvb a) {
    return temp_b;
}
__mdvw _mve_copy_w(__mdvw a) {
    return temp_w;
}
__mdvdw _mve_copy_dw(__mdvdw a) {
    return temp_dw;
}
__mdvqw _mve_copy_qw(__mdvqw a) {
    return temp_qw;
}
__mdvhf _mve_copy_hf(__mdvhf a) {
    return temp_hf;
}
__mdvf _mve_copy_f(__mdvf) {
    return temp_f;
}
__mdvdf _mve_copy_df(__mdvdf a) {
    return temp_df;
}