#ifndef __mve_H__
#define __mve_H__

#include <stdbool.h>

#include <stdint.h>

typedef int __vidx_var[4];
typedef int16_t hfloat;

// 256 bits - for 256 x 1-bit c
typedef struct __mdvc {
} __mdvc;

// 1024 bits - for 256 x 8-bit b
typedef struct __mdvb {
} __mdvb;

// 2048 bits - for 256 x 16-bit w
typedef struct __mdvw {
} __mdvw;

// 4096 bits - for 256 x 32-bit dw
typedef struct __mdvdw {
} __mdvdw;

// 8192 bits - for 256 x 64-bit qw
typedef struct __mdvqw {
} __mdvqw;

// 2048 bits - for 256 x 16-bit f
typedef struct __mdvhf {
} __mdvhf;

// 4096 bits - for 256 x 32-bit f
typedef struct __mdvf {
} __mdvf;

// 8192 bits - for 256 x 64-bit df
typedef struct __mdvdf {
} __mdvdf;

void mve_initializer(char *exp_name, int SA_num);
void mve_init_dims();
void mve_finisher();
void mve_flusher();
void _mve_set_load_stride(int dim, int stride);
void _mve_set_store_stride(int dim, int stride);
void _mve_set_dim_count(int count);
void _mve_set_dim_length(int dim, int length);
void _mve_set_mask();
void _mve_set_active_element(int dim, int element);
void _mve_unset_active_element(int dim, int element);
void _mve_set_only_element(int dim, int element);
void _mve_unset_only_element(int dim, int element);
void _mve_set_all_elements(int dim);
void _mve_unset_all_elements(int dim);
__mdvb _mve_shirs_b(__mdvb a, int imm);
__mdvw _mve_shirs_w(__mdvw a, int imm);
__mdvdw _mve_shirs_dw(__mdvdw a, int imm);
__mdvqw _mve_shirs_qw(__mdvqw a, int imm);
__mdvb _mve_shiru_b(__mdvb a, int imm);
__mdvw _mve_shiru_w(__mdvw a, int imm);
__mdvdw _mve_shiru_dw(__mdvdw a, int imm);
__mdvqw _mve_shiru_qw(__mdvqw a, int imm);
__mdvb _mve_shil_b(__mdvb a, int imm);
__mdvw _mve_shil_w(__mdvw a, int imm);
__mdvdw _mve_shil_dw(__mdvdw a, int imm);
__mdvqw _mve_shil_qw(__mdvqw a, int imm);
__mdvb _mve_rotir_b(__mdvb a, int imm);
__mdvw _mve_rotir_w(__mdvw a, int imm);
__mdvdw _mve_rotir_dw(__mdvdw a, int imm);
__mdvqw _mve_rotir_qw(__mdvqw a, int imm);
__mdvb _mve_rotil_b(__mdvb a, int imm);
__mdvw _mve_rotil_w(__mdvw a, int imm);
__mdvdw _mve_rotil_dw(__mdvdw a, int imm);
__mdvqw _mve_rotil_qw(__mdvqw a, int imm);
__mdvb _mve_shrrs_b(__mdvb a, __mdvb b);
__mdvw _mve_shrrs_w(__mdvw a, __mdvb b);
__mdvdw _mve_shrrs_dw(__mdvdw a, __mdvb b);
__mdvqw _mve_shrrs_qw(__mdvqw a, __mdvb b);
__mdvb _mve_shrru_b(__mdvb a, __mdvb b);
__mdvw _mve_shrru_w(__mdvw a, __mdvb b);
__mdvdw _mve_shrru_dw(__mdvdw a, __mdvb b);
__mdvqw _mve_shrru_qw(__mdvqw a, __mdvb b);
__mdvb _mve_shrl_b(__mdvb a, __mdvb b);
__mdvw _mve_shrl_w(__mdvw a, __mdvb b);
__mdvdw _mve_shrl_dw(__mdvdw a, __mdvb b);
__mdvqw _mve_shrl_qw(__mdvqw a, __mdvb b);
__mdvb _mve_set1_b(__uint8_t const a);
__mdvw _mve_set1_w(__int16_t const a);
__mdvdw _mve_set1_dw(__int32_t const a);
__mdvqw _mve_set1_qw(__int64_t const a);
__mdvhf _mve_set1_hf(hfloat const a);
__mdvf _mve_set1_f(float const a);
__mdvdf _mve_set1_df(double const a);
__mdvb _mve_load_b(__uint8_t const *mem_addr, __vidx_var stride);
__mdvw _mve_load_w(__int16_t const *mem_addr, __vidx_var stride);
__mdvdw _mve_load_dw(__int32_t const *mem_addr, __vidx_var stride);
__mdvqw _mve_load_qw(__int64_t const *mem_addr, __vidx_var stride);
__mdvhf _mve_load_hf(hfloat const *mem_addr, __vidx_var stride);
__mdvf _mve_load_f(float const *mem_addr, __vidx_var stride);
__mdvdf _mve_load_df(double const *mem_addr, __vidx_var stride);
void _mve_store_b(__uint8_t *mem_addr, __mdvb a, __vidx_var stride);
void _mve_store_w(__int16_t *mem_addr, __mdvw a, __vidx_var stride);
void _mve_store_dw(__int32_t *mem_addr, __mdvdw a, __vidx_var stride);
void _mve_store_qw(__int64_t *mem_addr, __mdvqw a, __vidx_var stride);
void _mve_store_hf(hfloat *mem_addr, __mdvhf a, __vidx_var stride);
void _mve_store_f(float *mem_addr, __mdvf a, __vidx_var stride);
void _mve_store_df(double *mem_addr, __mdvdf a, __vidx_var stride);
__mdvb _mve_loadr_b(__uint8_t const **mem_addr, __vidx_var stride);
__mdvw _mve_loadr_w(__int16_t const **mem_addr, __vidx_var stride);
__mdvdw _mve_loadr_dw(__int32_t const **mem_addr, __vidx_var stride);
__mdvqw _mve_loadr_qw(__int64_t const **mem_addr, __vidx_var stride);
__mdvhf _mve_loadr_hf(hfloat const **mem_addr, __vidx_var stride);
__mdvf _mve_loadr_f(float const **mem_addr, __vidx_var stride);
__mdvdf _mve_loadr_df(double const **mem_addr, __vidx_var stride);
void _mve_storer_b(__uint8_t **mem_addr, __mdvb a, __vidx_var stride);
void _mve_storer_w(__int16_t **mem_addr, __mdvw a, __vidx_var stride);
void _mve_storer_dw(__int32_t **mem_addr, __mdvdw a, __vidx_var stride);
void _mve_storer_qw(__int64_t **mem_addr, __mdvqw a, __vidx_var stride);
void _mve_storer_hf(hfloat **mem_addr, __mdvhf a, __vidx_var stride);
void _mve_storer_f(float **mem_addr, __mdvf a, __vidx_var stride);
void _mve_storer_df(double **mem_addr, __mdvdf a, __vidx_var stride);
__mdvb _mve_loadro_b(__uint8_t const **mem_addr, int offset, __vidx_var stride);
__mdvw _mve_loadro_w(__int16_t const **mem_addr, int offset, __vidx_var stride);
__mdvdw _mve_loadro_dw(__int32_t const **mem_addr, int offset, __vidx_var stride);
__mdvqw _mve_loadro_qw(__int64_t const **mem_addr, int offset, __vidx_var stride);
__mdvhf _mve_loadro_hf(hfloat const **mem_addr, int offset, __vidx_var stride);
__mdvf _mve_loadro_f(float const **mem_addr, int offset, __vidx_var stride);
__mdvdf _mve_loadro_df(double const **mem_addr, int offset, __vidx_var stride);
void _mve_storero_b(__uint8_t **mem_addr, int offset, __mdvb a, __vidx_var stride);
void _mve_storero_w(__int16_t **mem_addr, int offset, __mdvw a, __vidx_var stride);
void _mve_storero_dw(__int32_t **mem_addr, int offset, __mdvdw a, __vidx_var stride);
void _mve_storero_qw(__int64_t **mem_addr, int offset, __mdvqw a, __vidx_var stride);
void _mve_storero_hf(hfloat **mem_addr, int offset, __mdvhf a, __vidx_var stride);
void _mve_storero_f(float **mem_addr, int offset, __mdvf a, __vidx_var stride);
void _mve_storero_df(double **mem_addr, int offset, __mdvdf a, __vidx_var stride);
__mdvb _mve_dict_b(__uint8_t const *mem_addr, __mdvb a);
__mdvw _mve_dict_w(__int16_t const *mem_addr, __mdvb a);
__mdvdw _mve_dict_dw(__int32_t const *mem_addr, __mdvb a);
__mdvqw _mve_dict_qw(__int64_t const *mem_addr, __mdvb a);
__mdvb _mve_add_b(__mdvb a, __mdvb b);
__mdvw _mve_add_w(__mdvw a, __mdvw b);
__mdvdw _mve_add_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_add_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_add_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_add_f(__mdvf a, __mdvf b);
__mdvdf _mve_add_df(__mdvdf a, __mdvdf b);
__mdvb _mve_sub_b(__mdvb a, __mdvb b);
__mdvw _mve_sub_w(__mdvw a, __mdvw b);
__mdvdw _mve_sub_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_sub_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_sub_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_sub_f(__mdvf a, __mdvf b);
__mdvdf _mve_sub_df(__mdvdf a, __mdvdf b);
__mdvb _mve_mul_b(__mdvb a, __mdvb b);
__mdvw _mve_mul_w(__mdvw a, __mdvw b);
__mdvdw _mve_mul_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_mul_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_mul_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_mul_f(__mdvf a, __mdvf b);
__mdvdf _mve_mul_df(__mdvdf a, __mdvdf b);
__mdvdw _mve_mulmodp_dw(__mdvdw a, __mdvdw b);
__mdvb _mve_min_b(__mdvb a, __mdvb b);
__mdvw _mve_min_w(__mdvw a, __mdvw b);
__mdvdw _mve_min_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_min_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_min_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_min_f(__mdvf a, __mdvf b);
__mdvdf _mve_min_df(__mdvdf a, __mdvdf b);
__mdvb _mve_max_b(__mdvb a, __mdvb b);
__mdvw _mve_max_w(__mdvw a, __mdvw b);
__mdvdw _mve_max_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_max_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_max_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_max_f(__mdvf a, __mdvf b);
__mdvdf _mve_max_df(__mdvdf a, __mdvdf b);
__mdvb _mve_xor_b(__mdvb a, __mdvb b);
__mdvw _mve_xor_w(__mdvw a, __mdvw b);
__mdvdw _mve_xor_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_xor_qw(__mdvqw a, __mdvqw b);
__mdvb _mve_and_b(__mdvb a, __mdvb b);
__mdvw _mve_and_w(__mdvw a, __mdvw b);
__mdvdw _mve_and_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_and_qw(__mdvqw a, __mdvqw b);
__mdvb _mve_or_b(__mdvb a, __mdvb b);
__mdvw _mve_or_w(__mdvw a, __mdvw b);
__mdvdw _mve_or_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_or_qw(__mdvqw a, __mdvqw b);
void _mve_cmpeq_b(__mdvb a, __mdvb b);
void _mve_cmpeq_w(__mdvw a, __mdvw b);
void _mve_cmpeq_dw(__mdvdw a, __mdvdw b);
void _mve_cmpeq_qw(__mdvqw a, __mdvqw b);
void _mve_cmpeq_hf(__mdvhf a, __mdvhf b);
void _mve_cmpeq_f(__mdvf a, __mdvf b);
void _mve_cmpeq_df(__mdvdf a, __mdvdf b);
void _mve_cmpneq_b(__mdvb a, __mdvb b);
void _mve_cmpneq_w(__mdvw a, __mdvw b);
void _mve_cmpneq_dw(__mdvdw a, __mdvdw b);
void _mve_cmpneq_qw(__mdvqw a, __mdvqw b);
void _mve_cmpneq_hf(__mdvhf a, __mdvhf b);
void _mve_cmpneq_f(__mdvf a, __mdvf b);
void _mve_cmpneq_df(__mdvdf a, __mdvdf b);
void _mve_cmpgte_b(__mdvb a, __mdvb b);
void _mve_cmpgte_w(__mdvw a, __mdvw b);
void _mve_cmpgte_dw(__mdvdw a, __mdvdw b);
void _mve_cmpgte_qw(__mdvqw a, __mdvqw b);
void _mve_cmpgte_hf(__mdvhf a, __mdvhf b);
void _mve_cmpgte_f(__mdvf a, __mdvf b);
void _mve_cmpgte_df(__mdvdf a, __mdvdf b);
void _mve_cmpgt_b(__mdvb a, __mdvb b);
void _mve_cmpgt_w(__mdvw a, __mdvw b);
void _mve_cmpgt_dw(__mdvdw a, __mdvdw b);
void _mve_cmpgt_qw(__mdvqw a, __mdvqw b);
void _mve_cmpgt_hf(__mdvhf a, __mdvhf b);
void _mve_cmpgt_f(__mdvf a, __mdvf b);
void _mve_cmpgt_df(__mdvdf a, __mdvdf b);
__mdvb _mve_cvt_wtob(__mdvw a);
__mdvw _mve_cvt_dwtow(__mdvdw a);
__mdvb _mve_cvt_dwtob(__mdvdw a);
__mdvw _mve_cvtu_btow(__mdvb a);
__mdvw _mve_cvts_btow(__mdvb a);
__mdvdw _mve_cvtu_btodw(__mdvb a);
__mdvdw _mve_cvts_btodw(__mdvb a);
__mdvdw _mve_cvtu_wtodw(__mdvw a);
__mdvdw _mve_cvts_wtodw(__mdvw a);
__mdvqw _mve_cvtu_dwtoqw(__mdvdw a);
__mdvqw _mve_cvts_dwtoqw(__mdvdw a);

char *_mve_btos(__mdvb a);
char *_mve_wtos(__mdvw a);
char *_mve_dwtos(__mdvdw a);
char *_mve_qwtos(__mdvqw a);
char *_mve_hftos(__mdvhf a);
char *_mve_ftos(__mdvf a);
char *_mve_dftos(__mdvdf a);
__mdvb _mve_assign_b(__mdvb a, __mdvb b);
__mdvw _mve_assign_w(__mdvw a, __mdvw b);
__mdvdw _mve_assign_dw(__mdvdw a, __mdvdw b);
__mdvqw _mve_assign_qw(__mdvqw a, __mdvqw b);
__mdvhf _mve_assign_hf(__mdvhf a, __mdvhf b);
__mdvf _mve_assign_f(__mdvf a, __mdvf b);
__mdvdf _mve_assign_df(__mdvdf a, __mdvdf b);
void _mve_free_b();
void _mve_free_w();
void _mve_free_dw();
void _mve_free_qw();
void _mve_free_hf();
void _mve_free_f();
void _mve_free_df();
__mdvb _mve_copy_b(__mdvb a);
__mdvw _mve_copy_w(__mdvw a);
__mdvdw _mve_copy_dw(__mdvdw a);
__mdvqw _mve_copy_qw(__mdvqw a);
__mdvhf _mve_copy_hf(__mdvhf a);
__mdvf _mve_copy_f(__mdvf);
__mdvdf _mve_copy_df(__mdvdf a);

#endif
