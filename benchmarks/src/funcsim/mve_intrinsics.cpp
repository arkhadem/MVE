#include "mve.hpp"
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <string>

int idx_cvt(__dim_var dims[4], int dim3, int dim2, int dim1, int dim0) {
    int idx = (((((dim3 * dims[2].length) + dim2) * dims[1].length) + dim1) * dims[0].length) + dim0;
    return idx;
}

void mve_initializer(char *exp_name, int LANE_NUM) { mve::initializer(exp_name, LANE_NUM); }

void mve_init_dims() { mve::init_dims(); }

void mve_finisher() { mve::finisher(); }

void mve_flusher() { mve::flusher(); }

void _mve_set_load_stride(int dim, int stride) { mve::new_operation("_mve_set_load_stride", dim, stride); }

void _mve_set_store_stride(int dim, int stride) { mve::new_operation("_mve_set_store_stride", dim, stride); }

void _mve_set_dim_count(int count) { mve::new_operation("_mve_set_dim_count", count); }

void _mve_set_dim_length(int dim, int length) { mve::new_operation("_mve_set_dim_length", dim, length); }

void _mve_set_mask() {
    mve::set_mask();
    mve::new_operation("_mve_set_mask");
}

// Activates a certain element (no effect on other elements)
void _mve_set_active_element(int dim, int element) { mve::new_operation("_mve_set_active_element", dim, element); }

// Deactivates a certain element (no effect on other elements)
void _mve_unset_active_element(int dim, int element) { mve::new_operation("_mve_unset_active_element", dim, element); }

// Activates only a certain element and deactivates others
void _mve_set_only_element(int dim, int element) { mve::new_operation("_mve_set_only_element", dim, element); }

// Deactivates only a certain element and activates others
void _mve_unset_only_element(int dim, int element) { mve::new_operation("_mve_unset_only_element", dim, element); }

// Activates all elements
void _mve_set_all_elements(int dim) { mve::new_operation("_mve_set_all_elements", dim, -1); }

// Deactivates all elements
void _mve_unset_all_elements(int dim) { mve::new_operation("_mve_unset_all_elements", dim, -1); }

// Activates firsthalf elements
void _mve_set_firsthalf_elements(int dim) { mve::new_operation("_mve_set_firsthalf_elements", dim, -1); }

// Deactivates firsthalf elements
void _mve_unset_firsthalf_elements(int dim) { mve::new_operation("_mve_unset_firsthalf_elements", dim, -1); }

// Activates secondhalf elements
void _mve_set_secondhalf_elements(int dim) { mve::new_operation("_mve_set_secondhalf_elements", dim, -1); }

// Deactivates secondhalf elements
void _mve_unset_secondhalf_elements(int dim) { mve::new_operation("_mve_unset_secondhalf_elements", dim, -1); }

// Shift immidiate right signed
__mdvb _mve_shirs_b(__mdvb a, int imm) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shirs_b", a.id, (double)imm, o_var);
    return o;
}
__mdvw _mve_shirs_w(__mdvw a, int imm) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shirs_w", a.id, (double)imm, o_var);
    return o;
}
__mdvdw _mve_shirs_dw(__mdvdw a, int imm) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shirs_dw", a.id, (double)imm, o_var);
    return o;
}
__mdvqw _mve_shirs_qw(__mdvqw a, int imm) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shirs_qw", a.id, (double)imm, o_var);
    return o;
}

// Shift immidiate right unsigned
__mdvb _mve_shiru_b(__mdvb a, int imm) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = (__uint8_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shiru_b", a.id, (double)imm, o_var);
    return o;
}
__mdvw _mve_shiru_w(__mdvw a, int imm) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (uint16_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shiru_w", a.id, (double)imm, o_var);
    return o;
}
__mdvdw _mve_shiru_dw(__mdvdw a, int imm) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (uint32_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shiru_dw", a.id, (double)imm, o_var);
    return o;
}
__mdvqw _mve_shiru_qw(__mdvqw a, int imm) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (uint64_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shiru_qw", a.id, (double)imm, o_var);
    return o;
}

// Shift immidiate left
__mdvb _mve_shil_b(__mdvb a, int imm) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b << imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shil_b", a.id, (double)imm, o_var);
    return o;
}
__mdvw _mve_shil_w(__mdvw a, int imm) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w << imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shil_w", a.id, (double)imm, o_var);
    return o;
}
__mdvdw _mve_shil_dw(__mdvdw a, int imm) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw << imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shil_dw", a.id, (double)imm, o_var);
    return o;
}
__mdvqw _mve_shil_qw(__mdvqw a, int imm) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw << imm;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shil_qw", a.id, (double)imm, o_var);
    return o;
}

// Rotate immidiate right
__mdvb _mve_rotir_b(__mdvb a, int imm) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b << (8 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotir_b", a.id, (double)imm, o_var);
    return o;
}
__mdvw _mve_rotir_w(__mdvw a, int imm) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w << (16 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotir_w", a.id, (double)imm, o_var);
    return o;
}
__mdvdw _mve_rotir_dw(__mdvdw a, int imm) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw << (32 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotir_dw", a.id, (double)imm, o_var);
    return o;
}
__mdvqw _mve_rotir_qw(__mdvqw a, int imm) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw << (64 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotir_qw", a.id, (double)imm, o_var);
    return o;
}

// Rotate immidiate left
__mdvb _mve_rotil_b(__mdvb a, int imm) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b << imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> (8 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotil_b", a.id, (double)imm, o_var);
    return o;
}
__mdvw _mve_rotil_w(__mdvw a, int imm) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w << imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> (16 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotil_w", a.id, (double)imm, o_var);
    return o;
}
__mdvdw _mve_rotil_dw(__mdvdw a, int imm) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw << imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> (32 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotil_dw", a.id, (double)imm, o_var);
    return o;
}
__mdvqw _mve_rotil_qw(__mdvqw a, int imm) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw << imm) | (a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> (64 - imm));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_rotil_qw", a.id, (double)imm, o_var);
    return o;
}

// Shift register right signed
__mdvb _mve_shrrs_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrrs_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_shrrs_w(__mdvw a, __mdvb b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrrs_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_shrrs_dw(__mdvdw a, __mdvb b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrrs_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_shrrs_qw(__mdvqw a, __mdvb b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrrs_qw", a.id, b.id, o_var);
    return o;
}

// Shift register right unsigned
__mdvb _mve_shrru_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = (__uint8_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrru_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_shrru_w(__mdvw a, __mdvb b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (uint16_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrru_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_shrru_dw(__mdvdw a, __mdvb b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (uint32_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrru_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_shrru_qw(__mdvqw a, __mdvb b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (uint64_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >> b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrru_qw", a.id, b.id, o_var);
    return o;
}

// Shift register left
__mdvb _mve_shrl_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b << b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrl_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_shrl_w(__mdvw a, __mdvb b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w << b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrl_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_shrl_dw(__mdvdw a, __mdvb b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw << b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrl_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_shrl_qw(__mdvqw a, __mdvb b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw << b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_shrl_qw", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_set1_b(__uint8_t const a) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_b", (double)a, o_var);
    return o;
}
__mdvw _mve_set1_w(int16_t const a) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_w", (double)a, o_var);
    return o;
}
__mdvdw _mve_set1_dw(int32_t const a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_dw", (double)a, o_var);
    return o;
}
__mdvqw _mve_set1_qw(int64_t const a) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_qw", (double)a, o_var);
    return o;
}
__mdvhf _mve_set1_hf(hfloat const a) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_hf", (double)a, o_var);
    return o;
}
__mdvf _mve_set1_f(float const a) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_f", (double)a, o_var);
    return o;
}
__mdvdf _mve_set1_df(double const a) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_set1_df", (double)a, o_var);
    return o;
}

__mdvb _mve_load_b(__uint8_t const *mem_addr, __vidx_var stride) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_b", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvw _mve_load_w(int16_t const *mem_addr, __vidx_var stride) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_w", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdw _mve_load_dw(int32_t const *mem_addr, __vidx_var stride) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                                }
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_dw", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvqw _mve_load_qw(int64_t const *mem_addr, __vidx_var stride) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_qw", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvhf _mve_load_hf(hfloat const *mem_addr, __vidx_var stride) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_hf", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvf _mve_load_f(float const *mem_addr, __vidx_var stride) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_f", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdf _mve_load_df(double const *mem_addr, __vidx_var stride) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0]));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_load_df", (void const *)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}

__mdvb _mve_dict_b(__uint8_t const *mem_addr, __mdvb a) {
    __mdvb o;
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = *(mem_addr + a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_dict_b", (void const *)mem_addr, a.id, o_var);
    return o;
}
__mdvw _mve_dict_w(int16_t const *mem_addr, __mdvb a) {
    __mdvw o;
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = *(mem_addr + a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_dict_w", (void const *)mem_addr, a.id, o_var);
    return o;
}
__mdvdw _mve_dict_dw(int32_t const *mem_addr, __mdvb a) {
    __mdvdw o;
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = *(mem_addr + a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                                }
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_dict_dw", (void const *)mem_addr, a.id, o_var);
    return o;
}
__mdvqw _mve_dict_qw(int64_t const *mem_addr, __mdvb a) {
    __mdvqw o;
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var o_var(mve::dims);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = *(mem_addr + a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_dict_qw", (void const *)mem_addr, a.id, o_var);
    return o;
}

void _mve_store_b(__uint8_t *mem_addr, __mdvb a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_b", a.id, (void const *)mem_addr, stride);
}
void _mve_store_w(int16_t *mem_addr, __mdvw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_w", a.id, (void const *)mem_addr, stride);
}
void _mve_store_dw(int32_t *mem_addr, __mdvdw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                                }
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_dw", a.id, (void const *)mem_addr, stride);
}
void _mve_store_qw(int64_t *mem_addr, __mdvqw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_qw", a.id, (void const *)mem_addr, stride);
}
void _mve_store_hf(hfloat *mem_addr, __mdvhf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_hf", a.id, (void const *)mem_addr, stride);
}
void _mve_store_f(float *mem_addr, __mdvf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_f", a.id, (void const *)mem_addr, stride);
}
void _mve_store_df(double *mem_addr, __mdvdf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    *(mem_addr + (dim3 * lstride[3] + dim2 * lstride[2] + dim1 * lstride[1] + dim0 * lstride[0])) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    delete[] lstride;
    mve::new_operation("_mve_store_df", a.id, (void const *)mem_addr, stride);
}

__mdvb _mve_loadr_b(__uint8_t const **mem_addr, __vidx_var stride) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    __uint8_t const *mem_addr_temp3;
    __uint8_t const *mem_addr_temp2;
    __uint8_t const *mem_addr_temp1;
    __uint8_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) {
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = *(mem_addr_temp0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_b", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvw _mve_loadr_w(int16_t const **mem_addr, __vidx_var stride) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int16_t const *mem_addr_temp3;
    int16_t const *mem_addr_temp2;
    int16_t const *mem_addr_temp1;
    int16_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_w", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdw _mve_loadr_dw(int32_t const **mem_addr, __vidx_var stride) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int32_t const *mem_addr_temp3;
    int32_t const *mem_addr_temp2;
    int32_t const *mem_addr_temp1;
    int32_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_dw", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvqw _mve_loadr_qw(int64_t const **mem_addr, __vidx_var stride) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int64_t const *mem_addr_temp3;
    int64_t const *mem_addr_temp2;
    int64_t const *mem_addr_temp1;
    int64_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_qw", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvhf _mve_loadr_hf(hfloat const **mem_addr, __vidx_var stride) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    hfloat const *mem_addr_temp3;
    hfloat const *mem_addr_temp2;
    hfloat const *mem_addr_temp1;
    hfloat const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_hf", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvf _mve_loadr_f(float const **mem_addr, __vidx_var stride) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    float const *mem_addr_temp3;
    float const *mem_addr_temp2;
    float const *mem_addr_temp1;
    float const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_f", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdf _mve_loadr_df(double const **mem_addr, __vidx_var stride) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    double const *mem_addr_temp3;
    double const *mem_addr_temp2;
    double const *mem_addr_temp1;
    double const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_df", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}

__mdvb _mve_loadro_b(__uint8_t const **mem_addr, int offset, __vidx_var stride) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    __uint8_t const *mem_addr_temp3;
    __uint8_t const *mem_addr_temp2;
    __uint8_t const *mem_addr_temp1;
    __uint8_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) {
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = *(mem_addr_temp0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_b", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvw _mve_loadro_w(int16_t const **mem_addr, int offset, __vidx_var stride) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int16_t const *mem_addr_temp3;
    int16_t const *mem_addr_temp2;
    int16_t const *mem_addr_temp1;
    int16_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_w", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdw _mve_loadro_dw(int32_t const **mem_addr, int offset, __vidx_var stride) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int32_t const *mem_addr_temp3;
    int32_t const *mem_addr_temp2;
    int32_t const *mem_addr_temp1;
    int32_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_dw", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvqw _mve_loadro_qw(int64_t const **mem_addr, int offset, __vidx_var stride) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    int64_t const *mem_addr_temp3;
    int64_t const *mem_addr_temp2;
    int64_t const *mem_addr_temp1;
    int64_t const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_qw", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvhf _mve_loadro_hf(hfloat const **mem_addr, int offset, __vidx_var stride) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    hfloat const *mem_addr_temp3;
    hfloat const *mem_addr_temp2;
    hfloat const *mem_addr_temp1;
    hfloat const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_hf", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvf _mve_loadro_f(float const **mem_addr, int offset, __vidx_var stride) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    float const *mem_addr_temp3;
    float const *mem_addr_temp2;
    float const *mem_addr_temp1;
    float const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_f", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}
__mdvdf _mve_loadro_df(double const **mem_addr, int offset, __vidx_var stride) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    int *lstride = mve::stride_evaluator(stride, true);
    assert(mve::dim_count > 1);
    double const *mem_addr_temp3;
    double const *mem_addr_temp2;
    double const *mem_addr_temp1;
    double const *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = *(mem_addr_temp0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_loadr_df", (void const **)mem_addr, o_var, stride);
    delete[] lstride;
    return o;
}

void _mve_storer_b(__uint8_t **mem_addr, __mdvb a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    __uint8_t *mem_addr_temp3;
    __uint8_t *mem_addr_temp2;
    __uint8_t *mem_addr_temp1;
    __uint8_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_b", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_w(int16_t **mem_addr, __mdvw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int16_t *mem_addr_temp3;
    int16_t *mem_addr_temp2;
    int16_t *mem_addr_temp1;
    int16_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_w", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_dw(int32_t **mem_addr, __mdvdw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int32_t *mem_addr_temp3;
    int32_t *mem_addr_temp2;
    int32_t *mem_addr_temp1;
    int32_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_dw", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_qw(int64_t **mem_addr, __mdvqw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int64_t *mem_addr_temp3;
    int64_t *mem_addr_temp2;
    int64_t *mem_addr_temp1;
    int64_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_qw", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_hf(hfloat **mem_addr, __mdvhf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    hfloat *mem_addr_temp3;
    hfloat *mem_addr_temp2;
    hfloat *mem_addr_temp1;
    hfloat *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_hf", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_f(float **mem_addr, __mdvf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    float *mem_addr_temp3;
    float *mem_addr_temp2;
    float *mem_addr_temp1;
    float *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_f", a.id, (void const **)mem_addr, stride);
}
void _mve_storer_df(double **mem_addr, __mdvdf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    double *mem_addr_temp3;
    double *mem_addr_temp2;
    double *mem_addr_temp1;
    double *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3];
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2];
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1];
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0];
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_df", a.id, (void const **)mem_addr, stride);
}

void _mve_storero_b(__uint8_t **mem_addr, int offset, __mdvb a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    __uint8_t *mem_addr_temp3;
    __uint8_t *mem_addr_temp2;
    __uint8_t *mem_addr_temp1;
    __uint8_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_b", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_w(int16_t **mem_addr, int offset, __mdvw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int16_t *mem_addr_temp3;
    int16_t *mem_addr_temp2;
    int16_t *mem_addr_temp1;
    int16_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_w", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_dw(int32_t **mem_addr, int offset, __mdvdw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int32_t *mem_addr_temp3;
    int32_t *mem_addr_temp2;
    int32_t *mem_addr_temp1;
    int32_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_dw", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_qw(int64_t **mem_addr, int offset, __mdvqw a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    int64_t *mem_addr_temp3;
    int64_t *mem_addr_temp2;
    int64_t *mem_addr_temp1;
    int64_t *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_qw", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_hf(hfloat **mem_addr, int offset, __mdvhf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    hfloat *mem_addr_temp3;
    hfloat *mem_addr_temp2;
    hfloat *mem_addr_temp1;
    hfloat *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_hf", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_f(float **mem_addr, int offset, __mdvf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    float *mem_addr_temp3;
    float *mem_addr_temp2;
    float *mem_addr_temp1;
    float *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_f", a.id, (void const **)mem_addr, stride);
}
void _mve_storero_df(double **mem_addr, int offset, __mdvdf a, __vidx_var stride) {
    __mdv_var a_var = mve::get_value(a.id);
    int *lstride = mve::stride_evaluator(stride, false);
    assert(mve::dim_count > 1);
    double *mem_addr_temp3;
    double *mem_addr_temp2;
    double *mem_addr_temp1;
    double *mem_addr_temp0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3]) {
            if (mve::dim_count == 4)
                mem_addr_temp3 = mem_addr[dim3] + offset;
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2]) {
                    if (mve::dim_count == 3)
                        mem_addr_temp2 = mem_addr[dim2] + offset;
                    else
                        mem_addr_temp2 = mem_addr_temp3 + dim2 * lstride[2];
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1]) {
                            if (mve::dim_count == 2)
                                mem_addr_temp1 = mem_addr[dim1] + offset;
                            else
                                mem_addr_temp1 = mem_addr_temp2 + dim1 * lstride[1];
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0]) {
                                    if (mve::dim_count == 1)
                                        mem_addr_temp0 = mem_addr[dim0] + offset;
                                    else
                                        mem_addr_temp0 = mem_addr_temp1 + dim0 * lstride[0];
                                    *(mem_addr_temp0) = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] lstride;
    mve::new_operation("_mve_storer_df", a.id, (void const **)mem_addr, stride);
}

__mdvb _mve_add_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_add_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_add_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_add_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_qw", a.id, b.id, o_var);
    return o;
}
__mdvhf _mve_add_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_hf", a.id, b.id, o_var);
    return o;
}
__mdvf _mve_add_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_f", a.id, b.id, o_var);
    return o;
}
__mdvdf _mve_add_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df + b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_add_df", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_sub_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_sub_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_sub_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_sub_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_qw", a.id, b.id, o_var);
    return o;
}
__mdvhf _mve_sub_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_hf", a.id, b.id, o_var);
    return o;
}
__mdvf _mve_sub_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_f", a.id, b.id, o_var);
    return o;
}
__mdvdf _mve_sub_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df - b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_sub_df", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_mul_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_mul_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_mul_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_mul_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_qw", a.id, b.id, o_var);
    return o;
}
__mdvhf _mve_mul_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_hf", a.id, b.id, o_var);
    return o;
}
__mdvf _mve_mul_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_f", a.id, b.id, o_var);
    return o;
}
__mdvdf _mve_mul_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df * b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mul_df", a.id, b.id, o_var);
    return o;
}
#define multmodp(p, a, b)                           \
    m = (uint32_t)1 << 31;                          \
    p = 0;                                          \
    for (;;) {                                      \
        if (a & m) {                                \
            p ^= b;                                 \
            if ((a & (m - 1)) == 0)                 \
                break;                              \
        }                                           \
        m >>= 1;                                    \
        b = b & 1 ? (b >> 1) ^ 0xedb88320 : b >> 1; \
    }

__mdvdw _mve_mulmodp_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    uint32_t m;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) {
                                        multmodp(o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw, b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw);
                                    }
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_mulmodp_dw", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_min_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_min_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_min_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_min_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_qw", a.id, b.id, o_var);
    return o;
}
__mdvhf _mve_min_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_hf", a.id, b.id, o_var);
    return o;
}
__mdvf _mve_min_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_f", a.id, b.id, o_var);
    return o;
}
__mdvdf _mve_min_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df < b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_min_df", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_max_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_max_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_max_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_max_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_qw", a.id, b.id, o_var);
    return o;
}
__mdvhf _mve_max_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_hf", a.id, b.id, o_var);
    return o;
}
__mdvf _mve_max_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_f", a.id, b.id, o_var);
    return o;
}
__mdvdf _mve_max_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df ? a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df : b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_max_df", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_xor_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b ^ b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_xor_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_xor_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w ^ b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_xor_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_xor_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw ^ b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_xor_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_xor_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw ^ b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_xor_qw", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_and_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b & b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_and_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_and_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w & b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_and_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_and_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw & b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_and_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_and_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw & b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_and_qw", a.id, b.id, o_var);
    return o;
}

__mdvb _mve_or_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b | b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_or_b", a.id, b.id, o_var);
    return o;
}
__mdvw _mve_or_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w | b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_or_w", a.id, b.id, o_var);
    return o;
}
__mdvdw _mve_or_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw | b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_or_dw", a.id, b.id, o_var);
    return o;
}
__mdvqw _mve_or_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw | b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_or_qw", a.id, b.id, o_var);
    return o;
}

void _mve_cmpeq_b(__mdvb a, __mdvb b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                                // if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                //     printf("[%d][%d][%d][%d]: %d = %d\n", dim3, dim2, dim1, dim0, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b, b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_b", a.id, b.id);
}
void _mve_cmpeq_w(__mdvw a, __mdvw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_w", a.id, b.id);
}
void _mve_cmpeq_dw(__mdvdw a, __mdvdw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_dw", a.id, b.id);
}
void _mve_cmpeq_qw(__mdvqw a, __mdvqw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_qw", a.id, b.id);
}
void _mve_cmpeq_hf(__mdvhf a, __mdvhf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_hf", a.id, b.id);
}
void _mve_cmpeq_f(__mdvf a, __mdvf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_f", a.id, b.id);
}
void _mve_cmpeq_df(__mdvdf a, __mdvdf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df == b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpeq_df", a.id, b.id);
}

void _mve_cmpneq_b(__mdvb a, __mdvb b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_b", a.id, b.id);
}
void _mve_cmpneq_w(__mdvw a, __mdvw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_w", a.id, b.id);
}
void _mve_cmpneq_dw(__mdvdw a, __mdvdw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_dw", a.id, b.id);
}
void _mve_cmpneq_qw(__mdvqw a, __mdvqw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_qw", a.id, b.id);
}
void _mve_cmpneq_hf(__mdvhf a, __mdvhf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_hf", a.id, b.id);
}
void _mve_cmpneq_f(__mdvf a, __mdvf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_f", a.id, b.id);
}
void _mve_cmpneq_df(__mdvdf a, __mdvdf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df != b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpneq_df", a.id, b.id);
}

void _mve_cmpgte_b(__mdvb a, __mdvb b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_b", a.id, b.id);
}
void _mve_cmpgte_w(__mdvw a, __mdvw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_w", a.id, b.id);
}
void _mve_cmpgte_dw(__mdvdw a, __mdvdw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_dw", a.id, b.id);
}
void _mve_cmpgte_qw(__mdvqw a, __mdvqw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_qw", a.id, b.id);
}
void _mve_cmpgte_hf(__mdvhf a, __mdvhf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_hf", a.id, b.id);
}
void _mve_cmpgte_f(__mdvf a, __mdvf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_f", a.id, b.id);
}
void _mve_cmpgte_df(__mdvdf a, __mdvdf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df >= b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgte_df", a.id, b.id);
}

void _mve_cmpgt_b(__mdvb a, __mdvb b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_b", a.id, b.id);
}
void _mve_cmpgt_w(__mdvw a, __mdvw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_w", a.id, b.id);
}
void _mve_cmpgt_dw(__mdvdw a, __mdvdw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_dw", a.id, b.id);
}
void _mve_cmpgt_qw(__mdvqw a, __mdvqw b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_qw", a.id, b.id);
}
void _mve_cmpgt_hf(__mdvhf a, __mdvhf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_hf", a.id, b.id);
}
void _mve_cmpgt_f(__mdvf a, __mdvf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_f", a.id, b.id);
}
void _mve_cmpgt_df(__mdvdf a, __mdvdf b) {
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)] = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df > b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                            }
                    }
            }
    }
    mve::new_operation("_mve_cmpgt_df", a.id, b.id);
}

__mdvb _mve_cvt_wtob(__mdvw a) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvt_wtob", a.id, o_var);
    return o;
}
__mdvw _mve_cvt_dwtow(__mdvdw a) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvt_dwtow", a.id, o_var);
    return o;
}
__mdvb _mve_cvt_dwtob(__mdvdw a) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvt_dwtob", a.id, o_var);
    return o;
}
__mdvw _mve_cvtu_btow(__mdvb a) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (int16_t)((uint16_t)((__uint8_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvtu_btow", a.id, o_var);
    return o;
}
__mdvw _mve_cvts_btow(__mdvb a) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = (int16_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvts_btow", a.id, o_var);
    return o;
}

__mdvdw _mve_cvtu_btodw(__mdvb a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (int32_t)((uint32_t)((__uint8_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvtu_btodw", a.id, o_var);
    return o;
}
__mdvdw _mve_cvts_btodw(__mdvb a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (int32_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvts_btodw", a.id, o_var);
    return o;
}

__mdvdw _mve_cvtu_wtodw(__mdvw a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (int32_t)((uint32_t)((uint16_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvtu_wtodw", a.id, o_var);
    return o;
}
__mdvdw _mve_cvts_wtodw(__mdvw a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = (int32_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvts_wtodw", a.id, o_var);
    return o;
}

__mdvqw _mve_cvtu_dwtoqw(__mdvdw a) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (int64_t)((uint64_t)((uint32_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw));
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvtu_dwtoqw", a.id, o_var);
    return o;
}

__mdvqw _mve_cvts_dwtoqw(__mdvdw a) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].mask[dim3])
            for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
                if (mve::dims[2].mask[dim2])
                    for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                        if (mve::dims[1].mask[dim1])
                            for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                                if (mve::dims[0].mask[dim0])
                                    if (mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)])
                                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = (int64_t)a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                            }
                    }
            }
    }
    o.id = mve::new_operation("_mve_cvts_dwtoqw", a.id, o_var);
    return o;
}

void print_qw(__mdvqw qw) {
    char *print_tmp = _mve_qwtos(qw);
    printf("%s\n", print_tmp);
}

void print_dw(__mdvdw dw) {
    char *print_tmp = _mve_dwtos(dw);
    printf("%s\n", print_tmp);
}

char *tab_maker(int num) {
    char *tabs = new char[50];
    memset(tabs, '\0', 50 * sizeof(char));
    for (int tab = 0; tab < num; tab++) {
        sprintf(tabs, "%s    ", tabs);
    }
    return tabs;
}

char *init_string() {
    int size = mve::dims[3].length * mve::dims[2].length * mve::dims[1].length * mve::dims[0].length * 16;
    char *str = new char[size];
    memset(str, '\0', size * sizeof(char));
    return str;
}

char *_mve_btos(__mdvb a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %d,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_wtos(__mdvw a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %d,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_dwtos(__mdvdw a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %d,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_qwtos(__mdvqw a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %ld,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_hftos(__mdvhf a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %d,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_ftos(__mdvf a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %f,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}
char *_mve_dftos(__mdvdf a) {
    __mdv_var a_var = mve::get_value(a.id);
    char *str = init_string();
    int tabs = 0;
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        if (mve::dims[3].length != 1) {
            sprintf(str, "%s%s[\n", str, tab_maker(tabs));
            tabs += 1;
        }
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            if (mve::dims[2].length != 1) {
                sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                tabs += 1;
            }
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                if (mve::dims[1].length != 1) {
                    sprintf(str, "%s%s[\n", str, tab_maker(tabs));
                    tabs += 1;
                }
                sprintf(str, "%s%s", str, tab_maker(tabs));
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    sprintf(str, "%s %lf,", str, a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df);
                }
                if (mve::dims[1].length != 1) {
                    tabs -= 1;
                    sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
                }
            }
            if (mve::dims[2].length != 1) {
                tabs -= 1;
                sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
            }
        }
        if (mve::dims[3].length != 1) {
            tabs -= 1;
            sprintf(str, "%s%s\n]\n", str, tab_maker(tabs));
        }
    }
    return str;
}

__mdvb _mve_assign_b(__mdvb a, __mdvb b) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_b", a.id, b.id, o_var);
    return o;
}

__mdvw _mve_assign_w(__mdvw a, __mdvw b) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_w", a.id, b.id, o_var);
    return o;
}

__mdvdw _mve_assign_dw(__mdvdw a, __mdvdw b) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_dw", a.id, b.id, o_var);
    return o;
}

__mdvqw _mve_assign_qw(__mdvqw a, __mdvqw b) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_qw", a.id, b.id, o_var);
    return o;
}

__mdvhf _mve_assign_hf(__mdvhf a, __mdvhf b) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_hf", a.id, b.id, o_var);
    return o;
}

__mdvf _mve_assign_f(__mdvf a, __mdvf b) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_f", a.id, b.id, o_var);
    return o;
}

__mdvdf _mve_assign_df(__mdvdf a, __mdvdf b) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    __mdv_var b_var = mve::get_value(b.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    if ((mve::mask[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)]) && (mve::dims[3].mask[dim3]) && (mve::dims[2].mask[dim2]) && (mve::dims[1].mask[dim1]) && (mve::dims[0].mask[dim0]))
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = b_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                    else
                        o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_assign_df", a.id, b.id, o_var);
    return o;
}

void _mve_free_b() {
    mve::free_register();
    mve::new_operation("_mve_free_b");
}
void _mve_free_w() {
    mve::free_register();
    mve::new_operation("_mve_free_w");
}
void _mve_free_dw() {
    mve::free_register();
    mve::new_operation("_mve_free_dw");
}
void _mve_free_qw() {
    mve::free_register();
    mve::new_operation("_mve_free_qw");
}
void _mve_free_hf() {
    mve::free_register();
    mve::new_operation("_mve_free_hf");
}
void _mve_free_f() {
    mve::free_register();
    mve::new_operation("_mve_free_f");
}
void _mve_free_df() {
    mve::free_register();
    mve::new_operation("_mve_free_df");
}

__mdvb _mve_copy_b(__mdvb a) {
    __mdvb o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_b", a.id, o_var);
    return o;
}

__mdvw _mve_copy_w(__mdvw a) {
    __mdvw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_w", a.id, o_var);
    return o;
}

__mdvdw _mve_copy_dw(__mdvdw a) {
    __mdvdw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_dw", a.id, o_var);
    return o;
}

__mdvqw _mve_copy_qw(__mdvqw a) {
    __mdvqw o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_qw", a.id, o_var);
    return o;
}

__mdvhf _mve_copy_hf(__mdvhf a) {
    __mdvhf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_hf", a.id, o_var);
    return o;
}

__mdvf _mve_copy_f(__mdvf a) {
    __mdvf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_f", a.id, o_var);
    return o;
}

__mdvdf _mve_copy_df(__mdvdf a) {
    __mdvdf o;
    __mdv_var o_var(mve::dims);
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    o_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df = a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                }
            }
        }
    }
    o.id = mve::new_operation("_mve_copy_df", a.id, o_var);
    return o;
}

__uint8_t _mve_redsum_b(__mdvb a) {
    __uint8_t result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].b;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_b", a.id);
    return result;
}

int16_t _mve_redsum_w(__mdvw a) {
    int16_t result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].w;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_w", a.id);
    return result;
}

int32_t _mve_redsum_dw(__mdvdw a) {
    int32_t result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].dw;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_dw", a.id);
    return result;
}

int64_t _mve_redsum_qw(__mdvqw a) {
    int64_t result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].qw;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_qw", a.id);
    return result;
}

hfloat _mve_redsum_hf(__mdvhf a) {
    hfloat result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].hf;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_hf", a.id);
    return result;
}

float _mve_redsum_f(__mdvf a) {
    float result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].f;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_f", a.id);
    return result;
}

double _mve_redsum_df(__mdvdf a) {
    double result = 0;
    __mdv_var a_var = mve::get_value(a.id);
    for (int dim3 = 0; dim3 < mve::dims[3].length; dim3++) {
        for (int dim2 = 0; dim2 < mve::dims[2].length; dim2++) {
            for (int dim1 = 0; dim1 < mve::dims[1].length; dim1++) {
                for (int dim0 = 0; dim0 < mve::dims[0].length; dim0++) {
                    result += a_var.val[idx_cvt(mve::dims, dim3, dim2, dim1, dim0)].df;
                }
            }
        }
    }
    mve::new_operation("_mve_redsum_df", a.id);
    return result;
}