#include "mve_variables.hpp"

#include <assert.h>
#include <stdio.h>

__mdv_var::__mdv_var(__dim_var dims[4]) {
    for (int dim = 0; dim < 4; dim++)
        my_dims[dim] = dims[dim];
    val = new __scalar_var[total_length()];
}
__mdv_var::__mdv_var(const __mdv_var &other) {
    for (int dim = 0; dim < 4; dim++)
        my_dims[dim] = other.my_dims[dim];
    val = new __scalar_var[total_length()];
    for (int dim = 0; dim < total_length(); dim++) {
        val[dim] = other.val[dim];
    }
}
__mdv_var &__mdv_var::operator=(const __mdv_var other) {
    for (int dim = 0; dim < 4; dim++)
        my_dims[dim] = other.my_dims[dim];
    val = new __scalar_var[total_length()];
    for (int dim = 0; dim < total_length(); dim++) {
        val[dim] = other.val[dim];
    }
    return *this;
}
__mdv_var::~__mdv_var() {
    if (val == NULL)
        return;
    delete[] val;
    val = NULL;
}
int __mdv_var::idx_cvt(int dim3, int dim2, int dim1, int dim0) {
    int idx = (((((dim3 * my_dims[2].length) + dim2) * my_dims[1].length) + dim1) * my_dims[0].length) + dim0;
    if (idx >= total_length()) {
        printf("Error: idx[%d][%d][%d][%d] or idx[%d] >= total length[%d]", dim3, dim2, dim1, dim0, idx, total_length());
        assert(false);
    }
    return idx;
}
int __mdv_var::total_length() {
    return my_dims[3].length * my_dims[2].length * my_dims[1].length * my_dims[0].length;
}