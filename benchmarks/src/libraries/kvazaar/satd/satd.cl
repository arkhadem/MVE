__kernel void satd_adreno_kernel(
    __global unsigned char *piOrg,
    __global unsigned char *piCur,
    __constant int *coeff,
    __global int *result) {

    __local int smem1[72];
    __local int smem2[72];

    int l0_id = get_global_id(0);
    int l1_id = get_global_id(1);
    int group_id = get_global_id(2);

    int row_wise_id = l1_id * 8 + l0_id;
    int col_wise_id = l0_id * 8 + l1_id;

    int global_id = group_id * 64 + row_wise_id;

    int row_wise_padded_id = l1_id * 9 + l0_id;
    int col_wise_padded_id = l0_id * 9 + l1_id;

    smem1[row_wise_padded_id] = piOrg[global_id] - piCur[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    int acc = 0;
    __constant int *my_coeff = coeff + l0_id * 8;
    __local int *my_smem_1 = smem1 + l1_id * 9;
    for (int col = 0; col < 8; col++) {
        acc += (*my_coeff * *my_smem_1);
        my_coeff += 1;
        my_smem_1 += 1;
    }

    smem2[row_wise_padded_id] = acc;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    acc = 0;
    my_coeff = coeff + l1_id * 8;
    __local int *my_smem_2 = smem2 + l0_id;
    for (int row = 0; row < 8; row++) {
        acc += (*my_coeff * *my_smem_2);
        my_coeff += 1;
        my_smem_2 += 9;
    }

    result[global_id] = acc;

}