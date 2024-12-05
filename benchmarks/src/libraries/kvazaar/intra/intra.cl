__kernel void intra_adreno_kernel(
    __global unsigned char* ref_top,
    __global unsigned char* ref_left,
    __global unsigned char* dst) {

    __local short smem_top[17];
    __local short smem_left[17];

    int l0_id = get_global_id(0);
    int l1_id = get_global_id(1);
    int group_id = get_global_id(2);

    int dst_id = l1_id * 8 + l0_id;

    if (dst_id < 17) {
        smem_top[dst_id] = ref_top[group_id * 17 + dst_id];
        smem_left[dst_id] = ref_left[group_id * 17 + dst_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    short top_right = smem_top[9];
    short bottom_left = smem_left[9];

    short result =    (7 - l0_id) * smem_left[l1_id + 1] +
                        top_right * (l0_id + 1) +
                        (7 - l1_id) * smem_top[l0_id + 1] +
                        bottom_left * (l1_id + 1) +
                        8;

    dst[group_id * 64 + dst_id] = result >> 4;
}