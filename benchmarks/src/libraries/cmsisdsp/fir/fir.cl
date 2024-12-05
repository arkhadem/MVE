#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__kernel void fir_adreno_kernel(
    int sample_count,
    int coeff_count,
    __global int *src,
    __constant int *coeff,
    __global int *dst,
    __local int* smem) {

    int input_size = sample_count + coeff_count - 1;
    int l_id = get_local_id(0);

    int g_id = get_group_id(0) * BLOCK_SIZE + l_id;

    // load cells
    if (g_id < input_size) {
        smem[l_id] = src[g_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // check if it's a worker thread
    if (l_id < BLOCK_SIZE) {
        // check if it's within the range
        if (g_id < sample_count) {
            __local int* my_smem = smem + l_id;
            __constant int *my_coeff = coeff;
            
            int src_temp;
            int coeff_temp;
            int acc = 0;
            for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
                src_temp = *my_smem;
                coeff_temp = *my_coeff;
                acc += src_temp * coeff_temp;
                my_smem += 1;
                my_coeff += 1;
            }

            dst[g_id] = acc;
            // printf("[%d] = %f\n", g_id, acc);
        }
    }

}