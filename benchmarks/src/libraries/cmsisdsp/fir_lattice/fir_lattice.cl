#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__kernel void fir_lattice_adreno_kernel(
    int sample_count,
    int coeff_count,
    __global int *src,
    __constant int *coeff,
    __global int *dst,
    __global int *initial_g_in,
    __global int *initial_g_out) {

    __local int smem[BLOCK_SIZE];

    int l_id = get_local_id(0);
    int l_id_p1 = l_id + 1;

    __local int *g_prev = smem + l_id;
    __local int *g_curr = smem + l_id_p1;
    __global int *initial_g_in_curr = initial_g_in;
    __global int *initial_g_out_curr = initial_g_out;
    bool first_thread = (l_id == 0);
    bool last_thread = (l_id == sample_count - 1);

    int f_prev = src[l_id];

    if (last_thread == false) {
        *g_curr = f_prev;
    } else {
        *initial_g_out_curr = f_prev;
        initial_g_out_curr += 1;
    }

    __constant int *my_coeff = coeff;
    int g_prev_temp;
    int g_curr_temp;
    int coeff_temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int coeff_idx = 1; coeff_idx <= coeff_count; coeff_idx++) {
        coeff_temp = *my_coeff;

        if (first_thread) {
            g_prev_temp = *initial_g_in_curr;
            initial_g_in_curr++;
        } else {
            g_prev_temp = *g_prev;
        }

        g_curr_temp = coeff_temp * f_prev + g_prev_temp;
        f_prev += coeff_temp * g_prev_temp;

        barrier(CLK_LOCAL_MEM_FENCE);
        
        my_coeff += 1;

        if (last_thread) {
            *initial_g_out_curr = g_curr_temp;
            initial_g_out_curr++;
        } else {
            *g_curr = g_curr_temp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    dst[l_id] = f_prev;
}