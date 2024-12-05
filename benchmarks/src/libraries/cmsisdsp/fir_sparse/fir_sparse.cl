__kernel void fir_sparse_adreno_kernel(
    int sample_count,
    int coeff_count,
    __global int *src,
    __constant int *coeff,
    __constant int *delay,
    __global int *dst) {

    int g_id = get_global_id(0);

    __global int *my_src = src + g_id;
    __constant int *my_coeff = coeff;
    __constant int *my_delay = delay;

    int src_temp;
    int coeff_temp;
    int delay_temp;
    int acc;

    if (g_id < sample_count) {
        acc = 0;
        for (int coeff_idx = 0; coeff_idx < coeff_count; coeff_idx++) {
            coeff_temp = *my_coeff;
            delay_temp = *my_delay;
            src_temp = *(my_src + delay_temp);
            acc += src_temp * coeff_temp;
            my_coeff += 1;
            my_delay += 1;
        }
        dst[g_id] = acc;
    }
}