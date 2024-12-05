__kernel void lpack_adreno_kernel(
    int da,
    __global int *dx,
    __global int *dyin,
    __global int *dyout) {
    int id = get_global_id(0);
    dyout[id] += dyin[id] + da * dx[id];
}