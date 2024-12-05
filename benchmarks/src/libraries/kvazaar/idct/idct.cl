
#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)  ((X) < (Y) ? (Y) : (X))

void partial_butterfly_inverse_adreno(
    int l0_id,
    __local short *smem,
    __constant short *coeff,
    char shift) {
    
    int e_0, e_1, e_2, e_3;
    int o_0, o_1, o_2, o_3;
    int ee_0, ee_1;
    int eo_0, eo_1;
    int add = 1 << (shift - 1);

    __local short *my_src = smem + l0_id;

    o_0 = coeff[8] * my_src[10] + coeff[24] * my_src[30] + coeff[40] * my_src[50] + coeff[56] * my_src[70];
    o_1 = coeff[9] * my_src[10] + coeff[25] * my_src[30] + coeff[41] * my_src[50] + coeff[57] * my_src[70];
    o_2 = coeff[10] * my_src[10] + coeff[26] * my_src[30] + coeff[42] * my_src[50] + coeff[58] * my_src[70];
    o_3 = coeff[11] * my_src[10] + coeff[27] * my_src[30] + coeff[43] * my_src[50] + coeff[59] * my_src[70];

    eo_0 = coeff[16] * my_src[20] + coeff[48] * my_src[60];
    eo_1 = coeff[17] * my_src[20] + coeff[49] * my_src[60];
    ee_0 = coeff[0] * my_src[0] + coeff[32] * my_src[40];
    ee_1 = coeff[1] * my_src[0] + coeff[33] * my_src[40];

    e_0 = ee_0 + eo_0;
    e_3 = ee_0 - eo_0;
    e_1 = ee_1 + eo_1;
    e_2 = ee_1 - eo_1;

    barrier(CLK_LOCAL_MEM_FENCE);

    __local short *my_dst = smem + l0_id * 10;

    my_dst[0] = (short)MAX(-32768, MIN(32767, (e_0 + o_0 + add) >> shift));
    my_dst[4] = (short)MAX(-32768, MIN(32767, (e_3 - o_3 + add) >> shift));

    my_dst[1] = (short)MAX(-32768, MIN(32767, (e_1 + o_1 + add) >> shift));
    my_dst[5] = (short)MAX(-32768, MIN(32767, (e_2 - o_2 + add) >> shift));

    my_dst[2] = (short)MAX(-32768, MIN(32767, (e_2 + o_2 + add) >> shift));
    my_dst[6] = (short)MAX(-32768, MIN(32767, (e_1 - o_1 + add) >> shift));

    my_dst[3] = (short)MAX(-32768, MIN(32767, (e_3 + o_3 + add) >> shift));
    my_dst[7] = (short)MAX(-32768, MIN(32767, (e_0 - o_0 + add) >> shift));

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void idct_adreno_kernel(
    int count,
    __global char *bitdepth,
    __constant short *coeff,
    __global const short *input,
    __global short *output) {

    // DIM2: input = 8
    // DIM1: row = 8
    // DIM1: col = 10
    __local short smem[640];

    int l0_id = get_local_id(0);
    int l1_id = get_local_id(1);
    int l2_id = get_global_id(2);

    int block_global_id_start = l2_id * 8;          // 8 inputs is for this thread block
    int global_id = block_global_id_start + l1_id;  // ID of the input for the current thread

    __global const short* my_input = input + block_global_id_start * 64 + l1_id * 8 + l0_id;
    __local short* my_smem = smem + l1_id * 10 + l0_id;
    for (int curr_global_id = 0; (curr_global_id < 8) && (curr_global_id + block_global_id_start < count); curr_global_id += 1) {
        *my_smem = *my_input;
        my_input += 64;
        my_smem += 80;
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    my_smem = smem + l1_id * 80;
    partial_butterfly_inverse_adreno(l0_id, my_smem, coeff, 7);

    my_smem = smem + l1_id * 80;
    partial_butterfly_inverse_adreno(l0_id, my_smem, coeff, 8);
    
    // storing outputs coalescly to smem
    __global short* my_output = output + block_global_id_start * 64 + l1_id * 8 + l0_id;
    my_smem = smem + l1_id * 10 + l0_id;
    for (int curr_global_id = 0; (curr_global_id < 8) && (curr_global_id + block_global_id_start < count); curr_global_id += 1) {
        *my_output = *my_smem;
        my_output += 64;
        my_smem += 80;
    }
}
