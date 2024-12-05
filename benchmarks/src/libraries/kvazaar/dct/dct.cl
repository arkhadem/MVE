
#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)  ((X) < (Y) ? (Y) : (X))

void partial_butterfly_adreno(
    int l0_id,
    __local short *smem,
    __constant short *coeff,
    char shift) {
    
    int e_0, e_1, e_2, e_3;
    int o_0, o_1, o_2, o_3;
    int ee_0, ee_1;
    int eo_0, eo_1;
    int add = 1 << (shift - 1);

    // E and O
    __local short *my_src = smem + l0_id * 10;

    e_0 = my_src[0] + my_src[7];
    o_0 = my_src[0] - my_src[7];

    e_1 = my_src[1] + my_src[6];
    o_1 = my_src[1] - my_src[6];

    e_2 = my_src[2] + my_src[5];
    o_2 = my_src[2] - my_src[5];

    e_3 = my_src[3] + my_src[4];
    o_3 = my_src[3] - my_src[4];

    // EE and EO
    ee_0 = e_0 + e_3;
    eo_0 = e_0 - e_3;
    ee_1 = e_1 + e_2;
    eo_1 = e_1 - e_2;

    barrier(CLK_LOCAL_MEM_FENCE);

    __local short *my_dst = smem + l0_id;

    my_dst[0] = (short)((coeff[0] * ee_0 + coeff[1] * ee_1 + add) >> shift);
    my_dst[40] = (short)((coeff[32] * ee_0 + coeff[33] * ee_1 + add) >> shift);
    my_dst[20] = (short)((coeff[16] * eo_0 + coeff[17] * eo_1 + add) >> shift);
    my_dst[60] = (short)((coeff[48] * eo_0 + coeff[49] * eo_1 + add) >> shift);

    my_dst[10] = (short)((coeff[8] * o_0 + coeff[9] * o_1 + coeff[10] * o_2 + coeff[11] * o_3 + add) >> shift);
    my_dst[30] = (short)((coeff[24] * o_0 + coeff[25] * o_1 + coeff[26] * o_2 + coeff[27] * o_3 + add) >> shift);
    my_dst[50] = (short)((coeff[40] * o_0 + coeff[41] * o_1 + coeff[42] * o_2 + coeff[43] * o_3 + add) >> shift);
    my_dst[70] = (short)((coeff[56] * o_0 + coeff[57] * o_1 + coeff[58] * o_2 + coeff[59] * o_3 + add) >> shift);
    
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void dct_adreno_kernel(
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
    partial_butterfly_adreno(l0_id, my_smem, coeff, 6);

    my_smem = smem + l1_id * 80;
    partial_butterfly_adreno(l0_id, my_smem, coeff, 9);
    
    // storing outputs coalescly to smem
    __global short* my_output = output + block_global_id_start * 64 + l1_id * 8 + l0_id;
    my_smem = smem + l1_id * 10 + l0_id;
    for (int curr_global_id = 0; (curr_global_id < 8) && (curr_global_id + block_global_id_start < count); curr_global_id += 1) {
        *my_output = *my_smem;
        my_output += 64;
        my_smem += 80;
    }
}