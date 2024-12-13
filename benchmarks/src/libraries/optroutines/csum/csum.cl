#ifndef BLOCK_16K
#define BLOCK_16K 16384
#endif
#ifndef BLOCK_8K
#define BLOCK_8K 8192
#endif
#ifndef BLOCK_4K
#define BLOCK_4K 4096
#endif
#ifndef BLOCK_2K
#define BLOCK_2K 2048
#endif
#ifndef BLOCK_1K
#define BLOCK_1K 1024
#endif
#ifndef BLOCK_512
#define BLOCK_512 512
#endif
#ifndef BLOCK_256
#define BLOCK_256 256
#endif
#ifndef BLOCK_128
#define BLOCK_128 128
#endif
#ifndef BLOCK_64
#define BLOCK_64 64
#endif
#ifndef BLOCK_32
#define BLOCK_32 32
#endif
#ifndef BLOCK_16
#define BLOCK_16 16
#endif
#ifndef BLOCK_8
#define BLOCK_8 8
#endif
#ifndef BLOCK_4
#define BLOCK_4 4
#endif
#ifndef BLOCK_2
#define BLOCK_2 2
#endif
#ifndef BLOCK_1
#define BLOCK_1 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE BLOCK_1K
#endif

typedef short int16_t;
typedef int int32_t;
typedef long int64_t;

__kernel void csum_adreno_kernel(
    __global int32_t *ptr,
    __global int16_t *partial_sum) {

    __local int64_t state_l[BLOCK_SIZE];

    int l_id = get_local_id(0);
    int g_id = get_global_id(1);

    __global int32_t *my_ptr = ptr + g_id * BLOCK_16K + l_id;
    __local int64_t *my_state_l = state_l + l_id;

    int64_t l_checksum = 0;
    for (int load = 0; load < BLOCK_16K; load += BLOCK_SIZE) {
        l_checksum += *my_ptr;
        my_ptr += BLOCK_SIZE;
    }

    *my_state_l = l_checksum;
    barrier(CLK_LOCAL_MEM_FENCE);

#if BLOCK_SIZE == BLOCK_16K
    if (l_id < BLOCK_8K)
        *my_state_l += state_l[l_id + BLOCK_8K];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_8K
    if (l_id < BLOCK_4K)
        *my_state_l += state_l[l_id + BLOCK_4K];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_4K
    if (l_id < BLOCK_2K)
        *my_state_l += state_l[l_id + BLOCK_2K];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_2K
    if (l_id < BLOCK_1K)
        *my_state_l += state_l[l_id + BLOCK_1K];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_1K
    if (l_id < BLOCK_512)
        *my_state_l += state_l[l_id + BLOCK_512];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_512
    if (l_id < BLOCK_256)
        *my_state_l += state_l[l_id + BLOCK_256];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_256
    if (l_id < BLOCK_128)
        *my_state_l += state_l[l_id + BLOCK_128];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_128
    if (l_id < BLOCK_64)
        *my_state_l += state_l[l_id + BLOCK_64];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_64
    if (l_id < BLOCK_32)
        *my_state_l += state_l[l_id + BLOCK_32];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_32
    if (l_id < BLOCK_16)
        *my_state_l += state_l[l_id + BLOCK_16];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_16
    if (l_id < BLOCK_8)
        *my_state_l += state_l[l_id + BLOCK_8];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_8
    if (l_id < BLOCK_4)
        *my_state_l += state_l[l_id + BLOCK_4];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_4
    if (l_id < BLOCK_2)
        *my_state_l += state_l[l_id + BLOCK_2];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if BLOCK_SIZE >= BLOCK_2
    if (l_id < BLOCK_1)
        *my_state_l += state_l[l_id + BLOCK_1];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    if (l_id == 0) {
        int64_t sum_tmp = state_l[0];
        sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);
        sum_tmp = (sum_tmp & 0xffffffff) + (sum_tmp >> 32);

        /* Fold 32-bit sum_tmp to 16 bits */
        sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);
        sum_tmp = (sum_tmp & 0xffff) + (sum_tmp >> 16);

        partial_sum[g_id] = sum_tmp;
    }
}