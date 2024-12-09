
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains two optimized matrix-multiplication kernels:
// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on
//   http://www.cedricnugteren.nl/tutorial.php
// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling
//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Both are fully configurable (and tunable!) using many parameters. Both kernels support
// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
//
// For kernel 0 matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// For kernel 1, both A and C are transposed w.r.t. the above
//
// Or as an image (assuming column-major)
//       K
//    o-------o
//    |       |
//  N | [B^T] |
//    |       |
//    o-------o
//        K               N
//    o-------o        o-----o
//  M |  [A]  |      M | [C] |
//    |       |        |     |
//    o-------o        o-----o
//
//
// This kernel is separated into multiple files. This is part 1 out of 4.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
#define GEMMK 1 // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
#define MWG 8 // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
#define NWG 8 // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
#define KWG 8 // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
#define MDIMC 8 // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
#define NDIMC 8 // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
#define MDIMA 8 // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
#define NDIMB 8 // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
#define KWI 1 // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef STRM
#define STRM 0 // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
#define STRN 0 // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
#define SA 0 // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
#define SB 0 // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
#define KREG 1 // Amount of register tiling in second dimension, multiple of 1 (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG / MDIMC)                 // Work per work-item (M-dimension)
#define NWI (NWG / NDIMC)                 // Work per work-item (N-dimension)
#define KDIMA ((MDIMC * NDIMC) / (MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC * NDIMC) / (NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG / MDIMA)                 // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG / KDIMA)                 // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG / KDIMB)                 // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG / NDIMB)                 // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
#define USE_VECTOR_MAD 0 // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
#define GLOBAL_MEM_FENCE 0 // Global synchronisation barrier for potential better performance
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
#define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
#define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
#define USE_SUBGROUP_SHUFFLING 0 // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define SUBGROUP_SIZE 8 // Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
#if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
#define SUBGROUP_SIZE 32 // Assumes subgroup size is always 32 on NVIDIA GPUs
#endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
#undef USE_SUBGROUP_SHUFFLING
#define USE_SUBGROUP_SHUFFLING 0 // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
inline void GlobalToLocalA(const __global int *restrict weight, LOCAL_PTR int *alm,
                           const int kSizeM, const int tid, const int kwg) {
    const int la0 = tid % MDIMA;
    const int la1 = tid / MDIMA;
#pragma unroll
    for (int _mia = 0; _mia < MWA / 1; _mia += 1) {
#pragma unroll
        for (int _kia = 0; _kia < KWA; _kia += 1) {

// Computes the indices based on strided/non-strided access
#if STRM == 0
            int mg = _mia + la0 * (MWA / 1);
#elif STRM == 1
            int mg = la0 + _mia * MDIMA;
#endif

            // Computes the indices for the global memory
            int kg = _kia + la1 * KWA;
            int idm = mg + get_group_id(0) * (MWG / 1);
            int idk = kg + kwg;

            // Loads the data from global memory (not transposed) into the local memory
            alm[kg * (MWG / 1) + mg] = weight[idk * (kSizeM / 1) + idm];
        }
    }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void GlobalToLocalB(const __global int *restrict input, LOCAL_PTR int *blm,
                           const int kSizeN, const int tid, const int kwg) {
    const int lb0 = tid % NDIMB;
    const int lb1 = tid / NDIMB;
#pragma unroll
    for (int _kib = 0; _kib < KWB; _kib += 1) {
#pragma unroll
        for (int _nib = 0; _nib < NWB / 1; _nib += 1) {

// Computes the indices based on strided/non-strided access
#if STRN == 0
            int ng = _nib + lb0 * (NWB / 1);
#elif STRN == 1
            int ng = lb0 + _nib * NDIMB;
#endif

            // Computes the indices for the global memory
            int kg = _kib + lb1 * KWB;
            int idn = ng + get_group_id(1) * (NWG / 1);
            int idk = kg + kwg;

            // Loads the data from global memory (transposed) into the local memory
            blm[kg * (NWG / 1) + ng] = input[idk * (kSizeN / 1) + idn];
        }
    }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
inline int GlobalToPrivateA(const __global int *restrict weight, const int _mi,
                              const int kSizeM, const int idk, const int kwg) {
// Computes the indices based on strided/non-strided access
#if STRM == 0
    int mg = _mi + get_local_id(0) * (MWI / 1);
#elif STRM == 1
    int mg = get_local_id(0) + _mi * MDIMC;
#endif

    // Computes the indices for the global memory
    int idm = mg + get_group_id(0) * (MWG / 1);

    // Loads the data from global memory (not transposed) and stores into registers
    return weight[idk * (kSizeM / 1) + idm];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
inline int GlobalToPrivateB(const __global int *restrict input, const int _ni,
                              const int kSizeN, const int idk) {
// Computes the indices based on strided/non-strided access
#if STRN == 0
    int ng = _ni + get_local_id(1) * (NWI / 1);
#elif STRN == 1
    int ng = get_local_id(1) + _ni * NDIMC;
#endif

    // Computes the indices for the global memory
    int idn = ng + get_group_id(1) * (NWG / 1);

    // Loads the data from global memory (transposed) and stores into registers
    return input[idk * (kSizeN / 1) + idn];
}
#endif

// =================================================================================================
#if GEMMK == 1

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix for kernel 1.
inline int GlobalToPrivateA2D(const __global int *restrict a_ptr, const int tid_y, const int _ni,
                                const int kSizeK, const int idk, const int _ki) {
#if PRECISION == 3232 || PRECISION == 6464
    const int a_index = (tid_y * NWI + _ni) * (kSizeK / 1) + idk / 1 + _ki;
    const __global int *restrict weight = (const __global int *restrict)a_ptr;
    return weight[a_index];
#else
    const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * 1;
    return a_ptr[a_index];
#endif
}

// Same as above, but now for the B input matrix
inline int GlobalToPrivateB2D(const __global int *restrict b_ptr, const int tid_x, const int _mi,
                                const int kSizeN, const int idk, const int _ki) {
#if PRECISION == 3232 || PRECISION == 6464
    const int b_index = (idk + _ki) * (kSizeN / 1) + tid_x * (MWI / 1) + _mi;
    const __global int *restrict input = (const __global int *restrict)b_ptr;
    return input[b_index];
#else
    const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * 1;
    return b_ptr[b_index];
#endif
}

#endif
// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
inline int LocalToPrivateA(LOCAL_PTR int *alm, const int _mi, const int kg) {
#if STRM == 0
    int mg = _mi + get_local_id(0) * (MWI / 1);
#elif STRM == 1
    int mg = get_local_id(0) + _mi * MDIMC;
#endif
    return alm[kg * (MWG / 1) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline int LocalToPrivateB(LOCAL_PTR int *blm, const int _ni, const int kg) {
#if STRN == 0
    int ng = _ni + get_local_id(1) * (NWI / 1);
#elif STRN == 1
    int ng = get_local_id(1) + _ni * NDIMC;
#endif
    return blm[kg * (NWG / 1) + ng];
}
#endif

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// The vectorised multiply-add function
inline int MultiplyAddVector(int cvec, const int avec, const int bval) {
#if USE_VECTOR_MAD == 1
    cvec += avec * bval;
#else
    cvec += (avec * bval);
#endif
    return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResults(__global int *output,
                         int c_value,
                         const __global int *restrict bias,
                         const int _mi,
                         const int _ni,
                         const int kSizeM,
                         const int min,
                         const int max) {
#if STRM == 0
    int mg = _mi + get_local_id(0) * (MWI / 1);
#elif STRM == 1
    int mg = get_local_id(0) + _mi * MDIMC;
#endif
#if STRN == 0
    int ng = _ni + get_local_id(1) * NWI;
#elif STRN == 1
    int ng = _ni % 1 + get_local_id(1) * 1 + (_ni / 1) * 1 * NDIMC;
#endif
    int idm = mg + get_group_id(0) * (MWG / 1);
    int idn = ng + get_group_id(1) * NWG;
    int index = idn * (kSizeM / 1) + idm;

    c_value += bias[idn];

    if (c_value > max)
        c_value = max;
    if (c_value < min)
        c_value = min;

    output[index] = c_value;
}

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// A common interface for subgroup functions

#if USE_SUBGROUP_SHUFFLING == 1

inline int clblast_get_sub_group_local_id() {

// Intel extension
#if SUBGROUP_SHUFFLING_INTEL == 1
    return get_sub_group_local_id();

// Nvidia inline PTX
#elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    int ret;
    asm volatile("mov.u32 %0, %%laneid;"
                 : "=r"(ret));
    return ret;
#endif
}

inline int clblast_sub_group_shuffle(int reg, int src) {

// Intel extension
#if SUBGROUP_SHUFFLING_INTEL == 1
    return intel_sub_group_shuffle(reg, src);

// Nvidia inline PTX
// Volta and later requires .sync shuffle instructions with an extra mask arg
#elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    int ret;
#if SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(ret)
                 : "f"(reg), "r"(src));
#else
    asm volatile("shfl.idx.b32 %0, %1, %2, 0x1f;"
                 : "=f"(ret)
                 : "f"(reg), "r"(src));
#endif
    return ret;
#endif
}
#endif

// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
inline void XgemmBody(const int kSizeM,
                      const int kSizeN,
                      const int kSizeK,
                      const __global int *restrict weight,
                      const __global int *restrict input,
                      const __global int *restrict bias,
                      __global int *output,
                      int min,
                      int max

#if SA == 1 && SB == 1
                      ,
                      LOCAL_PTR int *alm, LOCAL_PTR int *blm
#elif SA == 1
                      ,
                      LOCAL_PTR int *alm
#elif SB == 1
                      ,
                      LOCAL_PTR int *blm
#endif
) {

// Allocates workitem-private memory (registers)
#if GEMMK == 0
#pragma promote_to_registers
    int apm[MWI / 1]; // MWI * 1
#pragma promote_to_registers
    int bpm[NWI / 1]; // 1 * NWI
#elif GEMMK == 1
#if USE_SUBGROUP_SHUFFLING == 1
#pragma promote_to_registers
    int apm[KREG / 1];         // KREG (subgroup shuffling in NWI dimension)
#else
#pragma promote_to_registers
    int apm[NWI * (KREG / 1)]; // NWI * KREG
#endif
#pragma promote_to_registers
    int bpm[KREG * (MWI / 1)]; // KREG * MWI
#endif
#pragma promote_to_registers
    int cpm[NWI * (MWI / 1)]; // NWI * MWI

#if GEMMK == 1
    const __global int *restrict a_ptr = (const __global int *restrict)&weight[0];
    const __global int *restrict b_ptr = (const __global int *restrict)&input[0];
    const int tid_x = get_local_id(0) + MDIMC * get_group_id(0);
    const int tid_y = get_local_id(1) + NDIMC * get_group_id(1);
#endif

// Combined thread identifier (volatile to disable caching)
#if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC * get_local_id(1);
#endif

// Initializes the accumulation registers
#pragma unroll
    for (int _mi = 0; _mi < MWI / 1; _mi += 1) {
#pragma unroll
        for (int _ni = 0; _ni < NWI; _ni += 1) {
            cpm[_ni * (MWI / 1) + _mi] = 0;
        }
    }

    // Loops over all workgroup tiles
    for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

// Loads data: off-chip --> local (matrix A)
#if SA == 1
        GlobalToLocalA(weight, alm, kSizeM, tid, kwg);
#endif
// Loads data: off-chip --> local (matrix B)
#if SB == 1
        GlobalToLocalB(input, blm, kSizeN, tid, kwg);
#endif
#if SA == 1 || SB == 1
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        // Loops over all workitem tiles, unrolled by a factor KWI
        for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
#pragma unroll
            for (int _pit = 0; _pit < KWI * KREG; _pit += KREG) {
#if SA == 0 || SB == 0
                int idk = kwg + pwi + _pit;
#endif
#if SA == 1 || SB == 1
                int kg = pwi + _pit;
#endif

// Loads matrix A (kernel 0) or matrix B (kernel 1)
#pragma unroll
                for (int _mi = 0; _mi < MWI / 1; _mi += 1) {
// Loads data: local --> private (matrix A)
#if GEMMK == 0 && SA == 1
                    apm[_mi] = LocalToPrivateA(alm, _mi, kg);
// Loads data: off-chip --> private (matrix A)
#elif GEMMK == 0 && SA == 0
                    apm[_mi] = GlobalToPrivateA(weight, _mi, kSizeM, idk, kwg);
// Loads data: 2D global --> 2D private (matrix B)
#elif GEMMK == 1
#pragma unroll
                    for (int _ki = 0; _ki < KREG; _ki += 1) {
                        bpm[_ki * (MWI / 1) + _mi] = GlobalToPrivateB2D(b_ptr, tid_x, _mi, kSizeN, idk, _ki);
                    }
#endif
                }

// Loads matrix B (kernel 0) or matrix A (kernel 1)
#if GEMMK == 0
#pragma unroll
                for (int _ni = 0; _ni < NWI / 1; _ni += 1) {
// Loads data: local --> private (matrix B)
#if SB == 1
                    bpm[_ni] = LocalToPrivateB(blm, _ni, kg);
// Loads data: off-chip --> private (matrix B)
#else
                    bpm[_ni] = GlobalToPrivateB(input, _ni, kSizeN, idk);
#endif
                }
#elif GEMMK == 1
// Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups
#if USE_SUBGROUP_SHUFFLING == 1
                const int _ni = clblast_get_sub_group_local_id();
#pragma unroll
                for (int _ki = 0; _ki < KREG / 1; _ki += 1) {
                    apm[_ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);
                }
// Loads data: 2D global --> 2D private (matrix A)
#else
#pragma unroll
                for (int _ni = 0; _ni < NWI; _ni += 1) {
#pragma unroll
                    for (int _ki = 0; _ki < KREG / 1; _ki += 1) {
                        apm[_ni * (KREG / 1) + _ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);
                    }
                }
#endif
#endif

// Performs the accumulation (Cpm += Apm * Bpm)
#if GEMMK == 0
#pragma unroll
                for (int _ni = 0; _ni < NWI / 1; _ni += 1) {
#pragma unroll
                    for (int _mi = 0; _mi < MWI / 1; _mi += 1) {
                        const int aval = apm[_mi];
                        cpm[(_ni * 1 + 0) * (MWI / 1) + _mi] = MultiplyAddVector(cpm[(_ni * 1 + 0) * (MWI / 1) + _mi], aval, bpm[_ni]);
                    }
                }
#elif GEMMK == 1
#pragma unroll
                for (int _ni = 0; _ni < NWI; _ni += 1) {
#pragma unroll
                    for (int _mi = 0; _mi < MWI / 1; _mi += 1) {
#pragma unroll
                        for (int _ki = 0; _ki < KREG / 1; _ki += 1) {
#if USE_SUBGROUP_SHUFFLING == 1
                            const int aval = clblast_sub_group_shuffle(apm[_ki], _ni);
#else
                            const int aval = apm[_ni * (KREG / 1) + _ki];
#endif
                            cpm[_ni * (MWI / 1) + _mi] = MultiplyAddVector(cpm[_ni * (MWI / 1) + _mi], bpm[(1 * _ki + 0) * (MWI / 1) + _mi], aval);
                        }
                    }
                }
#endif
            }
        }
#if SA == 1 || SB == 1
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }
#if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
#endif

// Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
#if GEMMK == 0
    const int cld = kSizeM;
#elif GEMMK == 1
    const int cld = kSizeN;
#endif
#pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
#pragma unroll
        for (int _mi = 0; _mi < MWI / 1; _mi += 1) {
            StoreResults(output, cpm[_ni * (MWI / 1) + _mi], bias, _mi, _ni, cld, min, max);
        }
    }
}

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// The upper-triangular and lower-triangular kernels are only used in special cases
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)

// Main entry point of the kernel. This is the upper-triangular version.
#if RELAX_WORKGROUP_SIZE == 1
__kernel
#else
__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
    void
    XgemmUpper(const int kSizeN, const int kSizeK,
               const int arg_alpha,
               const int arg_beta,
               const __global int *restrict weight,
               const __global int *restrict input,
               __global int *output) {
    const int alpha = GetRealArg(arg_alpha);
    const int beta = GetRealArg(arg_beta);

    // Skip these threads if they do not contain threads contributing to the upper-triangle
    if ((get_group_id(1) + 1) * NWG < get_group_id(0) * MWG) {
        return;
    }

// Allocates workgroup-private memory (local memory)
#if SA == 1
    __local int alm[KWG * MWG / 1];
#endif
#if SB == 1
    __local int blm[KWG * NWG / 1];
#endif

// Computes the matrix-multiplication and stores the result in global memory
#if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, alm, blm);
#elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, alm);
#elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, blm);
#else
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta);
#endif
}

// Main entry point of the kernel. This is the lower-triangular version.
#if RELAX_WORKGROUP_SIZE == 1
__kernel
#else
__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
    void
    XgemmLower(const int kSizeN, const int kSizeK,
               const int arg_alpha,
               const int arg_beta,
               const __global int *restrict weight,
               const __global int *restrict input,
               __global int *output) {
    const int alpha = GetRealArg(arg_alpha);
    const int beta = GetRealArg(arg_beta);

    // Skip these threads if they do not contain threads contributing to the lower-triangle
    if (get_group_id(1) * NWG > (get_group_id(0) + 1) * MWG) {
        return;
    }

// Allocates workgroup-private memory (local memory)
#if SA == 1
    __local int alm[KWG * MWG / 1];
#endif
#if SB == 1
    __local int blm[KWG * NWG / 1];
#endif

// Computes the matrix-multiplication and stores the result in global memory
#if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, alm, blm);
#elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, alm);
#elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta, blm);
#else
    XgemmBody(kSizeN, kSizeN, kSizeK, weight, input, output, alpha, beta);
#endif
}

// =================================================================================================
// If not using a triangular version, include the regular kernel
#else

// Main entry point of the kernel. This is the regular full version.
__kernel void Xgemm(const int M,
                    const int N,
                    const int K,
                    const __global int *restrict input,
                    const __global int *restrict bias,
                    const __global int *restrict weight,
                    __global int *output,
                    const int min,
                    const int max) {

    // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)

// Allocates workgroup-private memory (local memory)
#if SA == 1
    __local int alm[KWG * MWG / 1];
#endif
#if SB == 1
    __local int blm[KWG * NWG / 1];
#endif

// Computes the matrix-multiplication and stores the result in global memory
#if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, weight, input, output, alpha, beta, alm, blm);
#elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, weight, input, output, alpha, beta, alm);
#elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, weight, input, output, alpha, beta, blm);
#else
    XgemmBody(M, N, K, weight, input, bias, output, min, max);
#endif
}

#endif

// End of the C++11 raw string literal

// =================================================================================================
