#ifndef D4235AB8_4213_4E1C_B3B4_A27F113F1AE6
#define D4235AB8_4213_4E1C_B3B4_A27F113F1AE6
#include <arm_neon.h>

/**
 * @brief equivalent of SSSE3's _mm_shuffle_epi8
 * 
 * @param src is the source vreg
 * @param idx is the index vreg(requires pre-processing)
 * @return int8x16_t
 */
int8x16_t vshufq_s8(int8x16_t src, int8x16_t idx);

/**
 * @brief equivalent of SSE2's _mm_packus_epi16
 * 
 * @param lo is the vreg that converts to the lower 64 bits of the result
 * @param hi is the vreg that converts to the higher 64 bits of the result
 * @return uint8x16_t 
 */
uint8x16_t vpackusq_s16(int16x8_t lo, int16x8_t hi);

/**
 * @brief equivalent of SSE2's _mm_packs_epi32
 * 
 * @param lo is the vreg that converts to the lower 64 bits of the result
 * @param hi is the vreg that converts to the higher 64 bits of the result
 * @return int16x8_t 
 */
int16x8_t vpacksq_s32(int32x4_t lo, int32x4_t hi);

/**
 * @brief equivalent of SSE4's _mm_cvtepu8_epi16
 * 
 * @param a is the src vreg, only lower 8 uint8_t are used and widen
 * @return int16x8_t 
 */
uint16x8_t vcvtepq_u8(uint8x16_t a);

/**
 * @brief equivalent of SSSE3's _mm_maddubs_epi16
 * 
 * @param a is unsigned
 * @param b is signed
 * @return int16x8_t 
 */
int16x8_t vmaddq_i8(uint8x16_t a, int8x16_t b);

/**
 * @brief equivalent of AVX2's _mm256_madd_epi16
 * 
 * @param a is signed
 * @param b is signed
 * @return int32x4_t
 */
int32x4_t vmaddq_s16(int16x8_t a, int16x8_t b);

#endif /* D4235AB8_4213_4E1C_B3B4_A27F113F1AE6 */
