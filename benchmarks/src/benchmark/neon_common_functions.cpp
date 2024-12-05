#include "neon_common_functions.hpp"
#include <arm_neon.h>
#include <stdint.h>

int8x16_t vshufq_s8(int8x16_t src, int8x16_t idx) {
    // load mask into vreg
    int8x16_t mask = vdupq_n_s8(0x70);
    src = vbicq_s8(src, mask); // conduct a bit-wise clear
    return vqtbl1q_s8(src, idx);
}

uint8x16_t vpackusq_s16(int16x8_t lo, int16x8_t hi) {
    return vcombine_u8(vmovn_s16(lo), vmovn_s16(hi));
}

int16x8_t vpacksq_s32(int32x4_t lo, int32x4_t hi) {
    return vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
}

// NOTE: v-convert-(zero)extended-primary(aka. "lower")-q
uint16x8_t vcvtepq_u8(uint8x16_t a) { return vmovl_u8(vget_low_u8(a)); }

/**
 * NOTE: semantics of original SSSE instruction:
 * Name `maddubs` is the abbr. for "Multiply and Add Packed Signed and Unsigned Bytes"
 * *********
 * Multiplies corresponding pairs of packed 8-bit unsigned integer
 *    values contained in the first source operand and packed 8-bit signed
 *    integer values contained in the second source operand, adds pairs of
 *    contiguous products with signed saturation, and writes the 16-bit sums to
 *    the corresponding bits in the destination.
 */
int16x8_t vmaddq_i8(uint8x16_t a, int8x16_t b) {
    int8x16_t signed_a = vreinterpretq_s8_u8(a);
    int16x8_t prod_lo = vmull_s8(vget_low_s8(signed_a), vget_low_s8(b));
    int16x8_t prod_hi = vmull_high_s8(signed_a, b);
    return vqaddq_s16(prod_lo, prod_hi);

    // int16x8_t a_odd = vreinterpretq_s16_u16(vshrq_n_u16(a, 8)); // {0, 1} ->
    // int16x8_t b_odd = vshrq_n_s16(b, 8);
    // int16x8_t a_even = vreinterpretq_s16_u16(vbicq_u16(a, vdupq_n_u16(0xff00)));
    // /* NOTE: b is a signed vector, do signed extension */
    // int16x8_t b_even = vshrq_n_s16(vshlq_n_s16(b, 8), 8);
    // int16x8_t prod_odd = vmulq_s16(a_odd, b_odd);
    // int16x8_t prod_even = vmulq_s16(a_even, b_even);
    // return vqaddq_s16(prod_even, prod_odd);
}

int32x4_t vmaddq_s16(int16x8_t a, int16x8_t b) {
    int32x4_t a_b_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    int32x4_t a_b_high = vmull_high_s16(a, b);
    return vqaddq_s32(a_b_low, a_b_high);
}