#ifndef A027C2A0_39D2_45C4_A2AA_098B8EB820A1
#define A027C2A0_39D2_45C4_A2AA_098B8EB820A1

#define png_byte unsigned char
#define png_bytep unsigned char *
#define png_uint_32 unsigned int

#define png_aligncastconst(type, value) \
    static_cast<type>(static_cast<const void *>(value))

#define png_aligncast(type, value) \
    static_cast<type>(static_cast<void *>(value))

#define png_ptrc(type, pointer) png_aligncastconst(const type *, pointer)

#define png_ptr(type, pointer) png_aligncast(type *, pointer)

#define png_ldr(type, pointer) \
    (temp_pointer = png_ptr(type, pointer), *temp_pointer)

#include "benchmark.hpp"

#endif /* A027C2A0_39D2_45C4_A2AA_098B8EB820A1 */
