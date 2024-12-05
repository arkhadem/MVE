#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "init.hpp"

std::map<std::string, std::map<std::string, initfunc>> init_functions;

void register_inits() {
    init_functions["libjpeg"]["rgb_to_gray"] = rgb_to_gray_init;
    init_functions["libjpeg"]["ycbcr_to_rgb"] = ycbcr_to_rgb_init;
    init_functions["libjpeg"]["upsample"] = upsample_init;
    init_functions["libjpeg"]["downsample"] = downsample_init;
    init_functions["libjpeg"]["huffman_encode"] = huffman_encode_init;

    init_functions["libpng"]["read_sub"] = read_sub_init;
    init_functions["libpng"]["read_up"] = read_up_init;
    init_functions["libpng"]["expand_palette"] = expand_palette_init;

    init_functions["libwebp"]["sharp_update_rgb"] = sharp_update_rgb_init;
    init_functions["libwebp"]["sharp_filter_row"] = sharp_filter_row_init;
    init_functions["libwebp"]["apply_alpha_multiply"] = apply_alpha_multiply_init;
    init_functions["libwebp"]["dispatch_alpha"] = dispatch_alpha_init;
    init_functions["libwebp"]["tm_prediction"] = tm_prediction_init;
    init_functions["libwebp"]["vertical_filter"] = vertical_filter_init;
    init_functions["libwebp"]["gradient_filter"] = gradient_filter_init;

    init_functions["boringssl"]["aes"] = aes_init;
    init_functions["boringssl"]["des"] = des_init;
    init_functions["boringssl"]["chacha20"] = chacha20_init;

    init_functions["zlib"]["adler32"] = adler32_init;
    init_functions["zlib"]["crc32"] = crc32_init;

    init_functions["skia"]["convolve_horizontally"] = convolve_horizontally_init;
    init_functions["skia"]["convolve_vertically"] = convolve_vertically_init;
    init_functions["skia"]["row_blend"] = row_blend_init;
    init_functions["skia"]["row_opaque"] = row_opaque_init;

    init_functions["webaudio"]["is_audible"] = is_audible_init;
    init_functions["webaudio"]["copy_with_gain"] = copy_with_gain_init;
    init_functions["webaudio"]["copy_with_sample"] = copy_with_sample_init;
    init_functions["webaudio"]["sum_from"] = sum_from_init;
    init_functions["webaudio"]["handle_nan"] = handle_nan_init;

    init_functions["optroutines"]["memchr"] = memchr_init;
    init_functions["optroutines"]["memcmp"] = memcmp_init;
    init_functions["optroutines"]["memset"] = memset_init;
    init_functions["optroutines"]["strlen"] = strlen_init;

    init_functions["cmsis"]["fir"] = fir_init;
    init_functions["cmsis"]["fir_lattice"] = fir_lattice_init;
    init_functions["cmsis"]["fir_sparse"] = fir_sparse_init;

    init_functions["kvazaar"]["dct"] = dct_init;
    init_functions["kvazaar"]["idct"] = idct_init;
    init_functions["kvazaar"]["satd"] = satd_init;
    init_functions["kvazaar"]["intra"] = intra_init;

    init_functions["linpack"]["lpack"] = lpack_init;
}