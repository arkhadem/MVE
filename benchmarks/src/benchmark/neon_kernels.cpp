#include "benchmark.hpp"

#include "init.hpp"
#include "neon_kernels.hpp"
#include "runner.hpp"
#include <cstddef>
#include <cstdio>

std::map<std::string, std::map<std::string, kernelfunc>> kernel_functions;

void register_kernels() {
    kernel_functions["libjpeg"]["rgb_to_gray"] = rgb_to_gray_neon;
    kernel_functions["libjpeg"]["ycbcr_to_rgb"] = ycbcr_to_rgb_neon;
    kernel_functions["libjpeg"]["upsample"] = upsample_neon;
    kernel_functions["libjpeg"]["downsample"] = downsample_neon;
    kernel_functions["libjpeg"]["huffman_encode"] = huffman_encode_neon;

    kernel_functions["libpng"]["read_sub"] = read_sub_neon;
    kernel_functions["libpng"]["read_up"] = read_up_neon;
    kernel_functions["libpng"]["expand_palette"] = expand_palette_neon;

    kernel_functions["libwebp"]["sharp_update_rgb"] = sharp_update_rgb_neon;
    kernel_functions["libwebp"]["sharp_filter_row"] = sharp_filter_row_neon;
    kernel_functions["libwebp"]["apply_alpha_multiply"] = apply_alpha_multiply_neon;
    kernel_functions["libwebp"]["dispatch_alpha"] = dispatch_alpha_neon;
    kernel_functions["libwebp"]["tm_prediction"] = tm_prediction_neon;
    kernel_functions["libwebp"]["vertical_filter"] = vertical_filter_neon;
    kernel_functions["libwebp"]["gradient_filter"] = gradient_filter_neon;

    kernel_functions["boringssl"]["aes"] = aes_neon;
    kernel_functions["boringssl"]["des"] = des_neon;
    kernel_functions["boringssl"]["chacha20"] = chacha20_neon;

    kernel_functions["zlib"]["adler32"] = adler32_neon;
    kernel_functions["zlib"]["crc32"] = crc32_neon;

    kernel_functions["skia"]["convolve_horizontally"] = convolve_horizontally_neon;
    kernel_functions["skia"]["convolve_vertically"] = convolve_vertically_neon;
    kernel_functions["skia"]["row_blend"] = row_blend_neon;
    kernel_functions["skia"]["row_opaque"] = row_opaque_neon;

    kernel_functions["webaudio"]["is_audible"] = is_audible_neon;
    kernel_functions["webaudio"]["copy_with_gain"] = copy_with_gain_neon;
    kernel_functions["webaudio"]["copy_with_sample"] = copy_with_sample_neon;
    kernel_functions["webaudio"]["sum_from"] = sum_from_neon;
    kernel_functions["webaudio"]["handle_nan"] = handle_nan_neon;

    kernel_functions["optroutines"]["memchr"] = memchr_neon;
    kernel_functions["optroutines"]["memcmp"] = memcmp_neon;
    kernel_functions["optroutines"]["memset"] = memset_neon;
    kernel_functions["optroutines"]["strlen"] = strlen_neon;

    kernel_functions["cmsisdsp"]["fir"] = fir_neon;
    kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_neon;
    kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_neon;

    kernel_functions["kvazaar"]["dct"] = dct_neon;
    kernel_functions["kvazaar"]["idct"] = idct_neon;
    kernel_functions["kvazaar"]["intra"] = intra_neon;
    kernel_functions["kvazaar"]["satd"] = satd_neon;

    kernel_functions["linpack"]["lpack"] = lpack_neon;
}
