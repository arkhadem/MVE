#include "benchmark.hpp"

#include "init.hpp"
#include "mve_kernels.hpp"
#include "runner.hpp"

#include "mve.hpp"

std::map<std::string, std::map<std::string, kernelfunc>> kernel_functions;

void register_kernels() {
    kernel_functions["libjpeg"]["rgb_to_gray"] = rgb_to_gray_mve;
    kernel_functions["libjpeg"]["ycbcr_to_rgb"] = ycbcr_to_rgb_mve;
    kernel_functions["libjpeg"]["upsample"] = upsample_mve;
    kernel_functions["libjpeg"]["downsample"] = downsample_mve;
    kernel_functions["libjpeg"]["huffman_encode"] = huffman_encode_mve;

    kernel_functions["libpng"]["read_sub"] = read_sub_mve;
    kernel_functions["libpng"]["read_up"] = read_up_mve;
    kernel_functions["libpng"]["expand_palette"] = expand_palette_mve;

    kernel_functions["libwebp"]["sharp_update_rgb"] = sharp_update_rgb_mve;
    kernel_functions["libwebp"]["sharp_filter_row"] = sharp_filter_row_mve;
    kernel_functions["libwebp"]["apply_alpha_multiply"] = apply_alpha_multiply_mve;
    kernel_functions["libwebp"]["dispatch_alpha"] = dispatch_alpha_mve;
    kernel_functions["libwebp"]["tm_prediction"] = tm_prediction_mve;
    kernel_functions["libwebp"]["vertical_filter"] = vertical_filter_mve;
    kernel_functions["libwebp"]["gradient_filter"] = gradient_filter_mve;

    kernel_functions["boringssl"]["aes"] = aes_mve;
    kernel_functions["boringssl"]["des"] = des_mve;
    kernel_functions["boringssl"]["chacha20"] = chacha20_mve;

    kernel_functions["zlib"]["adler32"] = adler32_mve;
    kernel_functions["zlib"]["crc32"] = crc32_mve;

    kernel_functions["skia"]["convolve_horizontally"] = convolve_horizontally_mve;
    kernel_functions["skia"]["convolve_vertically"] = convolve_vertically_mve;
    kernel_functions["skia"]["row_blend"] = row_blend_mve;
    kernel_functions["skia"]["row_opaque"] = row_opaque_mve;

    kernel_functions["webaudio"]["is_audible"] = is_audible_mve;
    kernel_functions["webaudio"]["copy_with_gain"] = copy_with_gain_mve;
    kernel_functions["webaudio"]["copy_with_sample"] = copy_with_sample_mve;
    kernel_functions["webaudio"]["sum_from"] = sum_from_mve;
    kernel_functions["webaudio"]["handle_nan"] = handle_nan_mve;

    kernel_functions["optroutines"]["memchr"] = memchr_mve;
    kernel_functions["optroutines"]["memcmp"] = memcmp_mve;
    kernel_functions["optroutines"]["memset"] = memset_mve;
    kernel_functions["optroutines"]["strlen"] = strlen_mve;

    kernel_functions["cmsisdsp"]["fir"] = fir_mve;
    kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_mve;
    kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_mve;

    kernel_functions["kvazaar"]["dct"] = dct_mve;
    kernel_functions["kvazaar"]["idct"] = idct_mve;
    kernel_functions["kvazaar"]["intra"] = intra_mve;
    kernel_functions["kvazaar"]["satd"] = satd_mve;

    kernel_functions["linpack"]["lpack"] = lpack_mve;
}
