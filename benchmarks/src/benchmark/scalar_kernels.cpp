#include "benchmark.hpp"

#include "init.hpp"
#include "runner.hpp"
#include "scalar_kernels.hpp"
#include <cstddef>
#include <cstdio>

std::map<std::string, std::map<std::string, kernelfunc>> kernel_functions;

void register_kernels() {
    kernel_functions["libjpeg"]["rgb_to_gray"] = rgb_to_gray_scalar;
    kernel_functions["libjpeg"]["ycbcr_to_rgb"] = ycbcr_to_rgb_scalar;
    kernel_functions["libjpeg"]["upsample"] = upsample_scalar;
    kernel_functions["libjpeg"]["downsample"] = downsample_scalar;
    kernel_functions["libjpeg"]["huffman_encode"] = huffman_encode_scalar;

    kernel_functions["libpng"]["read_sub"] = read_sub_scalar;
    kernel_functions["libpng"]["read_up"] = read_up_scalar;
    kernel_functions["libpng"]["expand_palette"] = expand_palette_scalar;

    kernel_functions["libwebp"]["sharp_update_rgb"] = sharp_update_rgb_scalar;
    kernel_functions["libwebp"]["sharp_filter_row"] = sharp_filter_row_scalar;
    kernel_functions["libwebp"]["apply_alpha_multiply"] = apply_alpha_multiply_scalar;
    kernel_functions["libwebp"]["dispatch_alpha"] = dispatch_alpha_scalar;
    kernel_functions["libwebp"]["tm_prediction"] = tm_prediction_scalar;
    kernel_functions["libwebp"]["vertical_filter"] = vertical_filter_scalar;
    kernel_functions["libwebp"]["gradient_filter"] = gradient_filter_scalar;

    kernel_functions["boringssl"]["aes"] = aes_scalar;
    kernel_functions["boringssl"]["des"] = des_scalar;
    kernel_functions["boringssl"]["chacha20"] = chacha20_scalar;

    kernel_functions["zlib"]["adler32"] = adler32_scalar;
    kernel_functions["zlib"]["crc32"] = crc32_scalar;

    kernel_functions["skia"]["convolve_horizontally"] = convolve_horizontally_scalar;
    kernel_functions["skia"]["convolve_vertically"] = convolve_vertically_scalar;
    kernel_functions["skia"]["row_blend"] = row_blend_scalar;
    kernel_functions["skia"]["row_opaque"] = row_opaque_scalar;

    kernel_functions["webaudio"]["is_audible"] = is_audible_scalar;
    kernel_functions["webaudio"]["copy_with_gain"] = copy_with_gain_scalar;
    kernel_functions["webaudio"]["copy_with_sample"] = copy_with_sample_scalar;
    kernel_functions["webaudio"]["sum_from"] = sum_from_scalar;
    kernel_functions["webaudio"]["handle_nan"] = handle_nan_scalar;

    kernel_functions["optroutines"]["memchr"] = memchr_scalar;
    kernel_functions["optroutines"]["memcmp"] = memcmp_scalar;
    kernel_functions["optroutines"]["memset"] = memset_scalar;
    kernel_functions["optroutines"]["strlen"] = strlen_scalar;
    kernel_functions["optroutines"]["csum"] = csum_scalar;

    kernel_functions["cmsisdsp"]["fir"] = fir_scalar;
    kernel_functions["cmsisdsp"]["fir_lattice"] = fir_lattice_scalar;
    kernel_functions["cmsisdsp"]["fir_sparse"] = fir_sparse_scalar;

    kernel_functions["kvazaar"]["dct"] = dct_scalar;
    kernel_functions["kvazaar"]["idct"] = idct_scalar;
    kernel_functions["kvazaar"]["intra"] = intra_scalar;
    kernel_functions["kvazaar"]["satd"] = satd_scalar;

    kernel_functions["linpack"]["lpack"] = lpack_scalar;

    kernel_functions["xnnpack"]["gemm"] = gemm_scalar;
    kernel_functions["xnnpack"]["spmm"] = spmm_scalar;
}