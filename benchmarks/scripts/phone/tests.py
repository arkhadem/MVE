
libjpeg_test = []
libpng_test = []
libwebp_test = []
boringssl_test = []
zlib_test = []
skia_test = []
webaudio_test = []
optroutines_test = []
cmsisdsp_test = []
kvazaar_test = []
linpack_test = []

libjpeg_test.append("rgb_to_gray")
libjpeg_test.append("ycbcr_to_rgb")
libjpeg_test.append("upsample")
libjpeg_test.append("downsample")
libjpeg_test.append("huffman_encode")
libpng_test.append("read_sub")
libpng_test.append("read_up")
libpng_test.append("expand_palette")
libwebp_test.append("sharp_update_rgb")
libwebp_test.append("sharp_filter_row")
libwebp_test.append("apply_alpha_multiply")
libwebp_test.append("dispatch_alpha")
libwebp_test.append("tm_prediction")
libwebp_test.append("vertical_filter")
libwebp_test.append("gradient_filter")
boringssl_test.append("aes")
boringssl_test.append("des")
boringssl_test.append("chacha20")
zlib_test.append("adler32")
zlib_test.append("crc32")
skia_test.append("convolve_horizontally")
skia_test.append("convolve_vertically")
skia_test.append("row_blend")
skia_test.append("row_opaque")
webaudio_test.append("is_audible")
webaudio_test.append("copy_with_gain")
webaudio_test.append("copy_with_sample")
webaudio_test.append("sum_from")
webaudio_test.append("handle_nan")
optroutines_test.append("memchr")
optroutines_test.append("memcmp")
optroutines_test.append("memset")
optroutines_test.append("strlen")
cmsisdsp_test.append("fir")
cmsisdsp_test.append("fir_lattice")
cmsisdsp_test.append("fir_sparse")
kvazaar_test.append("dct")
kvazaar_test.append("idct")
kvazaar_test.append("intra")
kvazaar_test.append("satd")
linpack_test.append("lpack")

tests_bench = {}

tests_bench["libjpeg"] = libjpeg_test
tests_bench["libpng"] = libpng_test
tests_bench["libwebp"] = libwebp_test
tests_bench["boringssl"] = boringssl_test
tests_bench["zlib"] = zlib_test
tests_bench["skia"] = skia_test
tests_bench["webaudio"] = webaudio_test
tests_bench["optroutines"] = optroutines_test
tests_bench["cmsisdsp"] = cmsisdsp_test
tests_bench["kvazaar"] = kvazaar_test
tests_bench["linpack"] = linpack_test

library_list = ["libjpeg",
                "libpng",
                "libwebp",
                "boringssl",
                "zlib",
                "skia",
                "webaudio",
                "optroutines",
                "cmsisdsp",
                "kvazaar",
                "linpack"]

platform_list = ["scalar", "neon"]