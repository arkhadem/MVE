
tests_bench = {}
library_list = []

platform_list = ["scalar", "neon", "adreno"]

all_library_list = ["libjpeg",
                    "libpng",
                    "libwebp",
                    "boringssl",
                    "zlib",
                    "skia",
                    "webaudio",
                    "optroutines",
                    "cmsisdsp",
                    "kvazaar",
                    "linpack",
                    "xnnpack"]

def init_all_tests():
    global tests_bench
    global library_list
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
    xnnpack_test = []
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
    optroutines_test.append("csum")
    cmsisdsp_test.append("fir")
    cmsisdsp_test.append("fir_lattice")
    cmsisdsp_test.append("fir_sparse")
    kvazaar_test.append("dct")
    kvazaar_test.append("idct")
    kvazaar_test.append("intra")
    kvazaar_test.append("satd")
    linpack_test.append("lpack")
    xnnpack_test.append("gemm")
    xnnpack_test.append("spmm")

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
    tests_bench["xnnpack"] = xnnpack_test

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
                    "linpack",
                    "xnnpack"]

def init_selected_tests():
    global tests_bench
    global library_list
    optroutines_test = []
    cmsisdsp_test = []
    kvazaar_test = []
    linpack_test = []
    xnnpack_test = []
    optroutines_test.append("csum")
    cmsisdsp_test.append("fir")
    cmsisdsp_test.append("fir_lattice")
    cmsisdsp_test.append("fir_sparse")
    kvazaar_test.append("dct")
    kvazaar_test.append("idct")
    kvazaar_test.append("intra")
    kvazaar_test.append("satd")
    linpack_test.append("lpack")
    xnnpack_test.append("gemm")
    xnnpack_test.append("spmm")

    tests_bench["optroutines"] = optroutines_test
    tests_bench["cmsisdsp"] = cmsisdsp_test
    tests_bench["kvazaar"] = kvazaar_test
    tests_bench["linpack"] = linpack_test
    tests_bench["xnnpack"] = xnnpack_test

    library_list = ["optroutines",
                    "cmsisdsp",
                    "kvazaar",
                    "linpack",
                    "xnnpack"]

xnnpack_layers =[   {"M": 49,	"N": 96,	"K": 48},
                    {"M": 196,	"N": 48,	"K": 24},
                    {"M": 784,	"N": 45,	"K": 12},
                    {"M": 49,	"N": 48,	"K": 192},
                    {"M": 49,	"N": 192,	"K": 48},
                    {"M": 49,	"N": 96,	"K": 96},
                    {"M": 196,	"N": 24,	"K": 96},
                    {"M": 196,	"N": 96,	"K": 24},
                    {"M": 196,	"N": 48,	"K": 48},
                    {"M": 784,	"N": 12,	"K": 48},
                    {"M": 784,	"N": 48,	"K": 12},
                    {"M": 784,	"N": 24,	"K": 24},
                    {"M": 49,	"N": 136,	"K": 68},
                    {"M": 196,	"N": 68,	"K": 34},
                    {"M": 49,	"N": 160,	"K": 80},
                    {"M": 196,	"N": 80,	"K": 40},
                    {"M": 196,	"N": 40,	"K": 96},
                    {"M": 3136,	"N": 16,	"K": 16},
                    {"M": 784,	"N": 62,	"K": 17},
                    {"M": 196,	"N": 48,	"K": 96},
                    {"M": 784,	"N": 24,	"K": 48},
                    {"M": 49,	"N": 68,	"K": 272},
                    {"M": 49,	"N": 272,	"K": 68},
                    {"M": 196,	"N": 34,	"K": 136},
                    {"M": 196,	"N": 136,	"K": 34},
                    {"M": 784,	"N": 17,	"K": 68},
                    {"M": 784,	"N": 68,	"K": 17},
                    {"M": 196,	"N": 120,	"K": 40},
                    {"M": 49,	"N": 200,	"K": 100},
                    {"M": 196,	"N": 100,	"K": 50},
                    {"M": 784,	"N": 58,	"K": 24},
                    {"M": 196,	"N": 48,	"K": 120},
                    {"M": 784,	"N": 72,	"K": 20},
                    {"M": 49,	"N": 80,	"K": 320},
                    {"M": 49,	"N": 320,	"K": 80},
                    {"M": 196,	"N": 40,	"K": 160},
                    {"M": 196,	"N": 160,	"K": 40},
                    {"M": 784,	"N": 20,	"K": 80},
                    {"M": 784,	"N": 80,	"K": 20},
                    {"M": 49,	"N": 96,	"K": 288},
                    {"M": 196,	"N": 144,	"K": 48},
                    {"M": 196,	"N": 48,	"K": 144},
                    {"M": 784,	"N": 24,	"K": 72},
                    {"M": 784,	"N": 88,	"K": 24},
                    {"M": 784,	"N": 24,	"K": 88},
                    {"M": 784,	"N": 88,	"K": 24},
                    {"M": 784,	"N": 88,	"K": 25},
                    {"M": 196,	"N": 96,	"K": 96},
                    {"M": 784,	"N": 96,	"K": 24},
                    {"M": 784,	"N": 48,	"K": 48},
                    {"M": 3136,	"N": 24,	"K": 24},
                    {"M": 196,	"N": 68,	"K": 136},
                    {"M": 784,	"N": 34,	"K": 68},
                    {"M": 196,	"N": 240,	"K": 40},
                    {"M": 196,	"N": 40,	"K": 240},
                    {"M": 49,	"N": 100,	"K": 400},
                    {"M": 49,	"N": 400,	"K": 100},
                    {"M": 196,	"N": 50,	"K": 200},
                    {"M": 196,	"N": 200,	"K": 50},
                    {"M": 784,	"N": 25,	"K": 100},
                    {"M": 784,	"N": 100,	"K": 25},
                    {"M": 49,	"N": 288,	"K": 144},
                    {"M": 196,	"N": 144,	"K": 72},
                    {"M": 784,	"N": 40,	"K": 72},
                    {"M": 784,	"N": 122,	"K": 24},
                    {"M": 196,	"N": 64,	"K": 192},
                    {"M": 196,	"N": 80,	"K": 160},
                    {"M": 784,	"N": 40,	"K": 80},
                    {"M": 49,	"N": 232,	"K": 232},
                    {"M": 196,	"N": 116,	"K": 116},
                    {"M": 784,	"N": 58,	"K": 58},
                    {"M": 49,	"N": 576,	"K": 96},
                    {"M": 49,	"N": 96,	"K": 576},
                    {"M": 196,	"N": 288,	"K": 48},
                    {"M": 3136,	"N": 36,	"K": 24},
                    {"M": 196,	"N": 184,	"K": 80},
                    {"M": 196,	"N": 80,	"K": 184},
                    {"M": 196,	"N": 200,	"K": 80},
                    {"M": 196,	"N": 80,	"K": 200},
                    {"M": 12544,"N": 16,	"K": 16},
                    {"M": 784,	"N": 120,	"K": 36},
                    {"M": 784,	"N": 32,	"K": 144},
                    {"M": 3136,	"N": 72,	"K": 16},
                    {"M": 196,	"N": 80,	"K": 240},
                    {"M": 784,	"N": 120,	"K": 40},
                    {"M": 784,	"N": 40,	"K": 120},
                    {"M": 3136,	"N": 50,	"K": 24},
                    {"M": 196,	"N": 100,	"K": 200},
                    {"M": 784,	"N": 50,	"K": 100},
                    {"M": 49,	"N": 144,	"K": 576},
                    {"M": 49,	"N": 576,	"K": 144},
                    {"M": 196,	"N": 72,	"K": 288},
                    {"M": 196,	"N": 288,	"K": 72},
                    {"M": 784,	"N": 36,	"K": 144},
                    {"M": 784,	"N": 144,	"K": 36},
                    {"M": 3136,	"N": 58,	"K": 24},
                    {"M": 49,	"N": 160,	"K": 576},
                    {"M": 3136,	"N": 60,	"K": 24},
                    {"M": 196,	"N": 384,	"K": 64},
                    {"M": 196,	"N": 64,	"K": 384},
                    {"M": 784,	"N": 192,	"K": 32},
                    {"M": 784,	"N": 32,	"K": 192},
                    {"M": 3136,	"N": 24,	"K": 64},
                    {"M": 3136,	"N": 68,	"K": 24},
                    {"M": 49,	"N": 160,	"K": 672},
                    {"M": 3136,	"N": 72,	"K": 24},
                    {"M": 3136,	"N": 24,	"K": 72},
                    {"M": 49,	"N": 352,	"K": 352},
                    {"M": 196,	"N": 176,	"K": 176},
                    {"M": 784,	"N": 88,	"K": 88},
                    {"M": 12544,"N": 16,	"K": 32},
                    {"M": 3136,	"N": 88,	"K": 24},
                    {"M": 196,	"N": 96,	"K": 384},
                    {"M": 3136,	"N": 24,	"K": 96},
                    {"M": 3136,	"N": 96,	"K": 24},
                    {"M": 49,	"N": 960,	"K": 160},
                    {"M": 49,	"N": 160,	"K": 960},
                    {"M": 49,	"N": 960,	"K": 160},
                    {"M": 49,	"N": 160,	"K": 960},
                    {"M": 196,	"N": 480,	"K": 80},
                    {"M": 784,	"N": 240,	"K": 40},
                    {"M": 196,	"N": 144,	"K": 288},
                    {"M": 784,	"N": 72,	"K": 144},
                    {"M": 3136,	"N": 122,	"K": 24},
                    {"M": 49,	"N": 1024,	"K": 192},
                    {"M": 196,	"N": 112,	"K": 480},
                    {"M": 196,	"N": 232,	"K": 232},
                    {"M": 196,	"N": 576,	"K": 96},
                    {"M": 196,	"N": 96,	"K": 576},
                    {"M": 3136,	"N": 144,	"K": 24},
                    {"M": 3136,	"N": 24,	"K": 144},
                    {"M": 49,	"N": 488,	"K": 488},
                    {"M": 196,	"N": 244,	"K": 244},
                    {"M": 784,	"N": 122,	"K": 122},
                    {"M": 12544,"N": 64,	"K": 16},
                    {"M": 196,	"N": 672,	"K": 112},
                    {"M": 49,	"N": 320,	"K": 960},
                    {"M": 12544,"N": 96,	"K": 16},
                    {"M": 49,	"N": 1280,	"K": 320},
                    {"M": 49,	"N": 1024,	"K": 464},
                    {"M": 196,	"N": 352,	"K": 352},
                    {"M": 784,	"N": 176,	"K": 176},
                    {"M": 49,	"N": 1024,	"K": 512},
                    {"M": 196,	"N": 512,	"K": 256},
                    {"M": 784,	"N": 256,	"K": 128},
                    {"M": 3136,	"N": 128,	"K": 64},
                    {"M": 12544,"N": 64,	"K": 32},
                    {"M": 49,	"N": 1024,	"K": 704},
                    {"M": 196,	"N": 488,	"K": 488},
                    {"M": 784,	"N": 244,	"K": 244},
                    {"M": 49,	"N": 1024,	"K": 1024},
                    {"M": 49,	"N": 2048,	"K": 976}]