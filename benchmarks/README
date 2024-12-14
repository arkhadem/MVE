This directory provides various implementations of the data-parallel mobile workloads of the Swan benchmark suite \[1\], **MVE** functional simulator, and scripts required for mobile measurements and simulations.

## Content

We provide the following implementations for 52 data-parallel kernels of 12 frequently-used mobile libraries:

1. **Scalar** implementations
2. Vectorized using Arm **Neon** intrinsics
3. Vectorized using **MVE** intrinsics
4. Vectorized using 1 dimensional MVE instrinsics for **RVV** evaluation
5. OpenCL implementation for **Adreno** mobile GPU
6. **CUDA** implementation for Duality Cache \[2\] evaluation 

The following table shows the libraries, number of kernels, and their implementations.

<div align="center">

| Library | #Kernels | Scalar | Neon | MVE | RVV | OpenCL | CUDA |
| :-----: | :------: | :----: | :--: | :-: | :-: | :----: | :--: |
| [Linpack](/benchmarks/src/libraries/linpack/) | 1 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [XNNPACK](/benchmarks/src/libraries/xnnpack/) | 2 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CMSIS-DSP](/benchmarks/src/libraries/cmsisdsp/) | 3 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Kvazaar](/benchmarks/src/libraries/kvazaar/) | 4 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Arm Opt. Rout.](/benchmarks/src/libraries/optroutines/) | 1 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Arm Opt. Rout.](/benchmarks/src/libraries/optroutines/) | 4 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [libjpeg](/benchmarks/src/libraries/libjpeg/) | 5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [libpng](/benchmarks/src/libraries/libpng/) | 3 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [libwebp](/benchmarks/src/libraries/libwebp/) | 7 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [Skia](/benchmarks/src/libraries/skia/) | 4 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [Webaudio](/benchmarks/src/libraries/webaudio/) | 5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [zlib](/benchmarks/src/libraries/zlib/) | 2 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |
| [boringssl](/benchmarks/src/libraries/boringssl/) | 3 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: |

</div>

## Directory structure

- [src/benchmark](/src/benchmark/): benchmark infrastructure to configure and launch kernels, and generate input data.
- [src/libraries](/src/libraries/)`/[LIB]/[KER]`: various implementation for `KER` kernel of `LIB` library.
- [src/funcsim](/src/funcsim/): **MVE** functional simulator.
- [src/fake_funcsim](/src/funcsim/): **MVE** fake intrinsic implementations for CPU energy measurements.
- [scripts](scripts/): Measurement scripts for mobile evaluation and simulation scripts.

## Build and Workflow

Please refer to the [main README](/README) for instructions for building and performing measurements and simulations.

## References

\[1\] A. Khadem, D. Fujiki, N. Talati, S. Mahlke and R. Das, "Vector-Processing for Mobile Devices: Benchmark and Analysis," *2023 IEEE International Symposium on Workload Characterization (IISWC)*, Ghent, Belgium, 2023.

\[2\] D. Fujiki, S. Mahlke, and R. Das, "Duality cache for data parallel acceleration," *In Proceedings of the 46th International Symposium on Computer Architecture (ISCA)*, New York, NY, USA, 2019.