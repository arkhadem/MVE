#!/usr/bin/python

import argparse
import tests
import power_profiler
import performance_profiler
import mask
import general

def run_power_profile(CSV_file, KERNEL_DIR, platform, libraries, kernels, core = None):
	CSV_file.write("Platform,Library,Kernel,Pre-current (mA),Pre-Voltage (mV),Post-Current (mA),Post-Voltage (mV),Power (W)\n")
	for library in libraries:
		for kernel in kernels[library]:
			if library == "xnnpack":
				for layer in tests.xnnpack_layers:
					M = layer["M"]
					N = layer["N"]
					K = layer["K"]
					pre_current, pre_voltage, post_current, post_voltage = power_profiler.get_power(KERNEL_DIR, platform, library, kernel, core, M, N, K)
					power = (pre_voltage + post_voltage) * (pre_current - post_current) / 2.0000 / 1000000.0000
					CSV_file.write(f"{platform},{library},{kernel}_{M}_{N}_{K},{pre_current},{pre_voltage},{post_current},{post_voltage},{power}\n")
			else:
				pre_current, pre_voltage, post_current, post_voltage = power_profiler.get_power(KERNEL_DIR, platform, library, kernel, core)
				power = (pre_voltage + post_voltage) * (pre_current - post_current) / 2.0000 / 1000000.0000
				CSV_file.write(f"{platform},{library},{kernel},{pre_current},{pre_voltage},{post_current},{post_voltage},{power}\n")

def run_performance_profile_scalar_neon(CSV_file, KERNEL_DIR, platform, libraries, kernels, core = None):
	CSV_file.write("Platform,Library,Kernel,Iterations,Total Time (us),Iteration Time (us)\n")
	for library in libraries:
		for kernel in kernels[library]:
			if library == "xnnpack":
				for layer in tests.xnnpack_layers:
					M = layer["M"]
					N = layer["N"]
					K = layer["K"]
					iterations, total_time, iteration_time = performance_profiler.get_performance_scalar_neon(KERNEL_DIR, platform, library, kernel, core, M, N, K)
					CSV_file.write(f"{platform},{library},{kernel}_{M}_{N}_{K},{iterations},{total_time},{iteration_time}\n")
			else:
				iterations, total_time, iteration_time = performance_profiler.get_performance_scalar_neon(KERNEL_DIR, platform, library, kernel, core)
				CSV_file.write(f"{platform},{library},{kernel},{iterations},{total_time},{iteration_time}\n")

def run_performance_profile_adreno(CSV_file, KERNEL_DIR, platform, libraries, kernels, core = None):
	CSV_file.write("Platform,Library,Kernel,Iterations,Total Time (us),Iteration Time (us), Create Buffer Time (us), Map Buffer Time (us), MemCpy Time (us), Kernel Launch Time (us), Kernel Execute Time (us)\n")
	for library in libraries:
		for kernel in kernels[library]:
			if library == "xnnpack":
				for layer in tests.xnnpack_layers:
					M = layer["M"]
					N = layer["N"]
					K = layer["K"]
					iterations, total_time, iteration_time, create_buffer_time, map_buffer_time, memcpy_time, kernel_launch_time, kernel_execute_time = performance_profiler.get_performance_adreno(KERNEL_DIR, platform, library, kernel, core, M, N, K)
					CSV_file.write(f"{platform},{library},{kernel}_{M}_{N}_{K},{iterations},{total_time},{iteration_time},{create_buffer_time},{map_buffer_time},{memcpy_time},{kernel_launch_time},{kernel_execute_time}\n")
			else:
				iterations, total_time, iteration_time, create_buffer_time, map_buffer_time, memcpy_time, kernel_launch_time, kernel_execute_time = performance_profiler.get_performance_adreno(KERNEL_DIR, platform, library, kernel, core)
				CSV_file.write(f"{platform},{library},{kernel},{iterations},{total_time},{iteration_time},{create_buffer_time},{map_buffer_time},{memcpy_time},{kernel_launch_time},{kernel_execute_time}\n")

def main():
	parser = argparse.ArgumentParser(description="MVE phone utility script.")
	parser.add_argument("--measurement", help="type of the measurement", choices=["power", "performance"], required=True)
	parser.add_argument("--output", help="output CSV file", required=True)
	parser.add_argument("--platform", help="experiment platfrom (default: scalar)", default="scalar", choices=tests.platform_list)
	parser.add_argument("--library", help="experiment library (default: all)", default="all", choices=["all"] + tests.all_library_list)
	parser.add_argument("--kernel", help="experiment kernel; choose specific kernel only when a specific library is selected (default: all)", default="all")
	parser.add_argument("--directory", help="expetiment directory path on phone (default: /data/local/tmp/MVE)", default="/data/local/tmp/MVE")
	parser.add_argument("--device", help="target device serial number OR IP:PORT; use when multiple devices are connected (default: first online device)", default=None)
	parser.add_argument("--core", help="experiment core; edit mask.py to use this option (default: no pinning to a core)", default=None, choices=mask.masks.keys())
	parser.add_argument("--verbose", help="print all executed commands and their logs", default=False, action='store_true')
	args = parser.parse_args()

	CSV_file = open(args.output, 'w')
	KERNEL_DIR = args.directory
	platform = args.platform
	libraries = None
	if platform in ["adreno"]:
		libraries = tests.init_selected_tests()
	else:
		libraries = tests.init_all_tests()
	if args.library != "all":
		assert args.library in libraries, f"library \"{args.library}\" is not supported for platform \"{platform}\"!"
		libraries = [args.library]
	kernels = tests.tests_bench
	if args.kernel != "all":
		assert args.library != "all", "please also specify the library name of your specific kernel"
		kernels = {args.library: [args.kernel]}
	core = args.core
	if core == None:
		print("Warning: please configure \"masks\" of \"mask.py\", and use \"-c <core_name>\" to pin the benchmark process to a specific core!")     
	general.VERBOSE = args.verbose
	general.check_device(args.device)
	general.run_shell_command(f"mkdir {KERNEL_DIR}/")
	general.run_command(f"adb -s {general.device_serial_number} push ./benchmark_phone_{platform} {KERNEL_DIR}/")
	if platform == "adreno":
		general.run_command(f"adb -s {general.device_serial_number} push src/libraries/*/*/*.cl {KERNEL_DIR}/")
	if args.measurement == "power":
		if platform not in ["scalar", "neon"]:
			print(f"Error: platform \"{platform}\" is not supported for power measurements!")
			exit(-1)
		run_power_profile(CSV_file, KERNEL_DIR, platform, libraries, kernels, core)
	else:
		run_performance_profile_scalar_neon(CSV_file, KERNEL_DIR, platform, libraries, kernels, core)
	CSV_file.close()

if __name__ == "__main__":
	main()