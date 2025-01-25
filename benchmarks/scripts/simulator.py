#!/usr/bin/python

import argparse
import time
import tests
import general
import compiler
import parser
import os

DYNAMORIO_ROOT = "../tools/DynamoRIO"
Simulator_ROOT = "../simulator"
LANE_NUMS = {"bs": 8192, "bh": 2048, "bp": 256, "ac": 2048}

all_schemes = ["bs", "bh", "bp", "ac"]
all_isas = ["mve", "rvv"]
all_executions = ["ino", "ooo", "dvi", "ora"]

def get_libraries_kernels(scheme, isa, args, exe="ino"):
	if scheme != "bs" or isa != "mve" or exe != "ino":
		tests.init_selected_tests()
	else:
		tests.init_all_tests()
	libraries = tests.library_list
	if args.library != "all":
		assert args.library in libraries, f"library \"{args.library}\" is not supported for isa \"{isa}\" and scheme \"{scheme}\"!"
		libraries = [args.library]
	kernels = tests.tests_bench
	if args.kernel != "all":
		assert args.library != "all", "please also specify the library name of your specific kernel"
		kernels = {args.library: [args.kernel]}
	return libraries, kernels

def run_dynamorio(directory, schemes, isas, exes, args):
	for scheme in schemes:
		LANE_NUM = LANE_NUMS[scheme]
		for isa in isas:
			libraries, kernels = get_libraries_kernels(scheme, isa, args)
			for library in libraries:
				for kernel in kernels[library]:
					if library == "xnnpack":
						for layer in tests.xnnpack_layers:
							M = layer["M"]
							N = layer["N"]
							K = layer["K"]
							dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.dfg"
							asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.asm"
							general.run_command(f"mkdir -p {directory}/{scheme}/{isa}/{library}/")
							general.run_command(f"rm *.dfg")
							general.run_command(f"rm *.log")
							time.sleep(1)
							trace_function = f"{kernel}_{isa}"
							command = f"{DYNAMORIO_ROOT}/bin64/drrun "
							command += f"-c {DYNAMORIO_ROOT}/samples/build/bin/libinstrace.so "
							command += f"-trace_function {trace_function} "
							command += f"-- ./benchmark_local_{isa} -l {library} -k {kernel} -n {LANE_NUM} -xm {M} -xn {N} -xk {K}"
							log = general.run_command(command)
							if "Finishing MVE computation" not in log:
								print(f"Error: benchmark_local_{isa} did not finish correctly!\nLOG:{log}")
								exit(-1)
							general.run_command(f"mv {library}_{kernel}_{LANE_NUM}_{M}_{N}_{K}.dfg {dfg_file}")
							general.run_command(f"mv *.log {asm_file}")
					else:
						dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.dfg"
						asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.asm"
						general.run_command(f"mkdir -p {directory}/{scheme}/{isa}/{library}/")
						general.run_command(f"rm *.dfg")
						general.run_command(f"rm *.log")
						time.sleep(1)
						trace_function = f"{kernel}_{isa}"
						command = f"{DYNAMORIO_ROOT}/bin64/drrun "
						command += f"-c {DYNAMORIO_ROOT}/samples/build/bin/libinstrace.so "
						command += f"-trace_function {trace_function} "
						command += f"-- ./benchmark_local_{isa} -l {library} -k {kernel} -n {LANE_NUM}"
						log = general.run_command(command)
						if "Finishing MVE computation" not in log:
							print(f"Error: benchmark_local_{isa} did not finish correctly!\nLOG:{log}")
							exit(-1)
						general.run_command(f"mv {library}_{kernel}_{LANE_NUM}.dfg {dfg_file}")
						general.run_command(f"mv *.log {asm_file}")

def run_compiler(directory, schemes, isas, exes, args):
	for scheme in schemes:
		for isa in isas:
			output = f"{isa}_{scheme}_compile.csv"
			print(f"Compiling ISA {isa} and scheme {scheme} files, writing instruction summary to {output}...")
			CSV_file = open(output, "w")
			CSV_file.write("Scheme,ISA,Library,Kernel,Config,Move,Mem Access,Arithmetic,Scalar\n")
			libraries, kernels = get_libraries_kernels(scheme, isa, args)
			for library in libraries:
				for kernel in kernels[library]:
					if library == "xnnpack":
						for layer in tests.xnnpack_layers:
							M = layer["M"]
							N = layer["N"]
							K = layer["K"]
							dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.dfg"
							asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.asm"
							instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.instr"
							Config, Move, Mem_Access, Arithmetic, Scalar = compiler.compile(dfg_file, asm_file, instr_file)
							CSV_file.write(f"{scheme},{isa},{library},{kernel}_{M}_{N}_{K},{Config},{Move},{Mem_Access},{Arithmetic},{Scalar}\n")
					else:
						dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.dfg"
						asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.asm"
						instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.instr"
						Config, Move, Mem_Access, Arithmetic, Scalar = compiler.compile(dfg_file, asm_file, instr_file)
						CSV_file.write(f"{scheme},{isa},{library},{kernel},{Config},{Move},{Mem_Access},{Arithmetic},{Scalar}\n")
			CSV_file.close()

def run_simulator(directory, schemes, isas, exes, args):
	for exe in exes:
		for scheme in schemes:
			for isa in isas:
				if exe != "ino" and isa == "rvv":
					continue
				libraries, kernels = get_libraries_kernels(scheme, isa, args, exe)
				for library in libraries:
					for kernel in kernels[library]:
						if library == "xnnpack":
							for layer in tests.xnnpack_layers:
								M = layer["M"]
								N = layer["N"]
								K = layer["K"]
								instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}_{M}_{N}_{K}.instr"
								ram_dir = f"{directory}/{scheme}/{isa}/{exe}/{library}"
								ram_file = f"{ram_dir}/{kernel}_{M}_{N}_{K}.ram"
								exists = False
								if os.path.isfile(ram_file):
									with open(ram_file, "r") as f:
										if "ramulator.active_cycles" in f.read():
											exists = True
								if exists:
									print(f"Skipping {ram_file} as it is already simulated!")
									continue
								
								command = f"mkdir -p {ram_dir} 2>&1 > /dev/null; sleep 1;"
								command += f"{Simulator_ROOT}/simulator_{scheme}_{exe} "
								command += f"{Simulator_ROOT}/configs/LPDDR4-config-MVE.cfg --mode=MVE --core=1 prime "
								command += f"--stats {ram_file} "
								command += instr_file + " "
								general.add_run_command(command)
						else:
							instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.instr"
							ram_dir = f"{directory}/{scheme}/{isa}/{exe}/{library}"
							ram_file = f"{ram_dir}/{kernel}.ram"
							exists = False
							if os.path.isfile(ram_file):
								with open(ram_file, "r") as f:
									if "ramulator.active_cycles" in f.read():
										exists = True
							if exists:
								print(f"Skipping {ram_file} as it is already simulated!")
								continue
							
							command = f"mkdir -p {ram_dir} 2>&1 > /dev/null; sleep 1;"
							command += f"{Simulator_ROOT}/simulator_{scheme}_{exe} "
							command += f"{Simulator_ROOT}/configs/LPDDR4-config-MVE.cfg --mode=MVE --core=1 prime "
							command += f"--stats {ram_file} "
							command += instr_file + " "
							general.add_run_command(command)
	general.run_parallel_commands()

def parse_simulation(directory, schemes, isas, exes, args):
	for exe in exes:
		for scheme in schemes:
			for isa in isas:
				if exe != "ino" and isa == "rvv":
					continue
				output = f"{isa}_{scheme}_{exe}.csv"
				print(f"Parsing ISA {isa} and scheme {scheme} results into {output}...")
				CSV_file = open(output, "w")
				CSV_file.write("Scheme,ISA,Exe,Library,Kernel,Idle Time (us),Compute Time (us),Data Access Time (us),Total Time (us),Compute Energy (mJ),Data Access Energy (mJ),Total Energy (mJ)\n")
				libraries, kernels = get_libraries_kernels(scheme, isa, args, exe)
				for library in libraries:
					for kernel in kernels[library]:
						if library == "xnnpack":
							for layer in tests.xnnpack_layers:
								M = layer["M"]
								N = layer["N"]
								K = layer["K"]
								ram_file = f"{directory}/{scheme}/{isa}/{exe}/{library}/{kernel}_{M}_{N}_{K}.ram"
								idle_time, compute_time, data_access_time, total_time, compute_energy, data_access_energy, total_energy = parser.parse(ram_file)
								CSV_file.write(f"{scheme},{isa},{exe},{library},{kernel}_{M}_{N}_{K},{float(idle_time):0.8f},{float(compute_time):0.8f},{float(data_access_time):0.8f},{float(total_time):0.8f},{float(compute_energy):0.8f},{float(data_access_energy):0.8f},{float(total_energy):0.8f}\n")
						else:
							ram_file = f"{directory}/{scheme}/{isa}/{exe}/{library}/{kernel}.ram"
							idle_time, compute_time, data_access_time, total_time, compute_energy, data_access_energy, total_energy = parser.parse(ram_file)
							CSV_file.write(f"{scheme},{isa},{exe},{library},{kernel},{float(idle_time):0.8f},{float(compute_time):0.8f},{float(data_access_time):0.8f},{float(total_time):0.8f},{float(compute_energy):0.8f},{float(data_access_energy):0.8f},{float(total_energy):0.8f}\n")
				CSV_file.close()

def main():
	parser = argparse.ArgumentParser(description="MVE phone utility script.")
	parser.add_argument("--action", help="type of the action", choices=["benchmark", "compile", "simulate", "parse"], required=True)
	parser.add_argument("--directory", help="input/output directory", required=True)
	parser.add_argument("--scheme", help="in-cache computing scheme (default: all)", choices=["all", "bs", "bh", "bp", "ac"], default="all")
	parser.add_argument("--isa", help="instruction-set architecture (default: all)", choices=["all", "mve", "rvv"], default="all")
	parser.add_argument("--exe", help="execution type (default: all)", choices=["ino", "ooo", "dvi", "ora"], default="all")
	parser.add_argument("--library", help="experiment library (default: all)", default="all", choices=["all"] + tests.all_library_list)
	parser.add_argument("--kernel", help="experiment kernel; choose specific kernel only when a specific library is selected (default: all)", default="all")
	parser.add_argument("--verbose", help="print all executed commands and their logs", default=False, action='store_true')
	parser.add_argument("-j", help="number of simulation threads", default=32, type=int)
	args = parser.parse_args()

	action = args.action
	directory = args.directory
	if directory[-1] == "/":
		directory = directory[:-1]
	schemes = None
	if args.scheme == "all":
		schemes = all_schemes
	else:
		assert args.scheme in all_schemes, f"scheme \"{args.scheme}\" is not supported!"
		schemes = [args.scheme]
	isas = None
	if args.isa == "all":
		isas = all_isas
	else:
		assert args.isa in all_isas, f"isa \"{args.isa}\" is not supported!"
		isas = [args.isa]
	exes = None
	if args.exe == "all":
		exes = all_executions
	else:
		assert args.exe in all_executions, f"execution type \"{args.exe}\" is not supported!"
		exes = [args.exe]
	general.NUM_PARALLEL_THREADS = args.j
	general.VERBOSE = args.verbose
	general.run_command(f"mkdir {directory}/")
	if action == "benchmark":
		run_dynamorio(directory, schemes, isas, exes, args)
	elif action == "compile":
		run_compiler(directory, schemes, isas, exes, args)
	elif action == "simulate":
		run_simulator(directory, schemes, isas, exes, args)
	elif action == "parse":
		parse_simulation(directory, schemes, isas, exes, args)
	else:
		print(f"Error: action \"{action}\" is not supported!")
		exit(-1)

if __name__ == "__main__":
	main()
