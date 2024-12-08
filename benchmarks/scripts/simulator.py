#!/usr/bin/python

import argparse
import time
import tests
import general
import compiler

DYNAMORIO_ROOT = "../tools/DynamoRIO"
RAMULATOR_ROOT = "../ramulator"
LANE_NUMS = {"bs": 8192, "bh": 2048, "bp": 256, "ac": 2048}

def run_dynamorio(directory, scheme, isa, libraries, kernels):
	LANE_NUM = LANE_NUMS[scheme]
	for library in libraries:
		for kernel in kernels[library]:
			dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.dfg"
			asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.asm"
			general.run_command(f"mkdir -p {directory}/{scheme}/{isa}/{library}/{kernel}/")
			general.run_command(f"rm {library}_{kernel}_{LANE_NUM}.dfg")
			general.run_command(f"rm *.log")
			time.sleep(1)
			trace_function = f"{kernel}_{isa}"
			command = f"{DYNAMORIO_ROOT}/bin64/drrun "
			command += f"-c {DYNAMORIO_ROOT}/samples/build/bin/libinstrace.so "
			command += f"-trace_function {trace_function} "
			command += f"-- ./browser_bench_local_lime -l {library} -k {kernel} -n {LANE_NUM}"
			general.run_command(command)
			general.run_command(f"mv {library}_{kernel}_{LANE_NUM}.dfg {dfg_file}")
			general.run_command(f"mv *.log {asm_file}")

def run_compiler(directory, scheme, isa, libraries, kernels):
	for library in libraries:
		for kernel in kernels[library]:
			dfg_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.dfg"
			asm_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.asm"
			instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.instr"
			compiler.compile(dfg_file, asm_file, instr_file)

def run_simulator(directory, scheme, isa, libraries, kernels):
	for library in libraries:
		for kernel in kernels[library]:
			instr_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.instr"
			ram_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.ram"
			command = f"{RAMULATOR_ROOT}/ramulator_{isa}_{scheme} "
			command += f"{RAMULATOR_ROOT}/configs/LPDDR4-config-MVE.cfg --mode=mve --core=1 prime "
			command += f"--stats {ram_file} "
			command += instr_file + " "
			command = command.replace("(", "\(")
			command = command.replace(")", "\)")
			general.run_command(command)

def parse_simulation(directory, scheme, isa, libraries, kernels):
	CSV_file.write("Scheme,ISA,Library,Kernel,Idle Time (us),Compute Time (us),Data Access Time (us),Total Time (us),Compute Energy (mJ),Data Access Energy (mJ),Total Energy (mJ)\n")
	for library in libraries:
		for kernel in kernels[library]:
			ram_file = f"{directory}/{scheme}/{isa}/{library}/{kernel}.ram"
			idle_time, compute_time, data_access_time, total_time, compute_energy, data_access_energy, total_energy = parser.parse()
			CSV_file.write(f"{scheme},{isa},{library},{kernel},{idle_time},{compute_time},{data_access_time},{total_time},{compute_energy},{data_access_energy},{total_energy}\n")

def main():
	parser = argparse.ArgumentParser(description="MVE phone utility script.")
	parser.add_argument("--action", help="type of the action", choices=["benchmark", "compile", "simulate", "parse"], required=True)
	parser.add_argument("--directory", help="input/output directory", required=True)
	parser.add_argument("--output", help="output CSV file (only for simulate)", default=None)
	parser.add_argument("--scheme", help="in-cache computing scheme (default: bs)", choices=["bs", "bh", "bp", "ac"], default="bs")
	parser.add_argument("--isa", help="instruction-set architecture (default: mve)", choices=["mve", "rvv"], default="MVE")
	parser.add_argument("--library", help="experiment library (default: all)", default="all", choices=["all"] + tests.all_library_list)
	parser.add_argument("--kernel", help="experiment kernel; choose specific kernel only when a specific library is selected (default: all)", default="all")
	parser.add_argument("--verbose", help="print all executed commands and their logs", default=False, action='store_true')
	args = parser.parse_args()

	action = args.action
	if action == "parse":
		assert args.output != None, "please specify the output CSV file for parsing the simulation!"
	else:
		assert args.output == None, "output CSV file is only for parsing the simulation!"
	directory = args.directory
	if directory[-1] == "/":
		directory = directory[:-1]
	scheme = args.scheme
	isa = args.isa
	if scheme != "bs" or isa != "mve":
		libraries = tests.selected_library_list
		if args.library != "all":
			assert args.library in libraries, f"library \"{args.library}\" is not supported for isa \"{isa}\" and scheme \"{scheme}\"!"
			libraries = [args.library]
	else:
		libraries = tests.all_library_list
		if args.library != "all":
			libraries = [args.library]
	kernels = tests.tests_bench
	if args.kernel != "all":
		assert args.library != "all", "please also specify the library name of your specific kernel"
		kernels = {args.library: [args.kernel]}	
	general.VERBOSE = args.verbose
	general.run_command(f"mkdir {directory}/")
	if action == "benchmark":
		run_dynamorio(directory, scheme, isa, libraries, kernels):
	elif action == "compile":
		run_compiler(directory, scheme, isa, libraries, kernels):
	elif action == "simulate":
		run_simulator(directory, scheme, isa, libraries, kernels):
	elif action == "parse":
		parse_simulation(directory, scheme, isa, libraries, kernels):
	else:
		print(f"Error: action \"{action}\" is not supported!")
		exit(-1)

if __name__ == "__main__":
	main()
