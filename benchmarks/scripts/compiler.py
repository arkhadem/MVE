
from ast import Pass
from asyncore import write
import sys
import os

WRITE_EN = 1

def writer(fh, str):
	if WRITE_EN:
		fh.write(str)

def compile(dfg_addr, asm_addr, instr_addr):
	asm_file = None
	dfg_file = None
	instr_file = None

	try:
		asm_file = open(asm_addr, 'r')
	except:
		print("[COMPILER]: File " + asm_addr + " does not exist!")
		return

	try:
		dfg_file = open(dfg_addr, 'r')
	except:
		print("[COMPILER]: File " + dfg_addr + " does not exist!")
		return

	if WRITE_EN:
		try:
			instr_file = open(instr_addr, 'w')
		except:
			print("[COMPILER]: Could not open " + instr_addr + "!")
			exit()

	asm_lines = asm_file.readlines()
	dfg_lines = dfg_file.readlines()

	last_asm_idx = 0
	dfg_line_idx = 0
	num_CPU_instr = 0

	while (dfg_line_idx < len(dfg_lines)):

		dfg_line = dfg_lines[dfg_line_idx]

		if "flushed" in dfg_line:
			dfg_line_idx += 1
			continue

		dfg_split = dfg_line.split()

		# 0: register-baed
		# 1: memory-baed
		dfg_type = -1

		# general parameters
		dfg_dst = ""
		dfg_src1 = ""
		dfg_src2 = ""
		dfg_opc = ""

		# register-based parameters
		dfg_config = ""
		dfg_val = ""

		# memory_based parameters
		dfg_val1 = ""
		dfg_val2 = ""
		dfg_val3 = ""
		dfg_val4 = ""
		dfg_mem_addrs = []

		if ("_mve_load" in dfg_line) or ("_mve_store" in dfg_line) or ("_mve_dict" in dfg_line):
			# memory-based operations
			dfg_type = 1
			dfg_opc = dfg_split[0]
			dfg_dst = dfg_split[1]
			dfg_src1 = dfg_split[2]
			dfg_src2 = dfg_split[3]
			dfg_val1 = dfg_split[4]
			dfg_val2 = dfg_split[5]
			dfg_val3 = dfg_split[6]
			dfg_val4 = dfg_split[7]
			if ("loadr" in dfg_line) or ("storer" in dfg_line):
				dfg_mem_addr_line = dfg_lines[dfg_line_idx + 1]
				dfg_line_idx += 1
				dfg_mem_addr_split = dfg_mem_addr_line.split()
				for dfg_mem_addr in dfg_mem_addr_split:
					dfg_mem_addrs.append(dfg_mem_addr)
		else:
			# register-based operations
			dfg_type = 0
			dfg_opc = dfg_split[0]
			dfg_dst = dfg_split[1]
			dfg_src1 = dfg_split[2]
			dfg_src2 = dfg_split[3]
			dfg_config = dfg_split[4]
			dfg_val = dfg_split[5]

		while dfg_opc not in asm_lines[last_asm_idx]:
			asm_line = asm_lines[last_asm_idx]
			asm_split = asm_line.split()

			if ("inter" in asm_line):
				print("[COMPILER]: Error in line: {asm_line} of file: {asm_addr}")
				exit(-1)
			elif ("func" in asm_line) or ("flusher" in asm_line):
				# Do nothing
				Pass
			else:
				assert (("load" == asm_split[0]) or ("store" == asm_split[0])), f"[COMPILER]: error in line \"{last_asm_idx}\" of \"{asm_addr}\": \"{asm_line}\""
				num_CPU_instr += int(asm_lines[last_asm_idx].split()[2])
				writer(instr_file, f"{asm_split[0]} {asm_split[1]} {num_CPU_instr}\n")
				num_CPU_instr = 0
			last_asm_idx += 1

		num_CPU_instr += int(asm_lines[last_asm_idx].split()[-1])
		last_asm_idx += 1
		
		if dfg_type == 0:
			# it's a register-based instr
			if ("free" in dfg_opc):
				writer(instr_file, f"{dfg_opc}\n")
			else:
				writer(instr_file, f"{dfg_opc} {dfg_dst} {dfg_src1} {dfg_src2} {dfg_config} {dfg_val} {num_CPU_instr}\n")
				num_CPU_instr = 0
		else:
			# it's a memory-based instr
			writer(instr_file, f"{dfg_opc} {dfg_dst} {dfg_src1} {dfg_src2} {dfg_val1} {dfg_val2} {dfg_val3} {dfg_val4} {num_CPU_instr}")
			for dfg_mem_addr in dfg_mem_addrs:
				writer(instr_file, f" {dfg_mem_addr}")
			writer(instr_file, "\n")
			num_CPU_instr = 0

		dfg_line_idx += 1
		
	while last_asm_idx != len(asm_lines):
		asm_line = asm_lines[last_asm_idx]
		asm_split = asm_line.split()
		if ("inter" in asm_line):
			print(f"[COMPILER]: Error in line: {asm_line} of file: {asm_addr}")
			exit(-1)
		elif ("func" in asm_line) or ("flusher" in asm_line):
			# Do nothing
			Pass
		else:
			assert (("load" == asm_split[0]) or ("store" == asm_split[0])), f"[COMPILER]: error in line \"{last_asm_idx}\" \"{asm_line}\""
			num_CPU_instr += int(asm_lines[last_asm_idx].split()[2])
			writer(instr_file, f"{asm_split[0]} {asm_split[1]} {num_CPU_instr}\n")
			num_CPU_instr = 0
		last_asm_idx += 1
	last_asm_idx += 1

