
from ast import Pass
import sys
import os

WRITE_EN = 1

CONFIG = 0
MOVE = 1
MEMORY = 2
ARITHMETIC = 3
SCALAR = 4
MAX = 5

instruction_dict = {
	"mve_init_dims": CONFIG,
	"_mve_set_load_stride": CONFIG,
	"_mve_set_store_stride": CONFIG,
	"_mve_set_dim_count": CONFIG,
	"_mve_set_dim_length": CONFIG,
	"_mve_set_mask": CONFIG,
	"_mve_set_active_element": CONFIG,
	"_mve_unset_active_element": CONFIG,
	"_mve_set_only_element": CONFIG,
	"_mve_unset_only_element": CONFIG,
	"_mve_set_all_elements": CONFIG,
	"_mve_unset_all_elements": CONFIG,
	"_mve_shirs_b": ARITHMETIC,
	"_mve_shirs_w": ARITHMETIC,
	"_mve_shirs_dw": ARITHMETIC,
	"_mve_shirs_qw": ARITHMETIC,
	"_mve_shiru_b": ARITHMETIC,
	"_mve_shiru_w": ARITHMETIC,
	"_mve_shiru_dw": ARITHMETIC,
	"_mve_shiru_qw": ARITHMETIC,
	"_mve_shil_b": ARITHMETIC,
	"_mve_shil_w": ARITHMETIC,
	"_mve_shil_dw": ARITHMETIC,
	"_mve_shil_qw": ARITHMETIC,
	"_mve_rotir_b": ARITHMETIC,
	"_mve_rotir_w": ARITHMETIC,
	"_mve_rotir_dw": ARITHMETIC,
	"_mve_rotir_qw": ARITHMETIC,
	"_mve_rotil_b": ARITHMETIC,
	"_mve_rotil_w": ARITHMETIC,
	"_mve_rotil_dw": ARITHMETIC,
	"_mve_rotil_qw": ARITHMETIC,
	"_mve_shrrs_b": ARITHMETIC,
	"_mve_shrrs_w": ARITHMETIC,
	"_mve_shrrs_dw": ARITHMETIC,
	"_mve_shrrs_qw": ARITHMETIC,
	"_mve_shrru_b": ARITHMETIC,
	"_mve_shrru_w": ARITHMETIC,
	"_mve_shrru_dw": ARITHMETIC,
	"_mve_shrru_qw": ARITHMETIC,
	"_mve_shrl_b": ARITHMETIC,
	"_mve_shrl_w": ARITHMETIC,
	"_mve_shrl_dw": ARITHMETIC,
	"_mve_shrl_qw": ARITHMETIC,
	"_mve_set1_b": ARITHMETIC,
	"_mve_set1_w": ARITHMETIC,
	"_mve_set1_dw": ARITHMETIC,
	"_mve_set1_qw": ARITHMETIC,
	"_mve_set1_hf": ARITHMETIC,
	"_mve_set1_f": ARITHMETIC,
	"_mve_set1_df": ARITHMETIC,
	"_mve_load_b": MEMORY,
	"_mve_load_w": MEMORY,
	"_mve_load_dw": MEMORY,
	"_mve_load_qw": MEMORY,
	"_mve_load_hf": MEMORY,
	"_mve_load_f": MEMORY,
	"_mve_load_df": MEMORY,
	"_mve_store_b": MEMORY,
	"_mve_store_w": MEMORY,
	"_mve_store_dw": MEMORY,
	"_mve_store_qw": MEMORY,
	"_mve_store_hf": MEMORY,
	"_mve_store_f": MEMORY,
	"_mve_store_df": MEMORY,
	"_mve_loadr_b": MEMORY,
	"_mve_loadr_w": MEMORY,
	"_mve_loadr_dw": MEMORY,
	"_mve_loadr_qw": MEMORY,
	"_mve_loadr_hf": MEMORY,
	"_mve_loadr_f": MEMORY,
	"_mve_loadr_df": MEMORY,
	"_mve_storer_b": MEMORY,
	"_mve_storer_w": MEMORY,
	"_mve_storer_dw": MEMORY,
	"_mve_storer_qw": MEMORY,
	"_mve_storer_hf": MEMORY,
	"_mve_storer_f": MEMORY,
	"_mve_storer_df": MEMORY,
	"_mve_loadro_b": MEMORY,
	"_mve_loadro_w": MEMORY,
	"_mve_loadro_dw": MEMORY,
	"_mve_loadro_qw": MEMORY,
	"_mve_loadro_hf": MEMORY,
	"_mve_loadro_f": MEMORY,
	"_mve_loadro_df": MEMORY,
	"_mve_storero_b": MEMORY,
	"_mve_storero_w": MEMORY,
	"_mve_storero_dw": MEMORY,
	"_mve_storero_qw": MEMORY,
	"_mve_storero_hf": MEMORY,
	"_mve_storero_f": MEMORY,
	"_mve_storero_df": MEMORY,
	"_mve_dict_b": MEMORY,
	"_mve_dict_w": MEMORY,
	"_mve_dict_dw": MEMORY,
	"_mve_dict_qw": MEMORY,
	"_mve_add_b": ARITHMETIC,
	"_mve_add_w": ARITHMETIC,
	"_mve_add_dw": ARITHMETIC,
	"_mve_add_qw": ARITHMETIC,
	"_mve_add_hf": ARITHMETIC,
	"_mve_add_f": ARITHMETIC,
	"_mve_add_df": ARITHMETIC,
	"_mve_sub_b": ARITHMETIC,
	"_mve_sub_w": ARITHMETIC,
	"_mve_sub_dw": ARITHMETIC,
	"_mve_sub_qw": ARITHMETIC,
	"_mve_sub_hf": ARITHMETIC,
	"_mve_sub_f": ARITHMETIC,
	"_mve_sub_df": ARITHMETIC,
	"_mve_mul_b": ARITHMETIC,
	"_mve_mul_w": ARITHMETIC,
	"_mve_mul_dw": ARITHMETIC,
	"_mve_mul_qw": ARITHMETIC,
	"_mve_mul_hf": ARITHMETIC,
	"_mve_mul_f": ARITHMETIC,
	"_mve_mul_df": ARITHMETIC,
	"_mve_mulmodp_dw": ARITHMETIC,
	"_mve_min_b": ARITHMETIC,
	"_mve_min_w": ARITHMETIC,
	"_mve_min_dw": ARITHMETIC,
	"_mve_min_qw": ARITHMETIC,
	"_mve_min_hf": ARITHMETIC,
	"_mve_min_f": ARITHMETIC,
	"_mve_min_df": ARITHMETIC,
	"_mve_max_b": ARITHMETIC,
	"_mve_max_w": ARITHMETIC,
	"_mve_max_dw": ARITHMETIC,
	"_mve_max_qw": ARITHMETIC,
	"_mve_max_hf": ARITHMETIC,
	"_mve_max_f": ARITHMETIC,
	"_mve_max_df": ARITHMETIC,
	"_mve_xor_b": ARITHMETIC,
	"_mve_xor_w": ARITHMETIC,
	"_mve_xor_dw": ARITHMETIC,
	"_mve_xor_qw": ARITHMETIC,
	"_mve_and_b": ARITHMETIC,
	"_mve_and_w": ARITHMETIC,
	"_mve_and_dw": ARITHMETIC,
	"_mve_and_qw": ARITHMETIC,
	"_mve_or_b": ARITHMETIC,
	"_mve_or_w": ARITHMETIC,
	"_mve_or_dw": ARITHMETIC,
	"_mve_or_qw": ARITHMETIC,
	"_mve_cmpeq_b": ARITHMETIC,
	"_mve_cmpeq_w": ARITHMETIC,
	"_mve_cmpeq_dw": ARITHMETIC,
	"_mve_cmpeq_qw": ARITHMETIC,
	"_mve_cmpeq_hf": ARITHMETIC,
	"_mve_cmpeq_f": ARITHMETIC,
	"_mve_cmpeq_df": ARITHMETIC,
	"_mve_cmpneq_b": ARITHMETIC,
	"_mve_cmpneq_w": ARITHMETIC,
	"_mve_cmpneq_dw": ARITHMETIC,
	"_mve_cmpneq_qw": ARITHMETIC,
	"_mve_cmpneq_hf": ARITHMETIC,
	"_mve_cmpneq_f": ARITHMETIC,
	"_mve_cmpneq_df": ARITHMETIC,
	"_mve_cmpgte_b": ARITHMETIC,
	"_mve_cmpgte_w": ARITHMETIC,
	"_mve_cmpgte_dw": ARITHMETIC,
	"_mve_cmpgte_qw": ARITHMETIC,
	"_mve_cmpgte_hf": ARITHMETIC,
	"_mve_cmpgte_f": ARITHMETIC,
	"_mve_cmpgte_df": ARITHMETIC,
	"_mve_cmpgt_b": ARITHMETIC,
	"_mve_cmpgt_w": ARITHMETIC,
	"_mve_cmpgt_dw": ARITHMETIC,
	"_mve_cmpgt_qw": ARITHMETIC,
	"_mve_cmpgt_hf": ARITHMETIC,
	"_mve_cmpgt_f": ARITHMETIC,
	"_mve_cmpgt_df": ARITHMETIC,
	"_mve_cvt_wtob": MOVE,
	"_mve_cvt_dwtow": MOVE,
	"_mve_cvt_dwtob": MOVE,
	"_mve_cvtu_btow": MOVE,
	"_mve_cvts_btow": MOVE,
	"_mve_cvtu_btodw": MOVE,
	"_mve_cvts_btodw": MOVE,
	"_mve_cvtu_wtodw": MOVE,
	"_mve_cvts_wtodw": MOVE,
	"_mve_cvtu_dwtoqw": MOVE,
	"_mve_cvts_dwtoqw": MOVE,
	"_mve_btos": MOVE,
	"_mve_wtos": MOVE,
	"_mve_dwtos": MOVE,
	"_mve_qwtos": MOVE,
	"_mve_hftos": MOVE,
	"_mve_ftos": MOVE,
	"_mve_dftos": MOVE,
	"_mve_assign_b": MOVE,
	"_mve_assign_w": MOVE,
	"_mve_assign_dw": MOVE,
	"_mve_assign_qw": MOVE,
	"_mve_assign_hf": MOVE,
	"_mve_assign_f": MOVE,
	"_mve_assign_df": MOVE,
	"_mve_copy_b": MOVE,
	"_mve_copy_w": MOVE,
	"_mve_copy_dw": MOVE,
	"_mve_copy_qw": MOVE,
	"_mve_copy_hf": MOVE,
	"_mve_copy_f": MOVE,
	"_mve_copy_df": MOVE}

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

	instruction_count = [0] * MAX

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
				instruction_count[SCALAR] += (num_CPU_instr + 1)
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
				instruction_count[instruction_dict[dfg_opc]] += 1
				instruction_count[SCALAR] += num_CPU_instr
				num_CPU_instr = 0
		else:
			# it's a memory-based instr
			writer(instr_file, f"{dfg_opc} {dfg_dst} {dfg_src1} {dfg_src2} {dfg_val1} {dfg_val2} {dfg_val3} {dfg_val4} {num_CPU_instr}")
			for dfg_mem_addr in dfg_mem_addrs:
				writer(instr_file, f" {dfg_mem_addr}")
			writer(instr_file, "\n")
			instruction_count[instruction_dict[dfg_opc]] += 1
			instruction_count[SCALAR] += num_CPU_instr
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
			instruction_count[SCALAR] += (num_CPU_instr + 1)
			num_CPU_instr = 0
		last_asm_idx += 1
	last_asm_idx += 1
	return instruction_count[CONFIG], instruction_count[MOVE], instruction_count[MEMORY], instruction_count[ARITHMETIC], instruction_count[SCALAR]

