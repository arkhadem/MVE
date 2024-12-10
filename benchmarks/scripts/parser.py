# from msilib.schema import Class
from ast import Pass
import sys
import os
from os.path import exists

FREQ = 2.8 * 1024 * 1024 * 1024
MEGA = 1000000
GIGA = 1000000000

def parse(ram_addr):
	if (exists(ram_addr) == False):
		print(f"[PARSER] Error in file: {ram_addr}")
		return 0, 0, 0, 0, 0, 0, 0

	ram_file = open(ram_addr, "r")
	lines = ram_file.readlines()

	idle_time = 0.00
	CB_NUM0 = 0
	compute_time = 0.00
	CB_NUM1 = 0
	data_access_time = 0.00
	CB_NUM2 = 0
	data_access_energy = 0.00
	compute_energy = 0.00

	for line in lines:
		line_split = line.split()
		if ("ramulator" not in line_split[0]):
			continue
		name = line_split[0].split('.')[1]

		value = float(line_split[1])

		if ("L2_MVE_host_device_cycles" in name):
			idle_time += value
			CB_NUM0 += 1
		elif ("L2_MVE_compute_cycles" in name):
			compute_time += value
			CB_NUM1 += 1
		elif ("L2_MVE_memory_cycles" in name):
			data_access_time += value
			CB_NUM2 += 1
		elif ("L1_cache_access_energy" in name):
			data_access_energy += value
		elif ("L2_cache_access_energy" in name):
			data_access_energy += value
		elif ("L3_cache_access_energy" in name):
			data_access_energy += value
		elif ("L2_MVE_compute_total_energy" in name):
			compute_energy = value

	assert CB_NUM1 == CB_NUM2
	assert CB_NUM1 == CB_NUM0

	if CB_NUM0 == 0:
		print(f"[PARSER] Error in file: {ram_addr}")
		return 0, 0, 0, 0, 0, 0, 0

	idle_time = idle_time * MEGA / float(CB_NUM1) / float(FREQ)
	compute_time = compute_time * MEGA / float(CB_NUM1) / float(FREQ)
	data_access_time = data_access_time * MEGA / float(CB_NUM1) / float(FREQ)
	total_time = idle_time + compute_time + data_access_time

	compute_energy = compute_energy / float(GIGA)
	data_access_energy = data_access_energy / float(GIGA)
	total_energy = compute_energy + data_access_energy

	if total_time == 0:
		print(f"[PARSER] Error in file: {ram_addr}")
		return 0, 0, 0, 0, 0, 0, 0
	
	return idle_time, compute_time, data_access_time, total_time, compute_energy, data_access_energy, total_energy