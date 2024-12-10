python -u scripts/simulator.py --action simulate --directory ./data --scheme bs --isa mve --verbose 2>&1 | tee bs_mve_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme bh --isa mve --verbose 2>&1 | tee bh_mve_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme bp --isa mve --verbose 2>&1 | tee bp_mve_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme ac --isa mve --verbose 2>&1 | tee ac_mve_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme bs --isa rvv --verbose 2>&1 | tee bs_rvv_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme bh --isa rvv --verbose 2>&1 | tee bh_rvv_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme bp --isa rvv --verbose 2>&1 | tee bp_rvv_simulate.txt
python -u scripts/simulator.py --action simulate --directory ./data --scheme ac --isa rvv --verbose 2>&1 | tee ac_rvv_simulate.txt