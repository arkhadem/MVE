python -u scripts/simulator.py --action parse --directory ./data --scheme bs --isa mve --output mve_bs.csv --verbose 2>&1 | tee bs_mve_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme bh --isa mve --output mve_bh.csv --verbose 2>&1 | tee bh_mve_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme bp --isa mve --output mve_bp.csv --verbose 2>&1 | tee bp_mve_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme ac --isa mve --output mve_ac.csv --verbose 2>&1 | tee ac_mve_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme bs --isa rvv --output rvv_bs.csv --verbose 2>&1 | tee bs_rvv_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme bh --isa rvv --output rvv_bh.csv --verbose 2>&1 | tee bh_rvv_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme bp --isa rvv --output rvv_bp.csv --verbose 2>&1 | tee bp_rvv_parse.txt
python -u scripts/simulator.py --action parse --directory ./data --scheme ac --isa rvv --output rvv_ac.csv --verbose 2>&1 | tee ac_rvv_parse.txt