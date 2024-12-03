make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator ramulator_MVE_bs_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator ramulator_MVE_bh_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator ramulator_MVE_bp_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator ramulator_MVE_ac_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=RVV_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator ramulator_RVV_bs_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=RVV_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator ramulator_RVV_bh_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=RVV_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator ramulator_RVV_bp_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=RVV_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator ramulator_RVV_ac_verbose