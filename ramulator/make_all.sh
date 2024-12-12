make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator ramulator_bs

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator ramulator_bh

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator ramulator_bp

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator ramulator_ac