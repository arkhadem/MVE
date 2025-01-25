make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator simulator_bs_ino

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator simulator_bh_ino

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator simulator_bp_ino

make clean
make -j CFLAGS='-DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator simulator_ac_ino

make clean
make -j CFLAGS='-DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator simulator_bs_ooo

make clean
make -j CFLAGS='-DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator simulator_bh_ooo

make clean
make -j CFLAGS='-DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator simulator_bp_ooo

make clean
make -j CFLAGS='-DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator simulator_ac_ooo

make clean
make -j CFLAGS='-DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator simulator_bs_dvi

make clean
make -j CFLAGS='-DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator simulator_bh_dvi

make clean
make -j CFLAGS='-DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator simulator_bp_dvi

make clean
make -j CFLAGS='-DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator simulator_ac_dvi

make clean
make -j CFLAGS='-DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator simulator_bs_ora

make clean
make -j CFLAGS='-DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator simulator_bh_ora

make clean
make -j CFLAGS='-DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
mv ramulator simulator_bp_ora

make clean
make -j CFLAGS='-DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
mv ramulator simulator_ac_ora