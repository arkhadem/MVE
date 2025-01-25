# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
# mv ramulator simulator_bs_ino_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
# mv ramulator simulator_bh_ino_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
# mv ramulator simulator_bp_ino_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=INORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
# mv ramulator simulator_ac_ino_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
# mv ramulator simulator_bs_ooo_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
# mv ramulator simulator_bh_ooo_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
# mv ramulator simulator_bp_ooo_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=OUTORDER_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
# mv ramulator simulator_ac_ooo_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
mv ramulator simulator_bs_dvi_verbose

make clean
make -j CFLAGS='-DDEBUG -DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
mv ramulator simulator_bh_dvi_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
# mv ramulator simulator_bp_dvi_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=DVI_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
# mv ramulator simulator_ac_dvi_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=1024 -DLANES_PER_SA=256 -DLATENCY_FILE_NAME=\"bs_intrinsics_latency\"'
# mv ramulator simulator_bs_ora_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"bh_intrinsics_latency\"'
# mv ramulator simulator_bh_ora_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=32 -DLANES_PER_SA=8 -DLATENCY_FILE_NAME=\"bp_intrinsics_latency\"'
# mv ramulator simulator_bp_ora_verbose

# make clean
# make -j CFLAGS='-DDEBUG -DEXE_TYPE=ORACLE_EXE -DISA_TYPE=MVE_ISA -DLANES_PER_CB=256 -DLANES_PER_SA=64 -DLATENCY_FILE_NAME=\"ac_intrinsics_latency\"'
# mv ramulator simulator_ac_ora_verbose