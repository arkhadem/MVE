red:=$(shell tput setaf 1)
green:=$(shell tput setaf 2)
reset:=$(shell tput sgr0)

SRC_DIR = src
OBJ_DIR = obj

BENCH_DIR = $(SRC_DIR)/benchmark
FUNC_DIR = $(SRC_DIR)/fake_funcsim
FAKEFUNC_DIR = $(SRC_DIR)/fake_funcsim
LOCAL_OBJ_DIR = $(OBJ_DIR)/local_obj
PHONE_OBJ_DIR = $(OBJ_DIR)/phone_obj
MVE_OBJ_DIR = $(OBJ_DIR)/mve_obj
LIB_SRC_DIR = $(SRC_DIR)/libraries
UNAME_P := $(shell uname -p)
FUNCSIM_LIB = $(OBJ_DIR)/funcsim.so

INIT_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/init.cpp) $(BENCH_DIR)/init.cpp
LOCAL_INIT_OBJS = $(INIT_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
PHONE_INIT_OBJS = $(INIT_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
SCALAR_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/scalar.cpp) $(BENCH_DIR)/scalar_kernels.cpp
LOCAL_SCALAR_OBJS = $(SCALAR_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
PHONE_SCALAR_OBJS = $(SCALAR_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
NEON_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/neon.cpp) $(wildcard $(LIB_SRC_DIR)/*/*/neon_assembly.S) $(BENCH_DIR)/neon_common_functions.cpp $(BENCH_DIR)/neon_kernels.cpp
LOCAL_NEON_OBJS_TMP = $(NEON_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
LOCAL_NEON_OBJS = $(LOCAL_NEON_OBJS_TMP:%.S=$(LOCAL_OBJ_DIR)/%.o)
PHONE_NEON_OBJS_TMP = $(NEON_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
PHONE_NEON_OBJS = $(PHONE_NEON_OBJS_TMP:%.S=$(PHONE_OBJ_DIR)/%.o)
ADRENO_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/adreno.cpp) $(BENCH_DIR)/adreno_runner.cpp $(BENCH_DIR)/adreno_kernels.cpp $(BENCH_DIR)/clutil.cpp
PHONE_ADRENO_OBJS = $(ADRENO_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
MVE_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/mve.cpp) $(BENCH_DIR)/mve_kernels.cpp
LOCAL_MVE_OBJS = $(MVE_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
PHONE_MVE_OBJS = $(MVE_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
RVV_KERNELS := $(wildcard $(LIB_SRC_DIR)/*/*/rvv.cpp) $(BENCH_DIR)/rvv_kernels.cpp
LOCAL_RVV_OBJS = $(RVV_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
PHONE_RVV_OBJS = $(RVV_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
EXE_KERNELS := $(BENCH_DIR)/exe_runner.cpp
LOCAL_EXE_OBJS = $(EXE_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
PHONE_EXE_OBJS = $(EXE_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)
SIM_KERNELS := $(BENCH_DIR)/sim_runner.cpp
LOCAL_SIM_OBJS = $(SIM_KERNELS:%.cpp=$(LOCAL_OBJ_DIR)/%.o)
FUNCSIM_KERNELS := $(wildcard $(FUNC_DIR)/*.cpp)
FAKE_FUNCSIM_KERNELS := $(wildcard $(FAKEFUNC_DIR)/*.cpp)
PHONE_FAKE_FUNCSIM_OBJS = $(FAKE_FUNCSIM_KERNELS:%.cpp=$(PHONE_OBJ_DIR)/%.o)

INC_PARAMS := $(foreach d, $(dir $(wildcard $(BENCH_DIR)/*.hpp)), -I$d)
INC_PARAMS += $(foreach d, $(dir $(wildcard $(LIB_SRC_DIR)/*/*.hpp)), -I$d)
INC_PARAMS += $(foreach d, $(dir $(wildcard $(LIB_SRC_DIR)/*/*/*.hpp)), -I$d)

FUNCSIM_INC_PARAM = -I$(FUNC_DIR)
FAKE_FUNCSIM_INC_PARAM = -I$(FAKEFUNC_DIR)

CFLAGS = ""

BASE_CFLAGS = $(INC_PARAMS) $(CFLAGS) -O2 -fno-tree-vectorize -fno-vectorize -Wall -Wno-unused-variable -Wno-error=unused-command-line-argument -DLOG_USE_COLOR -g3 -std=c++17
LOCAL_CFLAGS := $(FUNCSIM_INC_PARAM) $(BASE_CFLAGS)

ifeq ($(UNAME_P),aarch64)
	LOCAL_CFLAGS += -march=armv8.2-a+fp16+crypto+crc -DNEON
else
	ifneq ($(UNAME_P),x86_64)
        $(error $$UNAME_P not supported: [${UNAME_P}])
    endif
endif
PHONE_CFLAGS = $(FAKE_FUNCSIM_INC_PARAM) $(BASE_CFLAGS) --target=aarch64-none-linux-android23 --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot -march=armv8.2-a+fp16+crypto+crc -DNEON

LOCAL_CPP = clang++
PHONE_CPP = $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++

$(LOCAL_OBJ_DIR)/%.o: %.cpp
	$(info COMPILING SOURCE $(red)$<$(reset) INTO OBJECT $(green)$@$(reset) FOR LOCAL)
	@mkdir -p '$(@D)'
	@$(LOCAL_CPP) -c $(LOCAL_CFLAGS) $< -o $@

$(LOCAL_OBJ_DIR)/%.o: %.S
	$(info COMPILING SOURCE $(red)$<$(reset) INTO OBJECT $(green)$@$(reset) FOR LOCAL)
	@mkdir -p '$(@D)'
	@$(LOCAL_CPP) -c $(LOCAL_CFLAGS) $< -o $@

$(PHONE_OBJ_DIR)/%.o: %.cpp
	$(info COMPILING SOURCE $(red)$<$(reset) INTO OBJECT $(green)$@$(reset) FOR PHONE)
	@mkdir -p '$(@D)'
	@$(PHONE_CPP) -c $(PHONE_CFLAGS) $< -o $@

$(PHONE_OBJ_DIR)/%.o: %.S
	$(info COMPILING SOURCE $(red)$<$(reset) INTO OBJECT $(green)$@$(reset) FOR PHONE)
	@mkdir -p '$(@D)'
	@$(PHONE_CPP) -c $(PHONE_CFLAGS) $< -o $@

$(FUNCSIM_LIB): $(FUNCSIM_KERNELS)
	$(info BUILDING SHARED LIBRARY $(green)funcsim.so$(reset))
	@mkdir -p $(OBJ_DIR)
	@$(LOCAL_CPP) $(LOCAL_CFLAGS) -shared -fPIC $^ -o $@

all: local

local_scalar: $(LOCAL_INIT_OBJS) $(LOCAL_SCALAR_OBJS) $(LOCAL_EXE_OBJS) $(BENCH_DIR)/main.cpp
	$(LOCAL_CPP) $(LOCAL_CFLAGS) -o benchmark_$@ $?

local_neon: $(LOCAL_INIT_OBJS) $(LOCAL_NEON_OBJS) $(LOCAL_EXE_OBJS) $(BENCH_DIR)/main.cpp
	$(LOCAL_CPP) $(LOCAL_CFLAGS) -o benchmark_$@ $?

local_mve: $(LOCAL_INIT_OBJS) $(FUNCSIM_LIB) $(LOCAL_MVE_OBJS) $(LOCAL_SIM_OBJS) $(BENCH_DIR)/main.cpp
	$(LOCAL_CPP) $(LOCAL_CFLAGS) -o benchmark_$@ $? -L$(OBJ_DIR) -l:funcsim.so

local_rvv: $(LOCAL_INIT_OBJS) $(FUNCSIM_LIB) $(LOCAL_RVV_OBJS) $(LOCAL_SIM_OBJS) $(BENCH_DIR)/main.cpp
	$(LOCAL_CPP) $(LOCAL_CFLAGS) -o benchmark_$@ $? -L$(OBJ_DIR) -l:funcsim.so

phone_scalar: $(PHONE_INIT_OBJS) $(PHONE_SCALAR_OBJS) $(PHONE_EXE_OBJS) $(BENCH_DIR)/main.cpp
	$(PHONE_CPP) $(PHONE_CFLAGS) -o benchmark_$@ $?

phone_neon: $(PHONE_INIT_OBJS) $(PHONE_NEON_OBJS) $(PHONE_EXE_OBJS) $(BENCH_DIR)/main.cpp
	$(PHONE_CPP) $(PHONE_CFLAGS) -o benchmark_$@ $?

phone_mve: $(PHONE_INIT_OBJS) $(PHONE_MVE_OBJS) $(PHONE_EXE_OBJS) $(PHONE_FAKE_FUNCSIM_OBJS) $(BENCH_DIR)/main.cpp
	$(PHONE_CPP) $(PHONE_CFLAGS) -o benchmark_$@ $?

phone_rvv: $(PHONE_INIT_OBJS) $(PHONE_RVV_OBJS) $(PHONE_EXE_OBJS) $(PHONE_FAKE_FUNCSIM_OBJS) $(BENCH_DIR)/main.cpp
	$(PHONE_CPP) $(PHONE_CFLAGS) -o benchmark_$@ $?

phone_adreno: $(PHONE_INIT_OBJS) $(PHONE_ADRENO_OBJS) $(BENCH_DIR)/main.cpp
	$(PHONE_CPP) $(PHONE_CFLAGS) -lOpenCL -o benchmark_$@ $?

.PHONY: clean
clean:
	@rm -rf $(OBJ_DIR)
	@rm -rf benchmark_local_scalar
	@rm -rf benchmark_local_neon
	@rm -rf benchmark_local_mve
	@rm -rf benchmark_phone_scalar
	@rm -rf benchmark_phone_neon
	@rm -rf benchmark_phone_adreno

clean_data:
	@rm *_graph.txt