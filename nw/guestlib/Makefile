include ../include/make.mk

COMMON_C_FILES := src/init.c       \
				  src/memory.c     \
				  src/client.c     \
				  src/receiver.c
COMMON_OBJ_FILES := $(addprefix build/,$(notdir $(COMMON_C_FILES:.c=.o)))

OCL_CPP_FILES := src/cl/platform.c \
				 src/cl/device.c   \
				 src/cl/context.c  \
				 src/cl/kernel.c   \
				 src/cl/queue.c    \
				 src/cl/memory.c   \
				 src/cl/event.c    \
				 src/cl/image.c
OCL_OBJ_FILES := $(addprefix build/cl/,$(notdir $(OCL_CPP_FILES:.c=.o)))
OCL_LIB_SO := build/libOpenCL.so

TF_CPP_FILES := $(wildcard src/tf/*.c)
TF_OBJ_FILES := $(addprefix build/tf/,$(notdir $(TF_CPP_FILES:.c=.o)))
TF_LIB_SO := build/libtensorflow.so

CUDA_INCLUDE := /usr/local/cuda-9.0/include
CUDA_CPP_FILES := $(wildcard src/cuda/*.c)
CUDA_OBJ_FILES := $(addprefix build/cuda/,$(notdir $(CUDA_CPP_FILES:.c=.o)))
CUDA_LIB_SO := build/libcuda.so

MVNC_CPP_FILES := $(wildcard src/mvnc/*.c)
MVNC_OBJ_FILES := $(addprefix build/mvnc/,$(notdir $(MVNC_CPP_FILES:.c=.o)))
MVNC_LIB_SO := build/libmvnc.so

INCLUDE_DIR := -Iinclude -I$(CUDA_INCLUDE)

test_cl:
	gcc -Iinclude -rdynamic -shared \
		-o $(OCL_LIB_SO) $(OCL_CPP_FILES) $(COMMON_C_FILES) \
		../common/cmd_channel_shm.c ../common/cmd_channel.c ../common/socket.c \
		-pthread -DRUNTIME_TYPE=0 -fPIC

all: cl tf cuda mvnc

ifdef OPENCL_SUPPORTED
cl: $(OCL_OBJ_FILES) $(COMMON_OBJ_FILES)
	gcc -Iinclude -rdynamic -shared \
		-o $(OCL_LIB_SO) $(OCL_OBJ_FILES) $(COMMON_OBJ_FILES) \
		-pthread
else
cl:
endif

ifdef TF_C_SUPPORTED
tf: $(TF_OBJ_FILES) build/tf_init.o
	gcc -Iinclude -rdynamic -shared \
		-o $(TF_LIB_SO) $(TF_OBJ_FILES) build/tf_init.o \
		-pthread -ldl
else
tf:
endif

ifdef CUDA_SUPPORTED
cuda: $(CUDA_OBJ_FILES) $(COMMON_OBJ_FILES)
	gcc -rdynamic -shared \
		$(INCLUDE_DIR) \
		-o $(CUDA_LIB_SO) $(CUDA_OBJ_FILES) $(COMMON_OBJ_FILES) \
		-pthread
else
cuda:
endif

ifdef MVNC_SUPPORTED
mvnc: $(MVNC_OBJ_FILES) $(COMMON_OBJ_FILES)
	gcc -Iinclude -rdynamic -shared \
		-o $(MVNC_LIB_SO) $(MVNC_OBJ_FILES) $(COMMON_OBJ_FILES) \
		-pthread
else
mvnc:
endif

build/%.o: src/%.c | build_dir
	gcc -fPIC $(INCLUDE_DIR) -o $@ -c $< -DRUNTIME_TYPE=0

build/tf_init.o: src/init.c | build_dir
	gcc -fPIC $(INCLUDE_DIR) -o $@ -c $< -DUSE_TENSORFLOW_API

build_dir:
	mkdir -p build/cl build/tf build/cuda build/mvnc

clean:
	rm -rf build/*
