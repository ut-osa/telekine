HIP_PATH = /opt/rocm/hip
export HIP_PATH

ifdef $(VERBOSE)
HIPCC_VERBOSE := $(VERBOSE)
export HIPCC_VERBOSE
endif

HIP_PLATFORM = hcc
export HIP_PLATFORM
HIPCC = ./hipcc --amdgpu-target=gfx900
ifeq ($(NW_PATH),)
	NW_PATH := /home/thunt/nightwatch-combined/nwcc
endif

includes = -I$(PWD)/include -I/opt/rocm/include -I/opt/rocm/hcc/bin/../include \
			  -I/opt/rocm/hcc/bin/../hcc/include -I/opt/rocm/hcc/include \
			  -I/opt/rocm/hip/include/hip/hcc_detail/cuda -I/opt/rocm/hsa/include \
			  -I/opt/rocm/hip/include -I./nw/worker/include -I./nw/guestlib/include

FLAGS = -g -O3 -fPIC $(includes) $(shell pkg-config --cflags glib-2.0) -fmax-errors=10 \
         -D_GNU_SOURCE -Wall -Werror -Wno-deprecated-declarations \
         -Wno-unused-variable -Wno-unused-function -Wl,-z,defs \
         -Wno-enum-compare -Wno-strict-aliasing \
         -Wl,--no-allow-shlib-undefined -Wl,--no-undefined

CFLAGS = $(FLAGS) -Wno-discarded-qualifiers -Wno-unused-but-set-variable  -D__HIP_PLATFORM_HCC__=1
CXXFLAGS = $(FLAGS) -std=c++14 -Wno-ignored-attributes -Wno-deprecated-declarations \
			  -Wno-unused-command-line-argument
CXX=$(HIPCC)

NW_CFLAGS += $(CFLAGS) -fvisibility=hidden

LIBS := $(shell pkg-config --libs glib-2.0) -lrt -ldl -lstdc++ -lssl -lcrypto -pthread
libguestlib.so_LIBS := -shared -fvisibility=hidden $(LIBS)
worker_LIBS := -L/opt/rocm/lib -lhip_hcc -lhsa-runtime64 $(LIBS)
worker_reverse_socket_LIBS := $(worker_LIBS)

NW_SRCS := nw/common/cmd_channel.c nw/common/murmur3.c nw/common/cmd_handler.c \
			  nw/common/endpoint_lib.c nw/common/socket.c
worker_SRCS := $(NW_SRCS) hip_nw_worker.o hip_cpp_bridge.o nw/worker/worker.c \
			      nw/common/cmd_channel_shm_worker.c \
					nw/common/cmd_channel_min_worker.c \
			      nw/common/cmd_channel_socket_worker.c
worker_reverse_socket_SRCS := $(filter-out nw/worker/worker.c,$(worker_SRCS)) \
	                           nw/worker/worker_reverse_socket.c
libguestlib.so_SRCS = $(NW_SRCS) hip_nw_guestlib.c nw/guestlib/src/init.c \
						 nw/common/cmd_channel_shm.c nw/common/cmd_channel_min.c \
					    nw/common/cmd_channel_socket.c

NW_BINS = worker worker_reverse_socket libguestlib.so
TESTS = MatrixTranspose copy copy2

all: $(TESTS) $(NW_BINS) guestshim.so manager
.PHONY: all

.SECONDEXPANSION:
$(NW_BINS): %: $$(%_SRCS)
	@echo "  LINK $@"
	@$(CC) -I./nw/worker/include $(NW_CFLAGS) $^ $($*_LIBS) -o $@

$(TESTS): %: %.o guestshim.so libguestlib.so
	@echo "  LINK $@"
	@$(HIPCC) $^ -o $@ -Wl,-rpath=$(PWD)

manager:
	$(MAKE) -C nw/worker && ln -fs ./nw/worker/manager_tcp manager_tcp
.PHONY: manager

guestshim.so: lgm_memcpy.o guestshim.o program_state.o code_object_bundle.o \
              hip_function_info.o lgm_kernels.o aes_gcm.o
	@echo "  LINK $@"
	@$(HIPCC) -fPIC -shared $(includes) -o $@ $^ -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -Wl,-rpath=$(PWD) -L$(PWD) -lguestlib $(LIBS)
guestshim.so: libguestlib.so

clean:
	rm -rf hip_nw *.o .d *.so manager_tcp $(TESTS) $(NW_BINS)
	$(MAKE) -C nw/worker clean
.PHONY: clean

regen: hip.nw.cpp
	$(NW_PATH)/nwcc $(includes) -X="-D__HIP_PLATFORM_HCC__=1 -DPWD=\"$(PWD)\"" ./hip.nw.cpp
.PHONY: regen

include trackdeps.make
