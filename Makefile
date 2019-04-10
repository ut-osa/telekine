HIP_PATH = /opt/rocm/hip
export HIP_PATH

HIPCC_VERBOSE = 1
export HIPCC_VERBOSE
HIP_PLATFORM = hcc
export HIP_PLATFORM
HIPCC = ./hipcc --amdgpu-target=gfx900
ifeq ($(NW_PATH),)
	NW_PATH := /home/thunt/nightwatch-combined/nwcc
endif


TARGET = hcc

GEN_CFLAGS += -g -fmax-errors=10 `pkg-config --cflags glib-2.0` -pthread \
	-lrt -ldl -D_GNU_SOURCE -Wall -Wno-unused-but-set-variable \
	-Wno-unused-variable -Wno-unused-function -Wno-discarded-qualifiers \
	-Wno-deprecated-declarations -Wno-deprecated-declarations -Wl,-z,defs \
	-D__HIP_PLATFORM_HCC__=1 -Wno-enum-compare

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = ./MatrixTranspose

includes = -I$(PWD)/include -I/opt/rocm/include -I/opt/rocm/hcc/bin/../include \
			  -I/opt/rocm/hcc/bin/../hcc/include -I/opt/rocm/hcc/include \
			  -I/opt/rocm/hip/include/hip/hcc_detail/cuda -I/opt/rocm/hsa/include \
			  -I/opt/rocm/hip/include

clangargs = -D__HIP_PLATFORM_HCC__=1

CXXFLAGS = -g $(CFLAGS) -Wno-ignored-attributes -fPIC $(includes) -Wno-deprecated-declarations \
			  -Wno-unused-command-line-argument
CXX=$(HIPCC)

all: $(EXECUTABLE) worker libguestlib.so guestshim.so manager
.PHONY: all

GUESTLIB_LIBS+=`pkg-config --libs glib-2.0` -fvisibility=hidden
WORKER_LIBS+=`pkg-config --libs glib-2.0` -L/opt/rocm/lib -lhip_hcc -lhsa-runtime64

GENERAL_SOURCES = $(addprefix nw/common/,cmd_channel.c murmur3.c cmd_handler.c \
													endpoint_lib.c socket.c)
WORKER_SOURCES = hip_nw_worker.c \
					  $(addprefix nw/worker/,worker.c) \
					  $(addprefix nw/common/,cmd_channel_shm_worker.c \
					                         cmd_channel_min_worker.c \
													 cmd_channel_socket_worker.c)
GUESTLIB_SOURCES = hip_nw_guestlib.c $(addprefix nw/guestlib/src/,init.c) \
						 $(addprefix nw/common/,cmd_channel_shm.c cmd_channel_min.c \
													   cmd_channel_socket.c)

worker: $(GENERAL_SOURCES) $(WORKER_SOURCES) hip_cpp_bridge.o
	$(CC) -I./nw/worker/include $(includes) $(GEN_CFLAGS) $^ $(WORKER_LIBS) -lstdc++ -o $@

libguestlib.so: $(GENERAL_SOURCES) $(GUESTLIB_SOURCES)
	$(CC) -I./nw/guestlib/include $(includes) -shared -fPIC $(GEN_CFLAGS) $^ $(GUESTLIB_LIBS) -o $@


$(EXECUTABLE): $(OBJECTS) guestshim.so libguestlib.so
	$(HIPCC) -std=c++11 $^ -o $@ -Wl,-rpath=$(PWD)


regen: hip.nw.cpp
	$(NW_PATH)/nwcc $(includes) -X="$(clangargs) -DPWD=\"$(PWD)\"" ./hip.nw.cpp
.PHONY: regen

manager:
	$(MAKE) -C nw/worker && ln -fs ./nw/worker/manager_tcp manager_tcp
.PHONY: manager

guestshim.so: guestshim.o program_state.o code_object_bundle.o libguestlib.so lgm_memcpy.hpp libcrypto.so
	g++ -fPIC -shared $(includes) -o $@ guestshim.o program_state.o code_object_bundle.o \
	   -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -Wl,-rpath=$(PWD) -L$(PWD) -lguestlib -lpthread -lsodium -lcrypto

libcrypto.so: crypto/aes_gcm.cpp crypto/aes_gcm.h
	$(HIPCC) $(includes) -shared -fPIC crypto/aes_gcm.cpp -o $@ -lsodium -ldl

clean:
	rm -rf hip_nw *.o *.so manager_tcp MatrixTranspose
	$(MAKE) -C nw/worker clean
.PHONY: clean
