HIP_PATH = /opt/rocm/hip
export HIP_PATH

HIPCC_VERBOSE = 1
export HIPCC_VERBOSE
HIP_PLATFORM = hcc
export HIP_PLATFORM
HIPCC = ./hipcc --amdgpu-target=gfx900


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

HIP_SOs = libhc_am.so libhip_hcc.so libhsa-runtime64.so.1 libhsakmt.so.1

clangargs = -D__HIP_PLATFORM_HCC__=1

CXXFLAGS = -g $(CFLAGS) -fPIC $(includes) -Wno-deprecated-declarations \
			  -Wno-unused-command-line-argument
CXX=$(HIPCC)

all: $(EXECUTABLE) worker libguestlib.so guestshim.so crypto_guestshim.so
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
	../nwcc $(includes) -X="$(clangargs) -DPWD=\"$(PWD)\"" ./hip.nw.cpp
.PHONY: regen

guestshim.so: guestshim.o program_state.o code_object_bundle.o libguestlib.so
	g++ -fPIC -shared $(includes) -o $@ guestshim.o program_state.o code_object_bundle.o \
	   -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -Wl,-rpath=$(PWD) -L$(PWD) -lguestlib

crypto_guestshim.so: HIP-encryptedMemcpy/hip_wrapper.o HIP-encryptedMemcpy/crypto/aes_gcm.o \
							guestshim.o program_state.o code_object_bundle.o libguestlib.so
	$(HIPCC) -fPIC -shared $(includes) -o $@ HIP-encryptedMemcpy/hip_wrapper.o \
		HIP-encryptedMemcpy/crypto/aes_gcm.o \
		guestshim.o program_state.o code_object_bundle.o \
	   -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -Wl,-rpath=$(PWD) -L$(PWD) -lguestlib -lsodium -ldl

clean:
	rm -rf hip_nw *.o *.so
.PHONY: clean
