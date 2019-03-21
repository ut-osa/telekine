include ../../../include/make.mk

CC=gcc
override CFLAGS+= -g -fmax-errors=10      `pkg-config --cflags glib-2.0`     -pthread -lrt -ldl -D_GNU_SOURCE     -Wall -Wno-unused-but-set-variable     -Wno-unused-variable -Wno-unused-function     -Wno-discarded-qualifiers -Wno-deprecated-declarations     -Wl,-z,defs -D__HIP_PLATFORM_HCC__=1 -Wno-enum-compare

includes = -I../include -I../include/hip -I../include/hip/hcc_dtail \
           -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include \
	        -I/opt/rocm/hcc/include -I/opt/rocm/hip/include/hip/hcc_detail/cuda \
			  -I/opt/rocm/hsa/include -I/opt/rocm/hip/include

guest_includes = -I../guest_inc -I../guest_inc/hip \
					  -I../guest_inc/hip/hcc_detail -I../include \
					  -I../include/hip/hcc_detail -I../include/hsa

GUESTLIB_LIBS+=`pkg-config --libs glib-2.0` -fvisibility=hidden
WORKER_LIBS+=`pkg-config --libs glib-2.0` -L/opt/rocm/lib -lhip_hcc -lhsa-runtime64

all: libguestlib.so worker

GENERAL_SOURCES=$(addprefix ../../../common/,cmd_channel.c murmur3.c cmd_handler.c endpoint_lib.c socket.c)
WORKER_SPECIFIC_SOURCES=hip_nw_worker.c $(addprefix ../../../worker/,worker.c) $(addprefix ../../../common/,cmd_channel_shm_worker.c) $(addprefix ../../../common/,cmd_channel_min_worker.c)
GUESTLIB_SPECIFIC_SOURCES=hip_nw_guestlib.c $(addprefix ../../../guestlib/src/,init.c) $(addprefix ../../../common/,cmd_channel_shm.c) $(addprefix ../../../common/,cmd_channel_min.c)

worker: $(GENERAL_SOURCES) $(WORKER_SPECIFIC_SOURCES)
	$(CC) -I../../../worker/include $(includes) $(CFLAGS) $^ $(WORKER_LIBS) -o $@ ../hip_cpp_bridge.so

libguestlib.so: $(GENERAL_SOURCES) $(GUESTLIB_SPECIFIC_SOURCES)
	$(CC) -I../../../guestlib/include $(guest_includes) -shared -fPIC $(CFLAGS) $^ $(GUESTLIB_LIBS) -o $@

clean:
	-rm -rf worker libguestlib.so *.o

.PHONY: all clean
