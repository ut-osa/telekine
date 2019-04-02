HIP_PATH = /opt/rocm/hip
export HIP_PATH

HIPCC_VERBOSE = 1
export HIPCC_VERBOSE
HIPCC = ./hipcc

TARGET = hcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = ./MatrixTranspose


includes = -I$(PWD)/include -I/opt/rocm/include -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include \
	        -I/opt/rocm/hcc/include -I/opt/rocm/hip/include/hip/hcc_detail/cuda \
			  -I/opt/rocm/hsa/include -I/opt/rocm/hip/include

HIP_SOs = libhc_am.so libhip_hcc.so libhsa-runtime64.so.1 libhsakmt.so.1

clangargs = -D__HIP_PLATFORM_HCC__=1

CXXFLAGS = -g $(CFLAGS) -fPIC $(includes) -Wno-deprecated-declarations
CXX=$(HIPCC)

all: $(EXECUTABLE) hip_nw/worker guestshim.so hip_cpp_bridge.so

$(EXECUTABLE): $(OBJECTS) guestshim.so hip_nw/libguestlib.so
	$(HIPCC) -std=c++11 $^ -o $@ -Wl,-rpath=$(PWD)
	patchelf $(addprefix --remove-needed ,$(HIP_SOs)) $@


hip_nw/worker hip_nw/libguestlib.so: hip_nw/Makefile hip_cpp_bridge.so $(wildcard hip_nw/*.c) $(wildcard hip_nw/*.h)
	$(MAKE) -C hip_nw

hip_nw/Makefile: hip.nw.cpp hip_nw.mk
	../nwcc $(includes) -X="$(clangargs) -DPWD=\"$(PWD)\"" ./hip.nw.cpp
	cp hip_nw.mk ./hip_nw/Makefile

guestshim.o: guestshim.cpp
	$(CXX) $(CXXFLAGS) -c $^
program_state.o: program_state.cpp
	$(CXX) $(CXXFLAGS) -c $^
code_object_bundle.o: code_object_bundle.cpp
	$(CXX) $(CXXFLAGS) -c $^

guestshim.so: guestshim.o program_state.o code_object_bundle.o hip_nw/libguestlib.so
	g++ -fPIC -shared $(includes) -o $@ guestshim.o program_state.o code_object_bundle.o \
	   -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -Wl,-rpath=$(PWD)/hip_nw -L$(PWD)/hip_nw -lguestlib

hip_cpp_bridge.so: hip_cpp_bridge.o
	g++ -std=c++11 -fPIC -shared $(includes) -o $@ $<

clean:
	rm -rf hip_nw
	rm *.o *.so
