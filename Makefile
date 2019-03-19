includes = -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include \
	        -I/opt/rocm/hcc/include -I/opt/rocm/hip/include/hip/hcc_detail/cuda \
			  -I/opt/rocm/hsa/include -I/opt/rocm/hip/include

clangargs = -D__HIP_PLATFORM_HCC__=1

all: hip_nw/worker guestshim.so hip_cpp_bridge.so

hip_nw/worker: hip_nw/Makefile hip_cpp_bridge.so $(wildcard hip_nw/*.c) $(wildcard hip_nw/*.h)
	$(MAKE) -C hip_nw

hip_nw/Makefile: hip.nw.cpp hip_nw.mk
	../nwcc $(includes) -X="$(clangargs)" ./hip.nw.cpp
	cp hip_nw.mk ./hip_nw/Makefile

guestshim.so: guestshim.cpp
	g++ -fPIC -shared $(includes) -o $@ $< \
	   -Wl,--no-allow-shlib-undefined \
		-Wl,--no-undefined -L$(nw_path) -lguestlib

hip_cpp_bridge.so: hip_cpp_bridge.cpp
	g++ -fPIC -shared $(includes) -o $@ $<

clean:
	rm -rf hip_nw
	rm *.o *.so
