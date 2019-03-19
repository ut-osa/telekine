includes = -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include \
	        -I/opt/rocm/hcc/include -I/opt/rocm/hip/include/hip/hcc_detail/cuda \
			  -I/opt/rocm/hsa/include -I/opt/rocm/hip/include

clangargs = -D__HIP_PLATFORM_HCC__=1

all: hip_nw/worker

hip_nw/worker: hip_nw/Makefile $(wildcard hip_nw/*.c) $(wildcard hip_nw/*.h)
	$(MAKE) -C hip_nw

hip_nw/Makefile: hip.nw.cpp hip_nw.mk
	../nwcc $(includes) -X="$(clangargs)" ./hip.nw.cpp
	cp hip_nw.mk ./hip_nw/Makefile

clean:
	rm -rf hip_nw
