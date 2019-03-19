includes = -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include \
	        -I/opt/rocm/hcc/include -I/opt/rocm/hip/include/hip/hcc_detail/cuda \
			  -I/opt/rocm/hsa/include -I/opt/rocm/hip/include

clangargs = -D__HIP_PLATFORM_HCC__=1

all:
	../nwcc $(includes) -X="$(clangargs)" ./hip.nw.cpp
	cp hip_nw.mk ./hip_nw/Makefile
