BUILD_DIR=../build
GUESTLIB_DIR=../../nwcc/cl_nw

all: simple rate-test

simple:
	gcc -g -L$(GUESTLIB_DIR) -Wl,-rpath,$(GUESTLIB_DIR) -o $@ $@.c \
		-lguestlib -ldl -Wno-deprecated-declarations

rate-test:
	gcc -g -L$(GUESTLIB_DIR) -Wl,-rpath,$(GUESTLIB_DIR) -o $@ $@.c \
		-lguestlib -ldl -Wno-deprecated-declarations -pthread

swap:
	gcc -g -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o swap.out swap.c \
		-lOpenCL -ldl -Wno-deprecated-declarations
	gcc -g -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o swap-main.out swap-main.c \
		-lOpenCL -ldl -Wno-deprecated-declarations

overload:
	gcc -g -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o hello.out hello.c -lOpenCL -ldl
	gcc -g -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o hello_tf.out hello_tf.c -ltensorflow -ldl
	gcc -g -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -o hello_cuda.out hello_cuda.c -I/usr/local/cuda-9.0/include -lcuda -ldl
	nvcc -c hello_cuda.cu -o helloWorldDriverAPI.ptx
	gcc -I../include -o micro.out micro.c

binary:
	gcc -g -o save_binary.out binary.c -lOpenCL -Wno-deprecated-declarations

clean:
	rm -f *.out simple rate-test
