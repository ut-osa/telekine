/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}


/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void
vector_copy(T *C_d, T *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i];
    }
}


int main(int argc, char *argv[])
{
    uint8_t *A_d, *C_d;
    int deviceID = 0;
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, deviceID));
    printf("info: running on device %s\n", props.name);

    for (size_t N = 1; N < 0xFF0000; N < 0x100 ? N++ : N += 0xFFFF & rand()) {
      size_t Nbytes = N * sizeof(uint8_t);

      printf("info: allocate mem (0x%010zx B)\n", Nbytes);
      uint8_t *A_h = new uint8_t[Nbytes];
      CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
      uint8_t *C_h = new uint8_t[Nbytes];
      CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
      // Fill with i
      for (size_t i = 0; i < N; i++) A_h[i] = i;

      //printf("info: allocate device mem (0x%0zx B)\n", Nbytes);
      CHECK(hipSetDevice(deviceID));
      CHECK(hipMalloc(&A_d, Nbytes));
      CHECK(hipMalloc(&C_d, Nbytes));

      hipEvent_t e1, e2, e3, e4;
      CHECK(hipEventCreate(&e1));
      CHECK(hipEventCreate(&e2));
      CHECK(hipEventCreate(&e3));
      CHECK(hipEventCreate(&e4));

      CHECK(hipEventRecord(e1));
      //printf("info: copy Host2Device\n");
      CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
      CHECK(hipEventRecord(e2));

      const unsigned blocks = 512;
      const unsigned threadsPerBlock = 256;

      //printf("info: launch 'vector_copy' kernel\n");
      hipLaunchKernelGGL((vector_copy), dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);
      CHECK(hipEventRecord(e3));

      //printf("info: copy Device2Host\n");
      CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
      CHECK(hipEventRecord(e4));
      CHECK(hipEventSynchronize(e4));
      //printf("info: check result\n");
      bool error = false;
      for (size_t i = 0; i < N; i++)  {
        if (C_h[i] != A_h[i]) {
          printf("mismatch at index %zu. Expected %x, actual %x\n", i, A_h[i], C_h[i]);
          error = true;
        }
      }
      if (error) CHECK(hipErrorUnknown);

#define MBpSec(bytes, ms) ((bytes * 1000.0) / (ms * 1048576.0))

      float ms;
      CHECK(hipEventElapsedTime(&ms, e1, e2));
      printf("info: copy-in  %8.4f MB/Sec\n", MBpSec(Nbytes, ms));
      CHECK(hipEventElapsedTime(&ms, e2, e3));
      printf("info: kernel   %8.4f MB/Sec\n", MBpSec(Nbytes, ms));
      CHECK(hipEventElapsedTime(&ms, e3, e4));
      printf("info: copy-out %8.4f MB/Sec\n", MBpSec(Nbytes, ms));
      int device_count;
      CHECK(hipGetDeviceCount(&device_count));
      if (device_count > 1) {
        uint8_t *B_d;
        hipEvent_t e5, e6;
        CHECK(hipEventCreate(&e5));
        CHECK(hipEventCreate(&e6));
        CHECK(hipSetDevice(1));
        CHECK(hipMalloc(&B_d, Nbytes));
        CHECK(hipSetDevice(0));
        CHECK(hipEventRecord(e5));
        CHECK(hipMemcpy(B_d, A_d, Nbytes, hipMemcpyDeviceToDevice));
        CHECK(hipEventRecord(e6));
        CHECK(hipEventSynchronize(e6));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e5, e6));
        printf("info: copy-d2d %8.4f MB/Sec\n", MBpSec(Nbytes, ms));
        uint8_t *B_h = new uint8_t[Nbytes];
        CHECK(hipMemcpy(B_h, B_d, Nbytes, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < N; i++)  {
          if (C_h[i] != B_h[i]) {
            printf("mismatch at index %zu. Expected %x, actual %x\n", i, A_h[i], C_h[i]);
            CHECK(hipErrorUnknown);
          }
        }
        delete[] B_h;
      }
      delete[] A_h;
      delete[] C_h;
      CHECK(hipFree(A_d));
      CHECK(hipFree(C_d));
      CHECK(hipEventDestroy(e1));
      CHECK(hipEventDestroy(e2));
      CHECK(hipEventDestroy(e3));
      CHECK(hipEventDestroy(e4));
    }
    printf("PASSED!\n");
}
