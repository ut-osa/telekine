#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

uint64_t get_microseconds_since_epoch() {
    struct timeval tv;
    int ret = gettimeofday(&tv, NULL);
    if (ret != 0) {
        return 0;
    }
    uint64_t result = 0;
    result += static_cast<uint64_t>(tv.tv_sec) * 1000000;
    result += static_cast<uint64_t>(tv.tv_usec);
    return result;
}

__global__ void inc_kernel(uint8_t* ptr, int N, int r)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (tid >= N) return;
    for (int i = 0; i < r; i++) {
        ptr[tid]++;
    }
}


int main(int argc, char *argv[])
{
    int deviceID = 0;
    CHECK(hipSetDevice(deviceID));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    size_t Nbytes = 16 * 1024 * 1024;

    uint8_t* A_h = new uint8_t[Nbytes];
    uint8_t* C_h = new uint8_t[Nbytes];

    for (int i = 0; i < Nbytes; i++) {
        A_h[i] = rand() & 255;
    }

    uint8_t* A_d;
    CHECK(hipMalloc(&A_d, Nbytes));
    uint8_t* C_d;
    CHECK(hipMalloc(&C_d, Nbytes));

    uint64_t start_time, elapsed_time;

    for (int r = 0; r < 10; r++) {

        start_time = get_microseconds_since_epoch();
        CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
        CHECK(hipStreamSynchronize(stream));
        elapsed_time = get_microseconds_since_epoch() - start_time;
        fprintf(stderr, "HtD memcpy: %.3f ms\n", static_cast<double>(elapsed_time) / 1000.0);

        start_time = get_microseconds_since_epoch();
        CHECK(hipMemcpyAsync(C_d, A_d, Nbytes, hipMemcpyDeviceToDevice, stream));
        CHECK(hipStreamSynchronize(stream));
        elapsed_time = get_microseconds_since_epoch() - start_time;
        fprintf(stderr, "DtD memcpy: %.3f ms\n", static_cast<double>(elapsed_time) / 1000.0);

        start_time = get_microseconds_since_epoch();
        CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        elapsed_time = get_microseconds_since_epoch() - start_time;
        fprintf(stderr, "DtH memcpy: %.3f ms\n", static_cast<double>(elapsed_time) / 1000.0);

        for (int i = 0; i < Nbytes; i++) {
            if (C_h[i] != A_h[i]) {
                fprintf(stderr, "Wrong value!\n");
                break;
            }
        }

        fprintf(stderr, "=================\n");
    }

    return 0;
}
