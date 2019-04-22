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

__global__ void inc_kernel(float* ptr, int N, int r, float delta)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (tid >= N) return;
    for (int i = 0; i < r; i++) {
        ptr[tid] += delta;
    }
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s n_round\n", argv[0]);
        exit(1);
    }

    int n_round = atoi(argv[1]);

    int deviceID = 0;
    CHECK(hipSetDevice(deviceID));

    float delta = (float)rand() / RAND_MAX;
    int input_size = 1024 * 1024 * 4;
    size_t n_bytes = input_size * sizeof(float); // 16MB
    float* A_h = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        A_h[i] = (float)rand() / RAND_MAX;
    }
    float* A_d;
    CHECK(hipMalloc(&A_d, n_bytes));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    uint64_t start_time, current_time;
    uint64_t threshold = 1000000; // 1s

    start_time = get_microseconds_since_epoch();
    int last_r = -1;
    int r = 0;
    while (true) {
        CHECK(hipMemcpyAsync(A_d, A_h, n_bytes, hipMemcpyHostToDevice, stream));
        int num_thread = 256;
        int num_block = (input_size - 1) / num_thread + 1;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inc_kernel), num_block, num_thread, 0, stream,
            A_d, input_size, n_round, delta);
        CHECK(hipMemcpyAsync(A_h, A_d, n_bytes, hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        current_time = get_microseconds_since_epoch();
        if (current_time - start_time >= threshold) {
            fprintf(stderr, "%.3f rounds per sec\n",
                    static_cast<float>(r - last_r) / ((current_time - start_time) / 1000000.0));
            start_time = current_time;
            last_r = r;
        }
        r++;
    }

    return 0;
}
