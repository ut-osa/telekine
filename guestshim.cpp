#include <hsa/hsa.h>

#include <libhsakmt/hsakmttypes.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cstdio>

#include "hip_cpp_bridge.h"
#include "lgm_kernels.hpp"

#include "check_env.h"
#include "./command_scheduler.h"
#include "hip_function_info.hpp"
#include "nw/include/n_ava_channels.h"

#include "command_scheduler.h"

#include <chrono>

using namespace std;

std::map<hipStream_t, std::shared_ptr<CommandScheduler>> CommandScheduler::command_scheduler_map_{};
std::mutex CommandScheduler::command_scheduler_map_mu_{};

hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent)
{
    assert(kernelParams == nullptr);
    assert(extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER);
    assert(extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE);
    assert(extra[4] == HIP_LAUNCH_PARAM_END);

    hsa_kernel_dispatch_packet_t aql = {0};
    uint8_t *extra_buf = (uint8_t *)extra[1];
    size_t extra_size = *(size_t *)extra[3];

    hip_function_to_aql(&aql, f, globalWorkSizeX, globalWorkSizeY,
                        globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
                        localWorkSizeZ, sharedMemBytes);

    return CommandScheduler::GetForStream(hStream)->AddKernelLaunch(&aql,
            extra_buf, extra_size, startEvent, stopEvent);
}

extern "C" hipError_t
hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hipStream_t stream) {
    // fprintf(stderr, "Start hipMemcpyAsync (kind %d, size %d)\n", (int) kind, (int) sizeBytes);
    hipError_t ret = lgm::hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    // fprintf(stderr, "Finish hipMemcpyAsync (kind %d, size %d)\n", (int) kind, (int) sizeBytes);
    return ret;
}

extern "C" hipError_t
hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                   size_t sizeBytes, hipStream_t stream)
{
    return nw_hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);
}

extern "C" hipError_t
hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    // fprintf(stderr, "Start hipMemcpy (kind %d, size %d)\n", (int) kind, (int) sizeBytes);
    hipError_t ret = lgm::hipMemcpy(dst, src, sizeBytes, kind);
    // fprintf(stderr, "Finish hipMemcpy (kind %d, size %d)\n", (int) kind, (int) sizeBytes);
    return ret;
}

extern "C" hipError_t
hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)
{
    static std::map<std::pair<hipDeviceAttribute_t, int>, int> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk{mu};
    if (cache.count(std::make_pair(attr, deviceId)) == 0) {
        int value;
        hipError_t status = nw_hipDeviceGetAttribute(&value, attr, deviceId);
        if (status != hipSuccess) {
            return status;
        }
        cache[std::make_pair(attr, deviceId)] = value;
    }
    *pi = cache[std::make_pair(attr, deviceId)];
    return hipSuccess;
}

extern "C" hipError_t
hipStreamSynchronize(hipStream_t stream)
{
    // fprintf(stderr, "hipStreamSynchronize\n");
    return CommandScheduler::GetForStream(stream)->Wait();
}

hipError_t
hipHostMalloc(void** ptr, size_t size, unsigned int flags)
{
   void *res = malloc(size);
   if (res) {
      *ptr = res;
      return hipSuccess;
   }
   return hipErrorMemoryAllocation;
}

hipError_t
hipHostFree(void* ptr)
{
   free(ptr);
   return hipSuccess;
}

hipError_t
hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
					  size_t width, size_t height, hipMemcpyKind kind,
					  hipStream_t stream)
{
	hipError_t e = hipSuccess;
    if((width == dpitch) && (width == spitch)) {
            e = hipMemcpyAsync(dst, src, width*height, kind, stream);
    } else {
			if(kind != hipMemcpyDeviceToDevice){
				 for (int i = 0; i < height && e; ++i)
					  e = hipMemcpyAsync((unsigned char*)dst + i * dpitch,
											   (unsigned char*)src + i * spitch, width,
												kind, stream);
			} else {
				assert("DeviceToDevice hipMemcpy2DAsync not implemented!" && 0);
			}
    }

    return e;
}

extern "C" hipError_t
hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)
{
    static std::map<int, hipDeviceProp_t*> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk{mu};
    if (cache.count(deviceId) == 0) {
        hipDeviceProp_t* _prop = new hipDeviceProp_t;
        hipError_t status = __do_c_hipGetDeviceProperties((char *)_prop, deviceId);
        if (status != hipSuccess) {
            return status;
        }
        cache[deviceId] = _prop;
    }
    *prop = *cache[deviceId];
    return hipSuccess;
}

extern "C" hipError_t
hipModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX, uint32_t gridDimY,
                      uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                      uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                      void** kernelParams, void** extra)
{
   return hipHccModuleLaunchKernel(f,
               blockDimX * gridDimX, blockDimY * gridDimY, blockDimZ * gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream, kernelParams, extra,
               nullptr, nullptr);
}

extern "C" hsa_status_t HSA_API
nw_hsa_iterate_agents(
      hsa_status_t (*callback)(hsa_agent_t agent, void* data),
      void* data)
{
   hsa_agent_t agents[MAX_AGENTS];
   size_t n_agents = __do_c_get_agents(agents, MAX_AGENTS);
   for (auto agent = agents; agent < agents + n_agents; ++agent) {
      if (callback(*agent, data) != HSA_STATUS_SUCCESS)
         break;
   }
   return HSA_STATUS_SUCCESS;
}

template <uint32_t block_dim, typename RandomAccessIterator, typename N, typename T>
__global__ void hip_fill_n(RandomAccessIterator f, N n, T value) {
    const uint32_t grid_dim = gridDim.x * blockDim.x;

    size_t idx = blockIdx.x * block_dim + threadIdx.x;
    while (idx < n) {
        f[idx] = value;
        idx += grid_dim;
    }
}

template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline const T& clamp_integer(const T& x, const T& lower, const T& upper) {
    assert(!(upper < lower));

    return std::min(upper, std::max(x, lower));
}

template <typename T>
void ihipMemsetKernel(hipStream_t stream, T* ptr, T val, size_t sizeBytes) {
    static constexpr uint32_t block_dim = 256;

    const uint32_t grid_dim = clamp_integer<size_t>(sizeBytes / block_dim, 1, UINT32_MAX);

    hipLaunchKernelGGL(hip_fill_n<block_dim>, dim3(grid_dim), dim3{block_dim}, 0u, stream, ptr,
                       sizeBytes, std::move(val));
}

hipError_t ihipMemset(void* dst, int  value, size_t sizeBytes,
                      hipStream_t stream)
{
    hipError_t e = hipSuccess;

    if ((sizeBytes & 0x3) == 0) {
       // use a faster dword-per-workitem copy:
       try {
           value = value & 0xff;
           uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
           ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t));
       }
       catch (std::exception &ex) {
           e = hipErrorInvalidValue;
       }
    } else {
       // use a slow byte-per-workitem copy:
       try {
           ihipMemsetKernel<char> (stream, static_cast<char*> (dst), value, sizeBytes);
       }
       catch (std::exception &ex) {
           e = hipErrorInvalidValue;
       }
    }
    return e;
};

extern "C"
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream)
{
    return ihipMemset(dst, value, sizeBytes, stream);
}

extern "C"
hipError_t hipMemset(void* dst, int value, size_t sizeBytes)
{
    hipMemsetAsync(dst, value, sizeBytes,
                          CommandScheduler::GetDefStream());
    return hipDeviceSynchronize();
}

extern "C"
hipError_t hipDeviceSynchronize(void)
{
    return hipStreamSynchronize(CommandScheduler::GetDefStream());
}

namespace hip_impl
{
    void hipLaunchKernelGGLImpl(
        uintptr_t function_address,
        const dim3& numBlocks,
        const dim3& dimBlocks,
        uint32_t sharedMemBytes,
        hipStream_t stream,
        void** kernarg)
    {
          hipModuleLaunchKernel(
              hip_function_lookup(function_address, stream),
              numBlocks.x,
              numBlocks.y,
              numBlocks.z,
              dimBlocks.x,
              dimBlocks.y,
              dimBlocks.z,
              sharedMemBytes,
              stream,
              nullptr,
              kernarg);
    }
} // namespace hip_impl
