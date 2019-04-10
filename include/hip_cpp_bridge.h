#ifndef _HIP_CPP_BRIDGE_H_
#define _HIP_CPP_BRIDGE_H_ 1
#include <stddef.h>
#include <hsa/hsa.h>
#include <hip/hip_runtime.h>

#include <amd_hsa_kernel_code.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;

hipError_t
__do_c_hipGetDeviceProperties(char* prop, int deviceId);

hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, char *value, size_t max_value);

hsa_status_t HSA_API __do_c_hsa_agent_get_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value,
    size_t max_value);

hsa_status_t HSA_API __do_c_query_host_address(
    uint64_t kernel_object_,
    char *kernel_header_);

int __do_c_load_executable(
      const char *file_buf,
      size_t file_len,
      hsa_executable_t * executable,
      hsa_agent_t * agent);

size_t __do_c_get_agents(
      hsa_agent_t *agents,
      size_t agents_len);

size_t __do_c_get_isas(
      hsa_agent_t agent,
      hsa_isa_t *isas,
      size_t isas_len);

size_t __do_c_get_kerenel_symbols(
      const hsa_executable_t *exec,
      const hsa_agent_t *agent,
      hsa_executable_symbol_t *symbols,
      size_t symbols_len);

hipError_t
__do_c_hipHccModuleLaunchMultiKernel(
      int numKernels, hipFunction_t* f,
      uint32_t* globalWorkSizeX, uint32_t* globalWorkSizeY, uint32_t* globalWorkSizeZ,
      uint32_t* localWorkSizeX, uint32_t* localWorkSizeY, uint32_t* localWorkSizeZ,
      size_t* sharedMemBytes, hipStream_t stream,
      char* all_extra, size_t total_extra_size, size_t* extra_size);

hipError_t
nw_hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hipStream_t stream);

hipError_t
nw_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t
nw_hipCtxGetDevice(hipDevice_t* device);

hipError_t
nw_hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);

hipError_t
nw_hipStreamSynchronize(hipStream_t stream);

hipError_t
nw_hipCtxSetCurrent(hipCtx_t ctx);

hipError_t
nw_hipGetDevice(int* deviceId);

hipError_t
nw_hipSetDevice(int deviceId);

hipError_t
nw_hipStreamCreate(hipStream_t* stream, hsa_agent_t *agent);

hipError_t
nw_hipStreamDestroy(hipStream_t stream);

hipError_t
__do_c_hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                      uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                      uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                      uint32_t localWorkSizeZ, size_t sharedMemBytes,
                      hipStream_t stream, void** kernelParams, char* extra,
                      size_t extra_size, hipEvent_t start, hipEvent_t stop);

hipError_t
__do_c_hipModuleLaunchKernel(hipFunction_t *f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      hipStream_t stream, void** kernelParams, char* extra,
                      size_t extra_size);

hipError_t
__do_c_get_kernel_descriptor(const hsa_executable_symbol_t *symbol, const char *name, hipFunction_t *f);

#ifdef __cplusplus
}
#endif

#endif
