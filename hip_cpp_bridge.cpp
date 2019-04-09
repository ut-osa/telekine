#include "hip/hcc_detail/code_object_bundle.hpp"
#include "hip_cpp_bridge.h"


#include "hip_hcc_internal.h"
#include "trace_helper.h"

#include <string>
#include <stdio.h>
#include <stdlib.h>

using std::string;
struct ihipModuleSymbol_t{
   uint64_t _object;             // The kernel object.
   uint32_t _groupSegmentSize;
   uint32_t _privateSegmentSize;
   string   _name;       // TODO - review for performance cost.  Name is just used for debug.
};


namespace hip_impl {
inline std::uint64_t kernel_object(hsa_executable_symbol_t x) {
    std::uint64_t r = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r);

    return r;
}
}

extern "C" hipError_t
nw_hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hipStream_t stream)
{

   return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

extern "C" hipError_t
nw_hipCtxGetDevice(hipDevice_t* device)
{
   return hipCtxGetDevice(device);
}

extern "C" hipError_t
nw_hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)
{
   return hipDeviceGetAttribute(pi, attr, deviceId);
}

extern "C" hipError_t
nw_hipStreamSynchronize(hipStream_t stream)
{
   return hipStreamSynchronize(stream);
}


extern "C" hipError_t
nw_hipCtxSetCurrent(hipCtx_t ctx)
{
   return hipCtxSetCurrent(ctx);
}

extern "C" hipError_t
nw_hipGetDevice(int *deviceId)
{
   return hipGetDevice(deviceId);
}

extern "C" hipError_t
nw_hipSetDevice(int deviceId)
{
   return hipSetDevice(deviceId);
}

extern "C" hipError_t
nw_hipStreamCreate(hipStream_t* stream, hsa_agent_t *agent)
{
   hipError_t ret = hipStreamCreate(stream);
   if (!ret) {
      *agent = *static_cast<hsa_agent_t*>(
            (*stream)->locked_getAv()->get_hsa_agent());
   }
   return ret;
}

extern "C" hipError_t
nw_hipStreamDestroy(hipStream_t stream)
{
   return hipStreamDestroy(stream);
}

extern "C" hipError_t
__do_c_hipGetDeviceProperties(char* prop, int deviceId)
{
   return hipGetDeviceProperties((hipDeviceProp_t *)prop, deviceId);
}

hip_impl::Kernel_descriptor::Kernel_descriptor(std::uint64_t kernel_object,
                                              const std::string& name)
        : kernel_object_{kernel_object}, name_{name}
{
  bool supported{false};
  std::uint16_t min_v{UINT16_MAX};
  auto r = hsa_system_major_extension_supported(
      HSA_EXTENSION_AMD_LOADER, 1, &min_v, &supported);

  if (r != HSA_STATUS_SUCCESS || !supported) return;

  hsa_ven_amd_loader_1_01_pfn_t tbl{};

  r = hsa_system_get_major_extension_table(
      HSA_EXTENSION_AMD_LOADER,
      1,
      sizeof(tbl),
      reinterpret_cast<void*>(&tbl));

  if (r != HSA_STATUS_SUCCESS) return;
  if (!tbl.hsa_ven_amd_loader_query_host_address) return;

  r = tbl.hsa_ven_amd_loader_query_host_address(
      reinterpret_cast<void*>(kernel_object_),
      reinterpret_cast<const void**>(&kernel_header_));

  if (r != HSA_STATUS_SUCCESS) return;
}

extern "C"
hsa_status_t HSA_API __do_c_query_host_address(
    uint64_t kernel_object_,
    char *kernel_header_)
{
        amd_kernel_code_t const* kernel_ptr = nullptr;
        hsa_ven_amd_loader_1_01_pfn_t tbl{};

        auto r = hsa_system_get_major_extension_table(
            HSA_EXTENSION_AMD_LOADER,
            1,
            sizeof(tbl),
            reinterpret_cast<void*>(&tbl));

        if (r != HSA_STATUS_SUCCESS) return r;
        if (!tbl.hsa_ven_amd_loader_query_host_address) return r;

        r = tbl.hsa_ven_amd_loader_query_host_address(
            reinterpret_cast<void*>(kernel_object_),
            reinterpret_cast<const void**>(&kernel_ptr));
        if (kernel_ptr)
           memcpy(kernel_header_, kernel_ptr, sizeof(*kernel_header_));
        return r;
}

extern "C" hipError_t
__do_c_get_kernel_descriptor(const hsa_executable_symbol_t *symbol,
                             const char *name, hipFunction_t *f)
{
   auto descriptor = new hip_impl::Kernel_descriptor(hip_impl::kernel_object(*symbol), std::string(name));
   *f = reinterpret_cast<hipFunction_t>(descriptor);
   return hipSuccess;
}


extern "C"
hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, char *value, size_t max_value)
{
   return hsa_executable_symbol_get_info(executable_symbol,
                                         attribute, (void *)value);
}

extern "C"
hsa_status_t __do_c_hsa_agent_get_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value,
    size_t max_value)
{

   return hsa_agent_get_info(agent, attribute, value);
}

extern "C" int
__do_c_load_executable(
      const char *file_buf,
      size_t file_len,
      hsa_executable_t * executable,
      hsa_agent_t * agent)
{
   *executable = hip_impl::load_executable(string(file_buf, file_len),
            *executable, *agent);
   return 0;
}

extern "C" size_t
__do_c_get_agents(
      hsa_agent_t *agents,
      size_t agents_len)
{
   static const auto accelerators = hc::accelerator::get_all();
   size_t cur = 0;

   for (auto&& acc : accelerators) {
      auto agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

      if (!agent || !acc.is_hsa_accelerator())
         continue;
      if (cur < agents_len)
         agents[cur] = *agent;
      cur += 1;
   }
   assert(cur <= agents_len);
   return cur;
}

extern "C" size_t
__do_c_get_isas(
      hsa_agent_t agent,
      hsa_isa_t *isas,
      size_t isas_len)
{
   struct isa_arg {
      hsa_isa_t *isas;
      size_t cur;
      size_t isas_len;
   } arg = {isas, 0, isas_len};

   hsa_agent_iterate_isas(agent,
                          [](hsa_isa_t x, void* __arg) {
                             auto _arg = static_cast<isa_arg*>(__arg);
                             if (_arg->cur < _arg->isas_len)
                                _arg->isas[_arg->cur] = x;
                             _arg->cur++;
                             return HSA_STATUS_SUCCESS;
                          },
                          &arg);
   assert(arg.cur <= isas_len);
   return arg.cur;
}

extern "C" size_t
__do_c_get_kerenel_symbols(
      const hsa_executable_t *exec,
      const hsa_agent_t *agent,
      hsa_executable_symbol_t *symbols,
      size_t symbols_len)
{
   struct symbol_arg {
      hsa_executable_symbol_t *symbols;
      size_t cur;
      size_t symbols_len;
   } arg = {symbols, 0, symbols_len};

   auto copy_kernels = [](hsa_executable_t, hsa_agent_t,
                          hsa_executable_symbol_t s, void *__arg) {
      auto _arg = static_cast<symbol_arg*>(__arg);

      if (_arg->cur < _arg->symbols_len)
         _arg->symbols[_arg->cur] = s;
      _arg->cur++;
      return HSA_STATUS_SUCCESS;
   };
   hsa_executable_iterate_agent_symbols(*exec, *agent, copy_kernels, &arg);
   assert(arg.cur <= symbols_len);
   return arg.cur;
}

extern "C" hipError_t
__do_c_hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                      uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                      uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                      uint32_t localWorkSizeZ, size_t sharedMemBytes,
                      hipStream_t stream, void** kernelParams, char* _extra,
                      size_t extra_size, hipEvent_t start, hipEvent_t stop)
{
   void* new_extra[5] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, _extra,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &extra_size,
      HIP_LAUNCH_PARAM_END};

   assert(kernelParams == nullptr);

   return hipHccModuleLaunchKernel(f, globalWorkSizeX,
                                globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
                                localWorkSizeY, localWorkSizeZ, sharedMemBytes, stream,
                                kernelParams, new_extra, start, stop);
}

extern "C" hipError_t
__do_c_hipHccModuleLaunchMultiKernel(
      int numKernels, hipFunction_t* f,
      uint32_t* globalWorkSizeX, uint32_t* globalWorkSizeY, uint32_t* globalWorkSizeZ,
      uint32_t* localWorkSizeX, uint32_t* localWorkSizeY, uint32_t* localWorkSizeZ,
      size_t* sharedMemBytes, hipStream_t stream,
      char* all_extra, size_t total_extra_size, size_t* extra_size) {
   // if (numKernels == 1) {
   //    fprintf(stderr, "Launching a batch of size 1\n");
   // }
   char* extra = all_extra;
   for (int i = 0; i < numKernels; i++) {
      void* new_extra[5] = {
         HIP_LAUNCH_PARAM_BUFFER_POINTER, extra,
         HIP_LAUNCH_PARAM_BUFFER_SIZE, &extra_size[i],
         HIP_LAUNCH_PARAM_END};
      hipError_t status = hipHccModuleLaunchKernel(
         f[i], globalWorkSizeX[i], globalWorkSizeY[i], globalWorkSizeZ[i],
         localWorkSizeX[i], localWorkSizeY[i], localWorkSizeZ[i], sharedMemBytes[i],
         stream, nullptr, new_extra, nullptr, nullptr);
      if (status != hipSuccess) {
         return status;
      }
      extra += extra_size[i];
   }
   return hipSuccess;
}

extern "C" hipError_t
__do_c_hipModuleLaunchKernel(hipFunction_t *f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      hipStream_t stream, void** kernelParams, char *_extra,
                      size_t extra_size)
{
   void* new_extra[5] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, _extra,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &extra_size,
      HIP_LAUNCH_PARAM_END};

   assert(kernelParams == nullptr);

   return hipModuleLaunchKernel(*f, gridDimX,
                                gridDimY, gridDimZ, blockDimX,
                                blockDimY, blockDimZ, sharedMemBytes, stream,
                                kernelParams, new_extra);
}

// hipError_t
// __do_c_hipHccModuleLaunchMultiKernel(
//       int numKernels, hipFunction_t* f,
//       uint32_t* globalWorkSizeX, uint32_t* globalWorkSizeY, uint32_t* globalWorkSizeZ,
//       uint32_t* localWorkSizeX, uint32_t* localWorkSizeY, uint32_t* localWorkSizeZ,
//       size_t* sharedMemBytes, hipStream_t stream,
//       char* all_extra, size_t total_extra_size, size_t* extra_size)
// {
//    assert(0 && "TODO");
//    return hipSuccess;
// }

extern "C"
hsa_status_t HSA_API nw_hsa_executable_create_alt(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options,
    hsa_executable_t *executable)
{
   return hsa_executable_create_alt(profile, default_float_rounding_mode,
                                    options, executable);
}

extern "C"
hsa_status_t HSA_API nw_hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void *value)
{
   return hsa_executable_symbol_get_info(executable_symbol, attribute, value);
}

extern "C"
hsa_status_t HSA_API nw_hsa_system_major_extension_supported(
    uint16_t extension,
    uint16_t version_major,
    uint16_t *version_minor,
    bool* result)
{
   return hsa_system_major_extension_supported(extension, version_major,
                                               version_minor, result);
}

extern "C"
hsa_status_t HSA_API nw_hsa_system_get_major_extension_table(
    uint16_t extension,
    uint16_t version_major,
    size_t table_length,
    void *table)
{
   return hsa_system_get_major_extension_table(extension, version_major,
                                               table_length, table);
}

extern "C"
hsa_status_t HSA_API nw_hsa_isa_from_name(
    const char *name,
    hsa_isa_t *isa)
{
   return hsa_isa_from_name(name, isa);
}
