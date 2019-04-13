#include "hip_function_info.hpp"

#include <libhsakmt/hsakmttypes.h>
#include <hip/hip_runtime_api.h>

// Internal header, do not percolate upwards.
#include <hip_hcc_internal.h>
#include <hc.hpp>
#include <trace_helper.h>
#include <hip/hip_hcc.h>

#include <unordered_map>
#include <pthread.h>

static pthread_mutex_t stream_agent_lock = PTHREAD_MUTEX_INITIALIZER;
static std::unordered_map<hipStream_t, hsa_agent_t> stream_to_agent;

thread_local int current_device = 0;
thread_local hipCtx_t current_ctx = nullptr;
thread_local hipDevice_t current_ctx_device = -1;


const struct nw_kern_info *hip_function_kernel_info(hipFunction_t f)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    static std::unordered_map<hipFunction_t, struct nw_kern_info> cache;
    const struct nw_kern_info *ret;

    pthread_mutex_lock(&lock);
    auto it0 = cache.find(f);
    if (it0 == cache.end()) {
        struct nw_kern_info *info = &cache[f];
        if (nw_lookup_kern_info(f, info) != hipSuccess)
            assert(0 && "failed to do lookup\n");
        ret = info;
    } else {
        ret = &it0->second;
    }
    pthread_mutex_unlock(&lock);

    return ret;
}

inline
std::string name(uintptr_t function_address)
{
   const auto it = hip_impl::function_names().find(function_address);

   if (it == hip_impl::function_names().cend())  {
       throw std::runtime_error{
           "Invalid function passed to hipLaunchKernelGGL."};
   }

   return it->second;
}

extern "C" hsa_status_t HSA_API
nw_hsa_agent_get_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value)
{
  size_t value_size;
  const size_t attribute_u = static_cast<size_t>(attribute);
  switch (attribute_u) {
    case HSA_AGENT_INFO_NAME:
    case HSA_AGENT_INFO_VENDOR_NAME:
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME:
      value_size = HSA_PUBLIC_NAME_SIZE;
      break;
    case HSA_AGENT_INFO_FEATURE:
      value_size = sizeof(hsa_agent_feature_t);
      break;
    case HSA_AGENT_INFO_MACHINE_MODEL:
      value_size = sizeof(hsa_machine_model_t);
      break;
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
      value_size = sizeof(hsa_default_float_rounding_mode_t);
      break;
    case HSA_AGENT_INFO_FAST_F16_OPERATION:
      value_size = sizeof(bool);
      break;
    case HSA_AGENT_INFO_PROFILE:
      value_size = sizeof(hsa_profile_t);
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM:
      value_size = sizeof(uint16_t[3]);
      break;
    case HSA_AGENT_INFO_GRID_MAX_DIM:
      value_size = sizeof(hsa_dim3_t);
      break;
    case HSA_AGENT_INFO_QUEUE_TYPE:
      value_size = sizeof(hsa_queue_type32_t);
      break;
    case HSA_AGENT_INFO_DEVICE:
      value_size = sizeof(hsa_device_type_t);
      break;
    case HSA_AGENT_INFO_ISA:
      value_size = sizeof(hsa_isa_t);
      break;
    case HSA_AGENT_INFO_EXTENSIONS:
      value_size = sizeof(uint8_t);
      break;
    case HSA_AGENT_INFO_VERSION_MAJOR:
    case HSA_AGENT_INFO_VERSION_MINOR:
      value_size = sizeof(uint16_t);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      std::abort(); /* FIXME */
      break;
    default:
      value_size = sizeof(uint32_t);
      break;
  }
  return __do_c_hsa_agent_get_info(agent, attribute, (char *)value, value_size);
}


inline
std::string name(hsa_agent_t agent)
{
   char n[64] = {};
   nw_hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, n);

   return std::string{n};
}

hipError_t
hipStreamCreate(hipStream_t* stream)
{
   hsa_agent_t agent;

   hipError_t ret = nw_hipStreamCreate(stream, &agent);
   if (!ret) {
      pthread_mutex_lock(&stream_agent_lock);
      stream_to_agent.emplace(*stream, agent);
      pthread_mutex_unlock(&stream_agent_lock);
   }

   return ret;
}

hipError_t
hipStreamDestroy(hipStream_t stream)
{
   hipError_t ret = nw_hipStreamDestroy(stream);
   if (!ret) {
      pthread_mutex_lock(&stream_agent_lock);
      auto it = stream_to_agent.find(stream);
      if (it != stream_to_agent.end())
         stream_to_agent.erase(it);
      pthread_mutex_unlock(&stream_agent_lock);
   }
   return ret;
}

inline
hsa_agent_t target_agent(hipStream_t stream)
{
   static hsa_agent_t agents[MAX_AGENTS];
   static std::once_flag f;

   call_once(f, []() {
      size_t n_agents = __do_c_get_agents(agents, MAX_AGENTS);

      assert(n_agents > 0);
   });

   if (stream) {
      hsa_agent_t agent;
      pthread_mutex_lock(&stream_agent_lock);
      auto it = stream_to_agent.find(stream);
      if (it == stream_to_agent.end()) {
         std::printf("%s:%d no agent recoreded\n", __FILE__, __LINE__);
         std::abort();
      }
      agent = it->second;
      pthread_mutex_unlock(&stream_agent_lock);
      return agent;
   }
   return agents[current_device];
#if 0
   if (stream) {
       return *static_cast<hsa_agent_t*>(
           stream->locked_getAv()->get_hsa_agent());
   }
   else if (
       ihipGetTlsDefaultCtx() && ihipGetTlsDefaultCtx()->getDevice()) {
       return ihipGetDevice(
           ihipGetTlsDefaultCtx()->getDevice()->_deviceId)->_hsaAgent;
   }
   else {
       return *static_cast<hsa_agent_t*>(
           accelerator{}.get_default_view().get_hsa_agent());
   }
#endif
}

hipError_t
hipGetDevice(int* deviceId)
{
   hipError_t ret = hipSuccess;

   static std::once_flag f;
   std::call_once(f, [&ret] () {
      int id;
      ret = nw_hipGetDevice(&id);
      if (!ret)
         current_device = id;
   });

   *deviceId = current_device;
   return ret;
}

hipError_t
hipSetDevice(int deviceId)
{
   hipError_t ret = hipSuccess;

   if (current_device != deviceId) {
      ret = nw_hipSetDevice(deviceId);
      if (!ret)
         current_device = deviceId;
   }
   return ret;
}

extern "C" hipError_t
hipCtxGetDevice(hipDevice_t* device)
{
    if (current_ctx_device == -1) {
        hipError_t status = nw_hipCtxGetDevice(&current_ctx_device);
        if (status != hipSuccess) {
            return status;
        }
    }
    *device = current_ctx_device;
    return hipSuccess;
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx)
{
   hipError_t ret = hipSuccess;
   if (ctx != current_ctx) {
      ret = nw_hipCtxSetCurrent(ctx);
      if (!ret)
         current_ctx = ctx;
   }
   return ret;
}


hipFunction_t hip_function_lookup(uintptr_t function_address,
                                  hipStream_t stream)
{
     const auto it0 = hip_impl::functions().find(function_address);

     if (it0 == hip_impl::functions().cend()) {
         throw std::runtime_error{
             "No device code available for function: " +
             name(function_address)
         };
     }

     auto agent = target_agent(stream);

     const auto it1 = find_if(
         it0->second.cbegin(),
         it0->second.cend(),
         [=](const std::pair<hsa_agent_t, hipFunction_t>& x) {
         return x.first == agent;
     });

     if (it1 == it0->second.cend()) {
         throw std::runtime_error{
             "No code available for function: " + name(function_address) +
             ", for agent: " + name(agent)
         };
     }
     for (auto&& agent_kernel : it0->second) {
         if (agent.handle == agent_kernel.first.handle)
            return agent_kernel.second;
     }
     assert(false && "should be impossible!");
     return nullptr;
}

const char* ihipErrorString(hipError_t hip_error) {
    switch (hip_error) {
        case hipSuccess:
            return "hipSuccess";
        case hipErrorOutOfMemory:
            return "hipErrorOutOfMemory";
        case hipErrorNotInitialized:
            return "hipErrorNotInitialized";
        case hipErrorDeinitialized:
            return "hipErrorDeinitialized";
        case hipErrorProfilerDisabled:
            return "hipErrorProfilerDisabled";
        case hipErrorProfilerNotInitialized:
            return "hipErrorProfilerNotInitialized";
        case hipErrorProfilerAlreadyStarted:
            return "hipErrorProfilerAlreadyStarted";
        case hipErrorProfilerAlreadyStopped:
            return "hipErrorProfilerAlreadyStopped";
        case hipErrorInvalidImage:
            return "hipErrorInvalidImage";
        case hipErrorInvalidContext:
            return "hipErrorInvalidContext";
        case hipErrorContextAlreadyCurrent:
            return "hipErrorContextAlreadyCurrent";
        case hipErrorMapFailed:
            return "hipErrorMapFailed";
        case hipErrorUnmapFailed:
            return "hipErrorUnmapFailed";
        case hipErrorArrayIsMapped:
            return "hipErrorArrayIsMapped";
        case hipErrorAlreadyMapped:
            return "hipErrorAlreadyMapped";
        case hipErrorNoBinaryForGpu:
            return "hipErrorNoBinaryForGpu";
        case hipErrorAlreadyAcquired:
            return "hipErrorAlreadyAcquired";
        case hipErrorNotMapped:
            return "hipErrorNotMapped";
        case hipErrorNotMappedAsArray:
            return "hipErrorNotMappedAsArray";
        case hipErrorNotMappedAsPointer:
            return "hipErrorNotMappedAsPointer";
        case hipErrorECCNotCorrectable:
            return "hipErrorECCNotCorrectable";
        case hipErrorUnsupportedLimit:
            return "hipErrorUnsupportedLimit";
        case hipErrorContextAlreadyInUse:
            return "hipErrorContextAlreadyInUse";
        case hipErrorPeerAccessUnsupported:
            return "hipErrorPeerAccessUnsupported";
        case hipErrorInvalidKernelFile:
            return "hipErrorInvalidKernelFile";
        case hipErrorInvalidGraphicsContext:
            return "hipErrorInvalidGraphicsContext";
        case hipErrorInvalidSource:
            return "hipErrorInvalidSource";
        case hipErrorFileNotFound:
            return "hipErrorFileNotFound";
        case hipErrorSharedObjectSymbolNotFound:
            return "hipErrorSharedObjectSymbolNotFound";
        case hipErrorSharedObjectInitFailed:
            return "hipErrorSharedObjectInitFailed";
        case hipErrorOperatingSystem:
            return "hipErrorOperatingSystem";
        case hipErrorSetOnActiveProcess:
            return "hipErrorSetOnActiveProcess";
        case hipErrorInvalidHandle:
            return "hipErrorInvalidHandle";
        case hipErrorNotFound:
            return "hipErrorNotFound";
        case hipErrorIllegalAddress:
            return "hipErrorIllegalAddress";

        case hipErrorMissingConfiguration:
            return "hipErrorMissingConfiguration";
        case hipErrorMemoryAllocation:
            return "hipErrorMemoryAllocation";
        case hipErrorInitializationError:
            return "hipErrorInitializationError";
        case hipErrorLaunchFailure:
            return "hipErrorLaunchFailure";
        case hipErrorPriorLaunchFailure:
            return "hipErrorPriorLaunchFailure";
        case hipErrorLaunchTimeOut:
            return "hipErrorLaunchTimeOut";
        case hipErrorLaunchOutOfResources:
            return "hipErrorLaunchOutOfResources";
        case hipErrorInvalidDeviceFunction:
            return "hipErrorInvalidDeviceFunction";
        case hipErrorInvalidConfiguration:
            return "hipErrorInvalidConfiguration";
        case hipErrorInvalidDevice:
            return "hipErrorInvalidDevice";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorInvalidDevicePointer:
            return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection:
            return "hipErrorInvalidMemcpyDirection";
        case hipErrorUnknown:
            return "hipErrorUnknown";
        case hipErrorInvalidResourceHandle:
            return "hipErrorInvalidResourceHandle";
        case hipErrorNotReady:
            return "hipErrorNotReady";
        case hipErrorNoDevice:
            return "hipErrorNoDevice";
        case hipErrorPeerAccessAlreadyEnabled:
            return "hipErrorPeerAccessAlreadyEnabled";

        case hipErrorPeerAccessNotEnabled:
            return "hipErrorPeerAccessNotEnabled";
        case hipErrorRuntimeMemory:
            return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther:
            return "hipErrorRuntimeOther";
        case hipErrorHostMemoryAlreadyRegistered:
            return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered:
            return "hipErrorHostMemoryNotRegistered";
        case hipErrorTbd:
            return "hipErrorTbd";
        default:
            return "hipErrorUnknown";
    };
};

extern "C" const char*
hipGetErrorString(hipError_t hip_error) {
    return ihipErrorString(hip_error);
}

