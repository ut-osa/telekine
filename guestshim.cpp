#include <string>
#include <hsa/hsa.h>

#include <stdio.h>
#include <stdlib.h>

#include "hip_cpp_bridge.h"



#include <libhsakmt/hsakmttypes.h>
#include "hip/hip_runtime_api.h"

// Internal header, do not percolate upwards.
#include <hip/hip_hcc_internal.h>
#include "./hc.hpp"
#include "./trace_helper.h"
#include <hip/hip_hcc.h>

//#include <program_state.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

#include <iostream>

using namespace std;
using namespace hc;

#define MAX_AGENTS 16

static unordered_map<hipStream_t, hsa_agent_t> stream_to_agent;

hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent)
{
   size_t extra_size = *(size_t *)extra[3];
   assert(extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER);
   assert(extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE);
   assert(extra[4] == HIP_LAUNCH_PARAM_END);

   return __do_c_hipHccModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY,
                                       globalWorkSizeZ, localWorkSizeX,
                                       localWorkSizeY, localWorkSizeZ,
                                       sharedMemBytes, hStream, kernelParams,
                                       (char *)(extra[1]), extra_size,
                                       startEvent, stopEvent);
}

hipError_t
hipStreamCreate(hipStream_t* stream)
{
   hsa_agent_t agent;

   hipError_t ret = nw_hipStreamCreate(stream, &agent);
   if (!ret) {
      stream_to_agent.emplace(*stream, agent);
      cout << "stream to agent added!\n";
   }

   return ret;
}

hipError_t
hipStreamDestroy(hipStream_t stream)
{
   hipError_t ret = nw_hipStreamDestroy(stream);
   if (!ret) {
      auto it = stream_to_agent.find(stream);
      if (it != stream_to_agent.end())
         stream_to_agent.erase(it);
   }
   return ret;
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


extern "C" hipError_t
hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)
{
   return __do_c_hipGetDeviceProperties((char *)prop, deviceId);
}

extern "C" hipError_t
hipModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX, uint32_t gridDimY,
                      uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                      uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                      void** kernelParams, void** extra)
{
   size_t extra_size = *(size_t *)extra[3];
   assert(extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER);
   assert(extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE);
   assert(extra[4] == HIP_LAUNCH_PARAM_END);

   return __do_c_hipModuleLaunchKernel(&f, gridDimX, gridDimY, gridDimZ,
                                       blockDimX, blockDimY, blockDimZ,
                                       sharedMemBytes, hStream, kernelParams,
                                       (char *)(extra[1]), extra_size);
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

namespace hip_impl
{
    namespace
    {
        inline
        string name(uintptr_t function_address)
        {
            const auto it = function_names().find(function_address);

            if (it == function_names().cend())  {
                throw runtime_error{
                    "Invalid function passed to hipLaunchKernelGGL."};
            }

            return it->second;
        }

        inline
        string name(hsa_agent_t agent)
        {
            char n[64] = {};
            nw_hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, n);

            return string{n};
        }

        inline
        hsa_agent_t target_agent(hipStream_t stream)
        {
            static hsa_agent_t default_agent;
            static once_flag f;

            call_once(f, []() {
               hsa_agent_t agents[MAX_AGENTS];
               size_t n_agents = __do_c_get_agents(agents, MAX_AGENTS);

               assert(n_agents > 0);
               default_agent = agents[0];
            });

            if (stream) {
               auto it = stream_to_agent.find(stream);
               if (it == stream_to_agent.end()) {
                  std::printf("%s:%d no agent recoreded\n", __FILE__, __LINE__);
                  std::abort();
               } else {
                  return it->second;
               }
            }
            return default_agent;
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
    }

    void hipLaunchKernelGGLImpl(
        uintptr_t function_address,
        const dim3& numBlocks,
        const dim3& dimBlocks,
        uint32_t sharedMemBytes,
        hipStream_t stream,
        void** kernarg)
    {
        const auto it0 = functions().find(function_address);

        if (it0 == functions().cend()) {
            throw runtime_error{
                "No device code available for function: " +
                name(function_address)
            };
        }

        auto agent = target_agent(stream);

        const auto it1 = find_if(
            it0->second.cbegin(),
            it0->second.cend(),
            [=](const pair<hsa_agent_t, hipFunction_t>& x) {
            return x.first == agent;
        });

        if (it1 == it0->second.cend()) {
            throw runtime_error{
                "No code available for function: " + name(function_address) +
                ", for agent: " + name(agent)
            };
        }
        for (auto&& agent_kernel : it0->second) {
            if (agent.handle == agent_kernel.first.handle) {
                hipModuleLaunchKernel(
                    agent_kernel.second,
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
        }
    }
} // namespace hip_impl
