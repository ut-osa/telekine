ava_name("HIP");
ava_version("1.9.0");
ava_identifier(HIP);
ava_number(3);
ava_cflags();
ava_libs(-lcuda);
ava_export_qualifier();

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;
#include "hip_cpp_bridge.h"
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_runtime_api.h>
#include <hsa/hsa.h>

typedef struct {
    /* argument types */
    int func_argc;
    char func_arg_is_handle[64];
} Metadata;

ava_register_metadata(Metadata);

#if 0
hipError_t
hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)
{
    ava_argument(prop) {
        ava_out; ava_buffer(1);
    }
}
#endif

hipError_t
hipMalloc(void **dptr,
          size_t size)
{
    ava_argument(dptr) {
        ava_out; ava_buffer(1); ava_element {ava_opaque;}
    }
}

hipError_t
hipFree(void* ptr)
{
    ava_argument(ptr) {ava_opaque;}
}

hipError_t
hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes)
{
   ava_argument(src) ava_opaque;
   ava_argument(dst) {
         ava_out;
         ava_buffer(sizeBytes);
   }
}

hipError_t
hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes)
{
   ava_argument(src) {
         ava_in;
         ava_buffer(sizeBytes);
   }
   ava_argument(dst) ava_opaque;
}

hipError_t
hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
{

   ava_argument(dst) {
      ava_depends_on(kind);
      if (kind == hipMemcpyDeviceToHost) {
         ava_out;
         ava_buffer(sizeBytes);
      } else {
         ava_opaque;
      }
   }
   ava_argument(src) {
      ava_depends_on(kind);
      if (kind == hipMemcpyHostToDevice) {
         ava_in;
         ava_buffer(sizeBytes);
      } else {
         ava_opaque;
      }
   }
}

ava_utility size_t hipLaunchKernel_extra_size(void **extra) {
    size_t size = 1;
    while (extra[size - 1] != HIP_LAUNCH_PARAM_END)
        size++;
    return size;
}

hipError_t
__do_c_hipModuleLaunchKernel(hipFunction_t *f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      hipStream_t stream, void** kernelParams, char* extra,
                      size_t extra_size)
{
   ava_argument(f) {
      ava_in; ava_buffer(1);
      ava_element {
         ava_handle;
      }
   }
    ava_argument(kernelParams) {
        ava_in; ava_buffer(1);
        ava_element {
           ava_opaque;
        }
    }
    ava_argument(extra) {
        ava_in; ava_buffer(extra_size);
    }
}

hsa_status_t
HSA_API hsa_system_major_extension_supported(
      uint16_t extension,
      uint16_t version_major,
      uint16_t *version_minor,
      bool *result)
{
   ava_argument(result) {
      ava_out; ava_buffer(1);
   }
   ava_argument(version_minor) {
      ava_in; ava_out; ava_buffer(1);
   }
}

hsa_status_t
HSA_API hsa_executable_create_alt(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options,
    hsa_executable_t *executable)
{
   ava_argument(options) {
      ava_in; ava_buffer(strlen(options) + 1);
   }
   ava_argument(executable) {
      ava_out; ava_buffer(1);
   }
}

hsa_status_t
HSA_API hsa_isa_from_name(
    const char *name,
    hsa_isa_t *isa)
{
   ava_argument(name) {
      ava_in; ava_buffer(strlen(name) + 1);
   }
   ava_argument(isa) {
      ava_out; ava_buffer(1);
   }
}

#include <stdint.h>

hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, char *value, size_t max_value)
{
   ava_argument(value) {
      ava_depends_on(max_value);
      ava_out; ava_buffer(max_value);
   }
}

hsa_status_t HSA_API __do_c_hsa_agent_get_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value,
    size_t max_value)
{
   ava_argument(value) {
      ava_depends_on(max_value);
      ava_out; ava_buffer(max_value);
   }
}

int
__do_c_load_executable(const char *file_buf, size_t file_len,
                        hsa_executable_t *executable, hsa_agent_t *agent)
{
   ava_argument(file_buf) {
      ava_in; ava_buffer(file_len);
   }
   ava_argument(executable) {
      ava_in; ava_out; ava_buffer(1);
   }
   ava_argument(agent) {
      ava_in; ava_buffer(1);
   }
}

size_t
__do_c_get_agents(hsa_agent_t *agents, size_t max_agents)
{
   ava_argument(agents) {
      ava_out; ava_buffer(max_agents);
   }
}

size_t
__do_c_get_isas(hsa_agent_t agents, hsa_isa_t *isas, size_t max_isas)
{
   ava_argument(isas) {
      ava_out; ava_buffer(max_isas);
   }
}

size_t
__do_c_get_kerenel_symbols(
      const hsa_executable_t *exec,
      const hsa_agent_t *agent,
      hsa_executable_symbol_t *symbols,
      size_t max_symbols)
{
   ava_argument(exec) {
      ava_in; ava_buffer(1);
   }
   ava_argument(agent) {
      ava_in; ava_buffer(1);
   }
   ava_argument(symbols) {
      ava_out; ava_buffer(max_symbols);
   }
}

hsa_status_t
HSA_API __do_c_query_host_address(
    uint64_t kernel_object_,
    char *kernel_header_)
{
   ava_argument(kernel_object_) ava_opaque;
   ava_argument(kernel_header_) {
      ava_out; ava_buffer(sizeof(amd_kernel_code_t));
   }
}

hipError_t
__do_c_get_kernel_descriptor(const hsa_executable_symbol_t *symbol,
                             const char *name, hipFunction_t *f)
{
   ava_argument(symbol) {
      ava_in; ava_buffer(1);
   }
   ava_argument(name) {
      ava_in; ava_buffer(strlen(name) + 1);
   }
   ava_argument(f) {
      ava_out; ava_buffer(1);
      ava_element {
         ava_handle;
      }
   }
}

hsa_status_t
HSA_API hsa_init()
{
}
