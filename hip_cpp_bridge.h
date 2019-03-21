#ifndef _HIP_CPP_BRIDGE_H_
#define _HIP_CPP_BRIDGE_H_ 1
#include <stddef.h>
#include <hsa/hsa.h>

#include <amd_hsa_kernel_code.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;

hsa_status_t HSA_API __do_c_hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, char *value, size_t max_value);

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

#ifdef __cplusplus
}
#endif

#endif
