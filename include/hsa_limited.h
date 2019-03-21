#ifndef _HSA_LIMITED_H
#define _HSA_LIMITED_H

#include <amd_hsa_kernel_code.h>


#ifndef HSA_API
#define HSA_API
#endif

hsa_status_t HSA_API hsa_executable_create_alt(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options,
    hsa_executable_t *executable);

hsa_status_t HSA_API hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void *value);

hsa_status_t HSA_API hsa_system_major_extension_supported(
    uint16_t extension,
    uint16_t version_major,
    uint16_t *version_minor,
    bool* result);

hsa_status_t HSA_API hsa_system_get_major_extension_table(
    uint16_t extension,
    uint16_t version_major,
    size_t table_length,
    void *table);

hsa_status_t HSA_API hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_agent_t agent,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data);

hsa_status_t HSA_API hsa_isa_from_name(
    const char *name,
    hsa_isa_t *isa);

#endif
