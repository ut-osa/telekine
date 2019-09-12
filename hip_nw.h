#ifndef __HIP_NW_H__
#define __HIP_NW_H__

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <glib.h>

#include "common/cmd_channel.h"

#include "hip_nw_types.h"

#define HIP_API 3

enum hip_functions {
    CALL_HIP_HIP_DEVICE_SYNCHRONIZE, RET_HIP_HIP_DEVICE_SYNCHRONIZE, CALL_HIP_HIP_MALLOC, RET_HIP_HIP_MALLOC,
        CALL_HIP_HIP_FREE, RET_HIP_HIP_FREE, CALL_HIP_HIP_MEMCPY_HTO_D, RET_HIP_HIP_MEMCPY_HTO_D,
        CALL_HIP_HIP_MEMCPY_DTO_H, RET_HIP_HIP_MEMCPY_DTO_H, CALL_HIP_HIP_MEMCPY_DTO_D, RET_HIP_HIP_MEMCPY_DTO_D,
        CALL_HIP_NW_HIP_MEMCPY, RET_HIP_NW_HIP_MEMCPY, CALL_HIP_NW_HIP_MEMCPY_PEER_ASYNC,
        RET_HIP_NW_HIP_MEMCPY_PEER_ASYNC, CALL_HIP_HIP_MEMCPY_HTO_D_ASYNC, RET_HIP_HIP_MEMCPY_HTO_D_ASYNC,
        CALL_HIP_HIP_MEMCPY_DTO_H_ASYNC, RET_HIP_HIP_MEMCPY_DTO_H_ASYNC, CALL_HIP_HIP_MEMCPY_DTO_D_ASYNC,
        RET_HIP_HIP_MEMCPY_DTO_D_ASYNC, CALL_HIP_NW_HIP_MEMCPY_SYNC, RET_HIP_NW_HIP_MEMCPY_SYNC,
        CALL_HIP_HIP_GET_DEVICE_COUNT, RET_HIP_HIP_GET_DEVICE_COUNT, CALL_HIP_NW_HIP_SET_DEVICE,
        RET_HIP_NW_HIP_SET_DEVICE, CALL_HIP_HIP_MEM_GET_INFO, RET_HIP_HIP_MEM_GET_INFO, CALL_HIP_NW_HIP_STREAM_CREATE,
        RET_HIP_NW_HIP_STREAM_CREATE, CALL_HIP_NW_HIP_GET_DEVICE, RET_HIP_NW_HIP_GET_DEVICE, CALL_HIP_HIP_INIT,
        RET_HIP_HIP_INIT, CALL_HIP_HIP_CTX_GET_CURRENT, RET_HIP_HIP_CTX_GET_CURRENT, CALL_HIP_NW_HIP_STREAM_SYNCHRONIZE,
        RET_HIP_NW_HIP_STREAM_SYNCHRONIZE, CALL_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES,
        RET_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES, CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL,
        RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL, CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL,
        RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL, CALL_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED,
        RET_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED, CALL_HIP_NW_HSA_EXECUTABLE_CREATE_ALT,
        RET_HIP_NW_HSA_EXECUTABLE_CREATE_ALT, CALL_HIP_NW_HSA_ISA_FROM_NAME, RET_HIP_NW_HSA_ISA_FROM_NAME,
        CALL_HIP_HIP_PEEK_AT_LAST_ERROR, RET_HIP_HIP_PEEK_AT_LAST_ERROR, CALL_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE,
        RET_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE, CALL_HIP_HIP_MODULE_LOAD_DATA, RET_HIP_HIP_MODULE_LOAD_DATA,
        CALL_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO, RET_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO,
        CALL_HIP_NW_HIP_CTX_SET_CURRENT, RET_HIP_NW_HIP_CTX_SET_CURRENT, CALL_HIP_HIP_EVENT_CREATE,
        RET_HIP_HIP_EVENT_CREATE, CALL_HIP_HIP_EVENT_RECORD, RET_HIP_HIP_EVENT_RECORD, CALL_HIP_HIP_EVENT_SYNCHRONIZE,
        RET_HIP_HIP_EVENT_SYNCHRONIZE, CALL_HIP_HIP_EVENT_DESTROY, RET_HIP_HIP_EVENT_DESTROY,
        CALL_HIP_HIP_EVENT_ELAPSED_TIME, RET_HIP_HIP_EVENT_ELAPSED_TIME, CALL_HIP_HIP_MODULE_LOAD,
        RET_HIP_HIP_MODULE_LOAD, CALL_HIP_HIP_MODULE_UNLOAD, RET_HIP_HIP_MODULE_UNLOAD, CALL_HIP_NW_HIP_STREAM_DESTROY,
        RET_HIP_NW_HIP_STREAM_DESTROY, CALL_HIP_HIP_MODULE_GET_FUNCTION, RET_HIP_HIP_MODULE_GET_FUNCTION,
        CALL_HIP_HIP_GET_LAST_ERROR, RET_HIP_HIP_GET_LAST_ERROR, CALL_HIP_HIP_MEMSET, RET_HIP_HIP_MEMSET,
        CALL_HIP_HIP_STREAM_WAIT_EVENT, RET_HIP_HIP_STREAM_WAIT_EVENT, CALL_HIP___DO_C_HSA_AGENT_GET_INFO,
        RET_HIP___DO_C_HSA_AGENT_GET_INFO, CALL_HIP___DO_C_LOAD_EXECUTABLE, RET_HIP___DO_C_LOAD_EXECUTABLE,
        CALL_HIP___DO_C_GET_AGENTS, RET_HIP___DO_C_GET_AGENTS, CALL_HIP___DO_C_GET_ISAS, RET_HIP___DO_C_GET_ISAS,
        CALL_HIP___DO_C_GET_KERENEL_SYMBOLS, RET_HIP___DO_C_GET_KERENEL_SYMBOLS, CALL_HIP___DO_C_QUERY_HOST_ADDRESS,
        RET_HIP___DO_C_QUERY_HOST_ADDRESS, CALL_HIP___DO_C_GET_KERNEL_DESCRIPTOR, RET_HIP___DO_C_GET_KERNEL_DESCRIPTOR,
        CALL_HIP_NW_HIP_CTX_GET_DEVICE, RET_HIP_NW_HIP_CTX_GET_DEVICE, CALL_HIP_NW_LOOKUP_KERN_INFO,
        RET_HIP_NW_LOOKUP_KERN_INFO, CALL_HIP___DO_C_MASS_SYMBOL_INFO, RET_HIP___DO_C_MASS_SYMBOL_INFO
};

#include "hip_nw_utility_types.h"

struct ava_handle_pair_t {
    void *a;
    void *b;
};

struct ava_offset_pair_t {
    size_t a;
    size_t b;
};

struct ava_offset_pair_t *
ava_new_offset_pair(size_t a, size_t b)
{
    struct ava_offset_pair_t *ret = (struct ava_offset_pair_t *)malloc(sizeof(struct ava_offset_pair_t));
    ret->a = a;
    ret->b = b;
    return ret;
}

struct hip_metadata {
    int application;
    ava_extract_function extract;
    ava_replace_function replace;
    GPtrArray * /* ava_offset_pair_t* */ recorded_calls;
    GPtrArray * /* handle */ dependencies;
};

struct hip_hip_device_synchronize_call {
    struct command_base base;
    intptr_t __call_id;

};

struct hip_hip_device_synchronize_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_device_synchronize_call_record {
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_malloc_call {
    struct command_base base;
    intptr_t __call_id;
    void **dptr;
    size_t size;
};

struct hip_hip_malloc_ret {
    struct command_base base;
    intptr_t __call_id;
    void **dptr;
    hipError_t ret;
};

struct hip_hip_malloc_call_record {
    void **dptr;
    size_t size;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_free_call {
    struct command_base base;
    intptr_t __call_id;
    void *ptr;
};

struct hip_hip_free_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_free_call_record {
    void *ptr;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_hto_d_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t dst;
    size_t sizeBytes;
    void *src;
};

struct hip_hip_memcpy_hto_d_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_memcpy_hto_d_call_record {
    hipDeviceptr_t dst;
    size_t sizeBytes;
    void *src;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_dto_h_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t src;
    size_t sizeBytes;
    void *dst;
};

struct hip_hip_memcpy_dto_h_ret {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    hipError_t ret;
};

struct hip_hip_memcpy_dto_h_call_record {
    hipDeviceptr_t src;
    size_t sizeBytes;
    void *dst;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_dto_d_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
};

struct hip_hip_memcpy_dto_d_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_memcpy_dto_d_call_record {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_memcpy_call {
    struct command_base base;
    intptr_t __call_id;
    size_t sizeBytes;
    hipMemcpyKind kind;
    void *src;
    void *dst;
};

struct hip_nw_hip_memcpy_ret {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    hipError_t ret;
};

struct hip_nw_hip_memcpy_call_record {
    size_t sizeBytes;
    hipMemcpyKind kind;
    void *src;
    void *dst;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_memcpy_peer_async_call {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    int dstDeviceId;
    void *src;
    int srcDevice;
    size_t sizeBytes;
    hipStream_t stream;
};

struct hip_nw_hip_memcpy_peer_async_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_nw_hip_memcpy_peer_async_call_record {
    void *dst;
    int dstDeviceId;
    void *src;
    int srcDevice;
    size_t sizeBytes;
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_hto_d_async_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t dst;
    size_t sizeBytes;
    hipStream_t stream;
    void *src;
};

struct hip_hip_memcpy_hto_d_async_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_memcpy_hto_d_async_call_record {
    hipDeviceptr_t dst;
    size_t sizeBytes;
    hipStream_t stream;
    void *src;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_dto_h_async_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t src;
    size_t sizeBytes;
    void *dst;
    hipStream_t stream;
};

struct hip_hip_memcpy_dto_h_async_ret {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    hipError_t ret;
};

struct hip_hip_memcpy_dto_h_async_call_record {
    hipDeviceptr_t src;
    size_t sizeBytes;
    void *dst;
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memcpy_dto_d_async_call {
    struct command_base base;
    intptr_t __call_id;
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
};

struct hip_hip_memcpy_dto_d_async_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_memcpy_dto_d_async_call_record {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_memcpy_sync_call {
    struct command_base base;
    intptr_t __call_id;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
    void *src;
    void *dst;
};

struct hip_nw_hip_memcpy_sync_ret {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    hipError_t ret;
};

struct hip_nw_hip_memcpy_sync_call_record {
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
    void *src;
    void *dst;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_get_device_count_call {
    struct command_base base;
    intptr_t __call_id;
    int *count;
};

struct hip_hip_get_device_count_ret {
    struct command_base base;
    intptr_t __call_id;
    int *count;
    hipError_t ret;
};

struct hip_hip_get_device_count_call_record {
    int *count;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_set_device_call {
    struct command_base base;
    intptr_t __call_id;
    int deviceId;
};

struct hip_nw_hip_set_device_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_nw_hip_set_device_call_record {
    int deviceId;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_mem_get_info_call {
    struct command_base base;
    intptr_t __call_id;
    size_t *__free;
    size_t *total;
};

struct hip_hip_mem_get_info_ret {
    struct command_base base;
    intptr_t __call_id;
    size_t *__free;
    size_t *total;
    hipError_t ret;
};

struct hip_hip_mem_get_info_call_record {
    size_t *__free;
    size_t *total;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_stream_create_call {
    struct command_base base;
    intptr_t __call_id;
    hipStream_t *stream;
    hsa_agent_t *agent;
};

struct hip_nw_hip_stream_create_ret {
    struct command_base base;
    intptr_t __call_id;
    hipStream_t *stream;
    hsa_agent_t *agent;
    hipError_t ret;
};

struct hip_nw_hip_stream_create_call_record {
    hipStream_t *stream;
    hsa_agent_t *agent;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_get_device_call {
    struct command_base base;
    intptr_t __call_id;
    int *deviceId;
};

struct hip_nw_hip_get_device_ret {
    struct command_base base;
    intptr_t __call_id;
    int *deviceId;
    hipError_t ret;
};

struct hip_nw_hip_get_device_call_record {
    int *deviceId;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_init_call {
    struct command_base base;
    intptr_t __call_id;
    unsigned int flags;
};

struct hip_hip_init_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_init_call_record {
    unsigned int flags;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_ctx_get_current_call {
    struct command_base base;
    intptr_t __call_id;
    hipCtx_t *ctx;
};

struct hip_hip_ctx_get_current_ret {
    struct command_base base;
    intptr_t __call_id;
    hipCtx_t *ctx;
    hipError_t ret;
};

struct hip_hip_ctx_get_current_call_record {
    hipCtx_t *ctx;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_stream_synchronize_call {
    struct command_base base;
    intptr_t __call_id;
    hipStream_t stream;
};

struct hip_nw_hip_stream_synchronize_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_nw_hip_stream_synchronize_call_record {
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_hip_get_device_properties_call {
    struct command_base base;
    intptr_t __call_id;
    char *prop;
    int deviceId;
};

struct hip___do_c_hip_get_device_properties_ret {
    struct command_base base;
    intptr_t __call_id;
    char *prop;
    hipError_t ret;
};

struct hip___do_c_hip_get_device_properties_call_record {
    char *prop;
    int deviceId;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_hip_hcc_module_launch_kernel_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_kernel_dispatch_packet_t *aql;
    hipStream_t stream;
    void **kernelParams;
    size_t extra_size;
    hipEvent_t start;
    char *extra;
    hipEvent_t stop;
};

struct hip___do_c_hip_hcc_module_launch_kernel_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip___do_c_hip_hcc_module_launch_kernel_call_record {
    hsa_kernel_dispatch_packet_t *aql;
    hipStream_t stream;
    void **kernelParams;
    size_t extra_size;
    hipEvent_t start;
    char *extra;
    hipEvent_t stop;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_hip_hcc_module_launch_multi_kernel_call {
    struct command_base base;
    intptr_t __call_id;
    int numKernels;
    size_t *extra_size;
    hsa_kernel_dispatch_packet_t *aql;
    hipStream_t stream;
    hipEvent_t *stop;
    hipEvent_t *start;
    size_t total_extra_size;
    char *all_extra;
};

struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record {
    int numKernels;
    size_t *extra_size;
    hsa_kernel_dispatch_packet_t *aql;
    hipStream_t stream;
    hipEvent_t *stop;
    hipEvent_t *start;
    size_t total_extra_size;
    char *all_extra;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hsa_system_major_extension_supported_call {
    struct command_base base;
    intptr_t __call_id;
    uint16_t extension;
    uint16_t version_major;
    uint16_t *version_minor;
    _Bool *result;
};

struct hip_nw_hsa_system_major_extension_supported_ret {
    struct command_base base;
    intptr_t __call_id;
    uint16_t *version_minor;
    _Bool *result;
    hsa_status_t ret;
};

struct hip_nw_hsa_system_major_extension_supported_call_record {
    uint16_t extension;
    uint16_t version_major;
    uint16_t *version_minor;
    _Bool *result;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hsa_executable_create_alt_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_profile_t profile;
    char *options;
    hsa_default_float_rounding_mode_t default_float_rounding_mode;
    hsa_executable_t *executable;
};

struct hip_nw_hsa_executable_create_alt_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_executable_t *executable;
    hsa_status_t ret;
};

struct hip_nw_hsa_executable_create_alt_call_record {
    hsa_profile_t profile;
    char *options;
    hsa_default_float_rounding_mode_t default_float_rounding_mode;
    hsa_executable_t *executable;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hsa_isa_from_name_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_isa_t *isa;
    char *name;
};

struct hip_nw_hsa_isa_from_name_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_isa_t *isa;
    hsa_status_t ret;
};

struct hip_nw_hsa_isa_from_name_call_record {
    hsa_isa_t *isa;
    char *name;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_peek_at_last_error_call {
    struct command_base base;
    intptr_t __call_id;

};

struct hip_hip_peek_at_last_error_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_peek_at_last_error_call_record {
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_device_get_attribute_call {
    struct command_base base;
    intptr_t __call_id;
    int *pi;
    hipDeviceAttribute_t attr;
    int deviceId;
};

struct hip_nw_hip_device_get_attribute_ret {
    struct command_base base;
    intptr_t __call_id;
    int *pi;
    hipError_t ret;
};

struct hip_nw_hip_device_get_attribute_call_record {
    int *pi;
    hipDeviceAttribute_t attr;
    int deviceId;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_module_load_data_call {
    struct command_base base;
    intptr_t __call_id;
    void *image;
    hipModule_t *module;
};

struct hip_hip_module_load_data_ret {
    struct command_base base;
    intptr_t __call_id;
    hipModule_t *module;
    hipError_t ret;
};

struct hip_hip_module_load_data_call_record {
    void *image;
    hipModule_t *module;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_hsa_executable_symbol_get_info_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_executable_symbol_t executable_symbol;
    hsa_executable_symbol_info_t attribute;
    size_t max_value;
    char *value;
};

struct hip___do_c_hsa_executable_symbol_get_info_ret {
    struct command_base base;
    intptr_t __call_id;
    char *value;
    hsa_status_t ret;
};

struct hip___do_c_hsa_executable_symbol_get_info_call_record {
    hsa_executable_symbol_t executable_symbol;
    hsa_executable_symbol_info_t attribute;
    size_t max_value;
    char *value;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_ctx_set_current_call {
    struct command_base base;
    intptr_t __call_id;
    hipCtx_t ctx;
};

struct hip_nw_hip_ctx_set_current_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_nw_hip_ctx_set_current_call_record {
    hipCtx_t ctx;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_event_create_call {
    struct command_base base;
    intptr_t __call_id;
    hipEvent_t *event;
};

struct hip_hip_event_create_ret {
    struct command_base base;
    intptr_t __call_id;
    hipEvent_t *event;
    hipError_t ret;
};

struct hip_hip_event_create_call_record {
    hipEvent_t *event;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_event_record_call {
    struct command_base base;
    intptr_t __call_id;
    hipEvent_t event;
    hipStream_t stream;
};

struct hip_hip_event_record_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_event_record_call_record {
    hipEvent_t event;
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_event_synchronize_call {
    struct command_base base;
    intptr_t __call_id;
    hipEvent_t event;
};

struct hip_hip_event_synchronize_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_event_synchronize_call_record {
    hipEvent_t event;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_event_destroy_call {
    struct command_base base;
    intptr_t __call_id;
    hipEvent_t event;
};

struct hip_hip_event_destroy_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_event_destroy_call_record {
    hipEvent_t event;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_event_elapsed_time_call {
    struct command_base base;
    intptr_t __call_id;
    float *ms;
    hipEvent_t start;
    hipEvent_t stop;
};

struct hip_hip_event_elapsed_time_ret {
    struct command_base base;
    intptr_t __call_id;
    float *ms;
    hipError_t ret;
};

struct hip_hip_event_elapsed_time_call_record {
    float *ms;
    hipEvent_t start;
    hipEvent_t stop;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_module_load_call {
    struct command_base base;
    intptr_t __call_id;
    char *fname;
    hipModule_t *module;
};

struct hip_hip_module_load_ret {
    struct command_base base;
    intptr_t __call_id;
    hipModule_t *module;
    hipError_t ret;
};

struct hip_hip_module_load_call_record {
    char *fname;
    hipModule_t *module;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_module_unload_call {
    struct command_base base;
    intptr_t __call_id;
    hipModule_t module;
};

struct hip_hip_module_unload_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_module_unload_call_record {
    hipModule_t module;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_stream_destroy_call {
    struct command_base base;
    intptr_t __call_id;
    hipStream_t stream;
};

struct hip_nw_hip_stream_destroy_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_nw_hip_stream_destroy_call_record {
    hipStream_t stream;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_module_get_function_call {
    struct command_base base;
    intptr_t __call_id;
    hipFunction_t *function;
    char *kname;
    hipModule_t module;
};

struct hip_hip_module_get_function_ret {
    struct command_base base;
    intptr_t __call_id;
    hipFunction_t *function;
    hipError_t ret;
};

struct hip_hip_module_get_function_call_record {
    hipFunction_t *function;
    char *kname;
    hipModule_t module;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_get_last_error_call {
    struct command_base base;
    intptr_t __call_id;

};

struct hip_hip_get_last_error_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_get_last_error_call_record {
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_memset_call {
    struct command_base base;
    intptr_t __call_id;
    void *dst;
    int value;
    size_t sizeBytes;
};

struct hip_hip_memset_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_memset_call_record {
    void *dst;
    int value;
    size_t sizeBytes;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_hip_stream_wait_event_call {
    struct command_base base;
    intptr_t __call_id;
    hipStream_t stream;
    hipEvent_t event;
    unsigned int flags;
};

struct hip_hip_stream_wait_event_ret {
    struct command_base base;
    intptr_t __call_id;

    hipError_t ret;
};

struct hip_hip_stream_wait_event_call_record {
    hipStream_t stream;
    hipEvent_t event;
    unsigned int flags;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_hsa_agent_get_info_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_agent_t agent;
    hsa_agent_info_t attribute;
    size_t max_value;
    void *value;
};

struct hip___do_c_hsa_agent_get_info_ret {
    struct command_base base;
    intptr_t __call_id;
    void *value;
    hsa_status_t ret;
};

struct hip___do_c_hsa_agent_get_info_call_record {
    hsa_agent_t agent;
    hsa_agent_info_t attribute;
    size_t max_value;
    void *value;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_load_executable_call {
    struct command_base base;
    intptr_t __call_id;
    size_t file_len;
    char *file_buf;
    hsa_executable_t *executable;
    hsa_agent_t *agent;
};

struct hip___do_c_load_executable_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_executable_t *executable;
    int ret;
};

struct hip___do_c_load_executable_call_record {
    size_t file_len;
    char *file_buf;
    hsa_executable_t *executable;
    hsa_agent_t *agent;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_get_agents_call {
    struct command_base base;
    intptr_t __call_id;
    size_t max_agents;
    hsa_agent_t *agents;
};

struct hip___do_c_get_agents_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_agent_t *agents;
    size_t ret;
};

struct hip___do_c_get_agents_call_record {
    size_t max_agents;
    hsa_agent_t *agents;
    size_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_get_isas_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_agent_t agents;
    size_t max_isas;
    hsa_isa_t *isas;
};

struct hip___do_c_get_isas_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_isa_t *isas;
    size_t ret;
};

struct hip___do_c_get_isas_call_record {
    hsa_agent_t agents;
    size_t max_isas;
    hsa_isa_t *isas;
    size_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_get_kerenel_symbols_call {
    struct command_base base;
    intptr_t __call_id;
    hsa_executable_t *exec;
    hsa_agent_t *agent;
    size_t max_symbols;
    hsa_executable_symbol_t *symbols;
};

struct hip___do_c_get_kerenel_symbols_ret {
    struct command_base base;
    intptr_t __call_id;
    hsa_executable_symbol_t *symbols;
    size_t ret;
};

struct hip___do_c_get_kerenel_symbols_call_record {
    hsa_executable_t *exec;
    hsa_agent_t *agent;
    size_t max_symbols;
    hsa_executable_symbol_t *symbols;
    size_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_query_host_address_call {
    struct command_base base;
    intptr_t __call_id;
    uint64_t kernel_object_;
    char *kernel_header_;
};

struct hip___do_c_query_host_address_ret {
    struct command_base base;
    intptr_t __call_id;
    char *kernel_header_;
    hsa_status_t ret;
};

struct hip___do_c_query_host_address_call_record {
    uint64_t kernel_object_;
    char *kernel_header_;
    hsa_status_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_get_kernel_descriptor_call {
    struct command_base base;
    intptr_t __call_id;
    char *name;
    hsa_executable_symbol_t *symbol;
    hipFunction_t *f;
};

struct hip___do_c_get_kernel_descriptor_ret {
    struct command_base base;
    intptr_t __call_id;
    hipFunction_t *f;
    hipError_t ret;
};

struct hip___do_c_get_kernel_descriptor_call_record {
    char *name;
    hsa_executable_symbol_t *symbol;
    hipFunction_t *f;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_hip_ctx_get_device_call {
    struct command_base base;
    intptr_t __call_id;
    hipDevice_t *device;
};

struct hip_nw_hip_ctx_get_device_ret {
    struct command_base base;
    intptr_t __call_id;
    hipDevice_t *device;
    hipError_t ret;
};

struct hip_nw_hip_ctx_get_device_call_record {
    hipDevice_t *device;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip_nw_lookup_kern_info_call {
    struct command_base base;
    intptr_t __call_id;
    hipFunction_t f;
    struct nw_kern_info *info;
};

struct hip_nw_lookup_kern_info_ret {
    struct command_base base;
    intptr_t __call_id;
    struct nw_kern_info *info;
    hipError_t ret;
};

struct hip_nw_lookup_kern_info_call_record {
    hipFunction_t f;
    struct nw_kern_info *info;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct hip___do_c_mass_symbol_info_call {
    struct command_base base;
    intptr_t __call_id;
    size_t n;
    unsigned int *offsets;
    hsa_executable_symbol_t *syms;
    size_t pool_size;
    uint8_t *agents;
    hsa_symbol_kind_t *types;
    hipFunction_t *descriptors;
    char *pool;
};

struct hip___do_c_mass_symbol_info_ret {
    struct command_base base;
    intptr_t __call_id;
    unsigned int *offsets;
    uint8_t *agents;
    hsa_symbol_kind_t *types;
    hipFunction_t *descriptors;
    char *pool;
    hipError_t ret;
};

struct hip___do_c_mass_symbol_info_call_record {
    size_t n;
    unsigned int *offsets;
    hsa_executable_symbol_t *syms;
    size_t pool_size;
    uint8_t *agents;
    hsa_symbol_kind_t *types;
    hipFunction_t *descriptors;
    char *pool;
    hipError_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

#endif                                           // ndef __HIP_NW_H__
