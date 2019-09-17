
#define ava_is_worker 1
#define ava_is_guest 0

#include "worker.h"

#include "common/endpoint_lib.h"
#include "common/linkage.h"

// Must be included before hip_nw.h, so that API
// functions are declared properly.
#include <hip_cpp_bridge.h>
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_runtime_api.h>
#include <hsa_limited.h>
#include <stdint.h>
#include "hip_nw.h"

static GHashTable *hip_managed_buffer_map;
static GHashTable *hip_managed_by_coupled_map;
static pthread_mutex_t hip_managed_buffer_map_mutex = PTHREAD_MUTEX_INITIALIZER;
static GHashTable *hip_metadata_map;
static pthread_mutex_t hip_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
static GHashTable *hip_call_map;
static pthread_mutex_t hip_call_map_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_intptr_t hip_call_counter;

static void __handle_command_hip_init();
static void __handle_command_hip_destroy();
void __handle_command_hip(struct command_base *__cmd, int chan_no);

struct ava_coupled_record_t {
    GPtrArray * /* elements: struct call_id_and_handle_t* */ key_list;
    GPtrArray *buffer_list;
};

static void
ava_coupled_record_free(struct ava_coupled_record_t *r)
{
    // Dealloc all keys and buffers (by removing them from the buffer map and triggering their destroy callbacks)
    g_ptr_array_foreach(r->key_list, (GFunc) nw_hash_table_remove_flipped, hip_managed_buffer_map);
    // Dealloc the key_list itself (it has no destroy callback)
    g_ptr_array_unref(r->key_list);
    // Dealloc all buffers (by destroy callback)
    g_ptr_array_unref(r->buffer_list);
    free(r);
}

static struct ava_coupled_record_t *
ava_coupled_record_new()
{
    struct ava_coupled_record_t *ret = (struct ava_coupled_record_t *)malloc(sizeof(struct ava_coupled_record_t));
    ret->key_list = g_ptr_array_new_full(1, NULL);
    ret->buffer_list = g_ptr_array_new_full(1, (GDestroyNotify) g_array_unref);
    return ret;
}

/// Get the metadata for an object. This uses a lock to protect the hash_table. The `pure` annotation is a bit of a 
/// stretch since we do insert elements into the hash table, but having the compiler perform CSE on this function is 
/// pretty important and this function is idempotent.
__attribute__ ((pure))
static struct hip_metadata *
__ava_internal_metadata(void *p)
{
    void *ptr = (p);
    pthread_mutex_lock(&hip_metadata_map_mutex);
    void *metadata = g_hash_table_lookup(hip_metadata_map, ptr);
    if (metadata == NULL) {
        metadata = calloc(1, sizeof(struct hip_metadata));
        g_hash_table_insert(hip_metadata_map, ptr, metadata);
    }
    pthread_mutex_unlock(&hip_metadata_map_mutex);
    return (struct hip_metadata *)metadata;
}

#define ava_metadata(p) (&__ava_internal_metadata(p)->application)

static intptr_t
ava_get_call_id()
{
    return atomic_fetch_add(&hip_call_counter, 1);
}

static void
ava_add_call(intptr_t id, void *ptr)
{
    pthread_mutex_lock(&hip_call_map_mutex);
    gboolean b = g_hash_table_insert(hip_call_map, (void *)id, ptr);
    assert(b && "Adding a call ID which currently exists.");
    pthread_mutex_unlock(&hip_call_map_mutex);
}

static void *
ava_remove_call(intptr_t id)
{
    pthread_mutex_lock(&hip_call_map_mutex);
    void *ptr = nw_hash_table_steal_value(hip_call_map, (void *)id);
    assert(ptr != NULL && "Removing a call ID which does not exist");
    pthread_mutex_unlock(&hip_call_map_mutex);
    return ptr;
}

static struct ava_coupled_record_t *
ava_get_coupled_record(const void *coupled)
{
    struct ava_coupled_record_t *rec = g_hash_table_lookup(hip_managed_by_coupled_map, coupled);
    if (rec == NULL) {
        rec = ava_coupled_record_new();
        g_hash_table_insert(hip_managed_by_coupled_map, coupled, rec);
    }
    return rec;
}

static void *
ava_cached_alloc(int call_id, const void *coupled, size_t size)
{
    pthread_mutex_lock(&hip_managed_buffer_map_mutex);
    struct call_id_and_handle_t key = { call_id, coupled };
    GArray *buffer = (GArray *) g_hash_table_lookup(hip_managed_buffer_map, &key);
    if (buffer == NULL) {
        buffer = g_array_sized_new(FALSE, TRUE, 1, size);
        struct call_id_and_handle_t *pkey = (struct call_id_and_handle_t *)malloc(sizeof(struct call_id_and_handle_t));
        *pkey = key;
        g_hash_table_insert(hip_managed_buffer_map, pkey, buffer);
        struct ava_coupled_record_t *rec = ava_get_coupled_record(coupled);
        g_ptr_array_add(rec->key_list, pkey);
        g_ptr_array_add(rec->buffer_list, buffer);
    }
    // TODO: This will probably never shrink the buffer. We may need to implement that for large changes. 
    g_array_set_size(buffer, size);
    pthread_mutex_unlock(&hip_managed_buffer_map_mutex);
    return buffer->data;
}

static void *
ava_uncached_alloc(const void *coupled, size_t size)
{
    pthread_mutex_lock(&hip_managed_buffer_map_mutex);
    GArray *buffer = g_array_sized_new(FALSE, TRUE, 1, size);
    struct ava_coupled_record_t *rec = ava_get_coupled_record(coupled);
    g_ptr_array_add(rec->buffer_list, buffer);
    pthread_mutex_unlock(&hip_managed_buffer_map_mutex);
    return buffer->data;
}

static void
ava_coupled_free(const void *coupled)
{
    pthread_mutex_lock(&hip_managed_buffer_map_mutex);
    g_hash_table_remove(hip_managed_by_coupled_map, coupled);
    pthread_mutex_unlock(&hip_managed_buffer_map_mutex);
}

static inline void *
ava_static_alloc(int call_id, size_t size)
{
    return ava_cached_alloc(call_id, NULL, size);
}

static inline void
ava_add_recorded_call(void *handle, struct ava_offset_pair_t *pair)
{
    struct hip_metadata *__internal_metadata = __ava_internal_metadata(handle);
    if (__internal_metadata->recorded_calls == NULL) {
        __internal_metadata->recorded_calls = g_ptr_array_new_full(1, free);
    }
    g_ptr_array_add(__internal_metadata->recorded_calls, pair);
}

static inline void
ava_add_dependency(void *a, void *b)
{
    struct hip_metadata *__internal_metadata = __ava_internal_metadata(a);
    if (__internal_metadata->recorded_calls == NULL) {
        __internal_metadata->recorded_calls = g_ptr_array_new_full(1, NULL);
    }
    g_ptr_array_add(__internal_metadata->dependencies, b);
}

#include "hip_nw_utilities.h"

void __attribute__ ((constructor)) init_hip_worker(void)
{
    __handle_command_hip_init();
}

void __attribute__ ((destructor)) fini_hip_worker(void)
{
   __handle_command_hip_destroy();
}

static struct nw_handle_pool *handle_pool = NULL;

static hipError_t
__wrapper_hipDeviceSynchronize()
{
    hipError_t ret;
    ret = hipDeviceSynchronize();

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMalloc(void **dptr, size_t size)
{
    hipError_t ret;
    ret = hipMalloc(dptr, size);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipFree(void *ptr)
{
    hipError_t ret;
    ret = hipFree(ptr);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyHtoD(hipDeviceptr_t dst, size_t sizeBytes, void *src)
{
    hipError_t ret;
    ret = hipMemcpyHtoD(dst, src, sizeBytes);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyDtoH(hipDeviceptr_t src, size_t sizeBytes, void *dst)
{
    hipError_t ret;
    ret = hipMemcpyDtoH(dst, src, sizeBytes);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)
{
    hipError_t ret;
    ret = hipMemcpyDtoD(dst, src, sizeBytes);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipMemcpy(size_t sizeBytes, hipMemcpyKind kind, const void *src, void *dst)
{
    hipError_t ret;
    ret = nw_hipMemcpy(dst, src, sizeBytes, kind);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src, int srcDevice, size_t sizeBytes,
    hipStream_t stream)
{
    hipError_t ret;
    ret = nw_hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyHtoDAsync(hipDeviceptr_t dst, size_t sizeBytes, hipStream_t stream, void *src)
{
    hipError_t ret;
    ret = hipMemcpyHtoDAsync(dst, src, sizeBytes, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyDtoHAsync(hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream, void *dst)
{
    hipError_t ret;
    ret = hipMemcpyDtoHAsync(dst, src, sizeBytes, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    hipError_t ret;
    ret = hipMemcpyDtoDAsync(dst, src, sizeBytes, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipMemcpySync(size_t sizeBytes, hipMemcpyKind kind, const void *src, void *dst, hipStream_t stream)
{
    hipError_t ret;
    ret = nw_hipMemcpySync(dst, src, sizeBytes, kind, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipGetDeviceCount(int *count)
{
    hipError_t ret;
    ret = hipGetDeviceCount(count);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipSetDevice(int deviceId)
{
    hipError_t ret;
    ret = nw_hipSetDevice(deviceId);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemGetInfo(size_t * __free, size_t * total)
{
    hipError_t ret;
    ret = hipMemGetInfo(__free, total);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipStreamCreate(hipStream_t * stream, hsa_agent_t * agent)
{
    hipError_t ret;
    ret = nw_hipStreamCreate(stream, agent);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipGetDevice(int *deviceId)
{
    hipError_t ret;
    ret = nw_hipGetDevice(deviceId);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipInit(unsigned int flags)
{
    hipError_t ret;
    ret = hipInit(flags);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipCtxGetCurrent(hipCtx_t * ctx)
{
    hipError_t ret;
    ret = hipCtxGetCurrent(ctx);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipStreamSynchronize(hipStream_t stream)
{
    hipError_t ret;
    ret = nw_hipStreamSynchronize(stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_hipGetDeviceProperties(char *prop, int deviceId)
{
    hipError_t ret;
    ret = __do_c_hipGetDeviceProperties(prop, deviceId);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_hipHccModuleLaunchKernel(hsa_kernel_dispatch_packet_t * aql, hipStream_t stream,
    void **kernelParams, size_t extra_size, hipEvent_t start, char *extra, hipEvent_t stop)
{
    hipError_t ret;
    ret = __do_c_hipHccModuleLaunchKernel(aql, stream, kernelParams, extra, extra_size, start, stop);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_hipHccModuleLaunchMultiKernel(int numKernels, size_t * extra_size, hipEvent_t * stop,
    hipStream_t stream, hsa_kernel_dispatch_packet_t * aql, hipEvent_t * start, size_t total_extra_size,
    char *all_extra)
{
    hipError_t ret;
    ret =
        __do_c_hipHccModuleLaunchMultiKernel(numKernels, aql, stream, all_extra, total_extra_size, extra_size, start,
        stop);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_hipHccModuleLaunchMultiKernel_and_memcpy(int numKernels, hsa_kernel_dispatch_packet_t * aql,
    size_t * extra_size, hipEvent_t * stop, hipStream_t stream, hipEvent_t * start, size_t total_extra_size,
    size_t sizeBytes, char *all_extra, hipMemcpyKind kind, void *dst, const void *src)
{
    hipError_t ret;
    ret =
        __do_c_hipHccModuleLaunchMultiKernel_and_memcpy(numKernels, aql, stream, all_extra, total_extra_size,
        extra_size, start, stop, dst, src, sizeBytes, kind);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper_nw_hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major, uint16_t * version_minor,
    _Bool * result)
{
    hsa_status_t ret;
    ret = nw_hsa_system_major_extension_supported(extension, version_major, version_minor, result);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper_nw_hsa_executable_create_alt(hsa_profile_t profile, const char *options,
    hsa_default_float_rounding_mode_t default_float_rounding_mode, hsa_executable_t * executable)
{
    hsa_status_t ret;
    ret = nw_hsa_executable_create_alt(profile, default_float_rounding_mode, options, executable);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper_nw_hsa_isa_from_name(hsa_isa_t * isa, const char *name)
{
    hsa_status_t ret;
    ret = nw_hsa_isa_from_name(name, isa);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipPeekAtLastError()
{
    hipError_t ret;
    ret = hipPeekAtLastError();

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId)
{
    hipError_t ret;
    ret = nw_hipDeviceGetAttribute(pi, attr, deviceId);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipModuleLoadData(const void *image, hipModule_t * module)
{
    hipError_t ret;
    ret = hipModuleLoadData(module, image);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper___do_c_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, size_t max_value, char *value)
{
    hsa_status_t ret;
    ret = __do_c_hsa_executable_symbol_get_info(executable_symbol, attribute, value, max_value);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipCtxSetCurrent(hipCtx_t ctx)
{
    hipError_t ret;
    ret = nw_hipCtxSetCurrent(ctx);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipEventCreate(hipEvent_t * event)
{
    hipError_t ret;
    ret = hipEventCreate(event);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    hipError_t ret;
    ret = hipEventRecord(event, stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipEventSynchronize(hipEvent_t event)
{
    hipError_t ret;
    ret = hipEventSynchronize(event);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipEventDestroy(hipEvent_t event)
{
    hipError_t ret;
    ret = hipEventDestroy(event);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    hipError_t ret;
    ret = hipEventElapsedTime(ms, start, stop);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipModuleLoad(const char *fname, hipModule_t * module)
{
    hipError_t ret;
    ret = hipModuleLoad(module, fname);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipModuleUnload(hipModule_t module)
{
    hipError_t ret;
    ret = hipModuleUnload(module);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipStreamDestroy(hipStream_t stream)
{
    hipError_t ret;
    ret = nw_hipStreamDestroy(stream);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipModuleGetFunction(hipFunction_t * function, const char *kname, hipModule_t module)
{
    hipError_t ret;
    ret = hipModuleGetFunction(function, module, kname);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipGetLastError()
{
    hipError_t ret;
    ret = hipGetLastError();

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipMemset(void *dst, int value, size_t sizeBytes)
{
    hipError_t ret;
    ret = hipMemset(dst, value, sizeBytes);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    hipError_t ret;
    ret = hipStreamWaitEvent(stream, event, flags);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper___do_c_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, size_t max_value, void *value)
{
    hsa_status_t ret;
    ret = __do_c_hsa_agent_get_info(agent, attribute, value, max_value);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static int
__wrapper___do_c_load_executable(size_t file_len, const char *file_buf, hsa_executable_t * executable,
    hsa_agent_t * agent)
{
    int ret;
    ret = __do_c_load_executable(file_buf, file_len, executable, agent);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static size_t
__wrapper___do_c_get_agents(size_t max_agents, hsa_agent_t * agents)
{
    size_t ret;
    ret = __do_c_get_agents(agents, max_agents);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static size_t
__wrapper___do_c_get_isas(hsa_agent_t agents, size_t max_isas, hsa_isa_t * isas)
{
    size_t ret;
    ret = __do_c_get_isas(agents, isas, max_isas);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static size_t
__wrapper___do_c_get_kerenel_symbols(const hsa_executable_t * exec, const hsa_agent_t * agent, size_t max_symbols,
    hsa_executable_symbol_t * symbols)
{
    size_t ret;
    ret = __do_c_get_kerenel_symbols(exec, agent, symbols, max_symbols);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hsa_status_t
__wrapper___do_c_query_host_address(uint64_t kernel_object_, char *kernel_header_)
{
    hsa_status_t ret;
    ret = __do_c_query_host_address(kernel_object_, kernel_header_);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_get_kernel_descriptor(const char *name, const hsa_executable_symbol_t * symbol, hipFunction_t * f)
{
    hipError_t ret;
    ret = __do_c_get_kernel_descriptor(symbol, name, f);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_hipCtxGetDevice(hipDevice_t * device)
{
    hipError_t ret;
    ret = nw_hipCtxGetDevice(device);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper_nw_lookup_kern_info(hipFunction_t f, struct nw_kern_info *info)
{
    hipError_t ret;
    ret = nw_lookup_kern_info(f, info);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

static hipError_t
__wrapper___do_c_mass_symbol_info(size_t n, size_t pool_size, uint8_t * agents, hsa_symbol_kind_t * types,
    hipFunction_t * descriptors, const hsa_executable_symbol_t * syms, unsigned int *offsets, char *pool)
{
    hipError_t ret;
    ret = __do_c_mass_symbol_info(n, syms, types, descriptors, agents, offsets, pool, pool_size);

    /* Report resources */

#ifdef AVA_API_FUNCTION_CALL_RESOURCE
    nw_report_throughput_resource_consumption("ava_api_function_call", 1);
#endif

    return ret;
}

void
__handle_command_hip_init()
{
    hip_managed_buffer_map =
        g_hash_table_new_full(nw_hash_call_id_and_handle, nw_equal_call_id_and_handle, free,
        (GDestroyNotify) g_array_unref);
    hip_managed_by_coupled_map =
        g_hash_table_new_full(nw_hash_pointer, g_direct_equal, NULL, (GDestroyNotify) ava_coupled_record_free);
    hip_metadata_map = metadata_map_new();
    hip_call_map = metadata_map_new();
    atomic_init(&hip_call_counter, 0);

#ifndef AVA_DISABLE_HANDLE_TRANSLATION
    handle_pool = nw_handle_pool_new();
#endif

    register_command_handler(HIP_API, __handle_command_hip);
}

void
__handle_command_hip_destroy()
{
    g_hash_table_unref(hip_managed_buffer_map);
    g_hash_table_unref(hip_managed_by_coupled_map);
    g_hash_table_unref(hip_metadata_map);
    g_hash_table_unref(hip_call_map);
}

__thread struct command_channel *__chan;

void
__handle_command_hip(struct command_base *__cmd, int chan_no)
{
    int ava_is_in,
     ava_is_out;
    set_ava_chan_no(chan_no);
    __chan = nw_global_command_channel[chan_no];
    switch (__cmd->command_id) {

    case CALL_HIP_HIP_DEVICE_SYNCHRONIZE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipDeviceSynchronize = g_ptr_array_new_full(0, free);
        struct hip_hip_device_synchronize_call *__call = (struct hip_hip_device_synchronize_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_device_synchronize_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipDeviceSynchronize();

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_device_synchronize_ret *__ret =
            (struct hip_hip_device_synchronize_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_device_synchronize_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_DEVICE_SYNCHRONIZE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipDeviceSynchronize);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MALLOC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMalloc = g_ptr_array_new_full(0, free);
        struct hip_hip_malloc_call *__call = (struct hip_hip_malloc_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_malloc_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: void ** dptr */
        void **dptr;
        dptr =
            ((__call->dptr) != (NULL)) ? (((void **)command_channel_get_buffer(__chan, __cmd,
                    __call->dptr))) : (__call->dptr);
        if (__call->dptr != NULL) {
            const size_t __size = ((size_t) 1);
            dptr = (void **)calloc(__size, sizeof(void *));
            g_ptr_array_add(__ava_alloc_list_hipMalloc, dptr);
        } else {
            dptr = NULL;
        }

        /* Input: size_t size */
        size_t size;
        size = __call->size;
        size = __call->size;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMalloc(dptr, size);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void ** dptr */
            if ((dptr) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(void *));
            }
        }
        struct hip_hip_malloc_ret *__ret =
            (struct hip_hip_malloc_ret *)command_channel_new_command(__chan, sizeof(struct hip_hip_malloc_ret),
            __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MALLOC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void ** dptr */
        if ((dptr) != (NULL)) {
            __ret->dptr =
                (void **)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dptr,
                (1) * sizeof(void *));
        } else {
            __ret->dptr = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMalloc);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_FREE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipFree = g_ptr_array_new_full(0, free);
        struct hip_hip_free_call *__call = (struct hip_hip_free_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_free_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: void * ptr */
        void *ptr;
        ptr = __call->ptr;
        ptr = __call->ptr;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipFree(ptr);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_free_ret *__ret =
            (struct hip_hip_free_ret *)command_channel_new_command(__chan, sizeof(struct hip_hip_free_ret),
            __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_FREE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipFree);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_HTO_D:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyHtoD = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_hto_d_call *__call = (struct hip_hip_memcpy_hto_d_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_hto_d_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t dst */
        hipDeviceptr_t dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: void * src */
        void *src;
        src =
            ((__call->src) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->src))) : (__call->src);
        if (__call->src != NULL)
            src =
                ((__call->src) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                        __call->src))) : (__call->src);
        else
            src = NULL;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyHtoD(dst, sizeBytes, src);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_memcpy_hto_d_ret *__ret =
            (struct hip_hip_memcpy_hto_d_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_hto_d_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_HTO_D;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyHtoD);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_DTO_H:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyDtoH = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_dto_h_call *__call = (struct hip_hip_memcpy_dto_h_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_dto_h_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t src */
        hipDeviceptr_t src;
        src = __call->src;
        src = __call->src;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: void * dst */
        void *dst;
        dst =
            ((__call->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->dst))) : (__call->dst);
        if (__call->dst != NULL) {
            const size_t __size = ((size_t) sizeBytes);
            dst = (void *)calloc(__size, sizeof(void));
            g_ptr_array_add(__ava_alloc_list_hipMemcpyDtoH, dst);
        } else {
            dst = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyDtoH(src, sizeBytes, dst);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * dst */
            if ((dst) != (NULL) && (sizeBytes) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (sizeBytes) * sizeof(void));
            }
        }
        struct hip_hip_memcpy_dto_h_ret *__ret =
            (struct hip_hip_memcpy_dto_h_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_dto_h_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_DTO_H;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void * dst */
        if ((dst) != (NULL) && (sizeBytes) > (0)) {
            __ret->dst =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dst,
                (sizeBytes) * sizeof(void));
        } else {
            __ret->dst = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoH);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_DTO_D:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyDtoD = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_dto_d_call *__call = (struct hip_hip_memcpy_dto_d_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_dto_d_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t dst */
        hipDeviceptr_t dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: hipDeviceptr_t src */
        hipDeviceptr_t src;
        src = __call->src;
        src = __call->src;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyDtoD(dst, src, sizeBytes);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_memcpy_dto_d_ret *__ret =
            (struct hip_hip_memcpy_dto_d_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_dto_d_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_DTO_D;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoD);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_MEMCPY:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipMemcpy = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_memcpy_call *__call = (struct hip_nw_hip_memcpy_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_memcpy_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipMemcpyKind kind */
        hipMemcpyKind kind;
        kind = __call->kind;
        kind = __call->kind;

        /* Input: const void * src */
        void *src;
        src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                    __call->src))) : (__call->src);
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && kind == hipMemcpyHostToDevice) {
            if (__call->src != NULL)
                src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                            __call->src))) : (__call->src);
            else
                src = NULL;
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (__call->src != NULL) {
                    const size_t __size = ((size_t) (kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    src = (void *)calloc(__size, sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpy, src);
                    if (kind == hipMemcpyHostToDevice) {
                        void *__tmp_src_0;
                        __tmp_src_0 = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                                    __call->src))) : (__call->src);
                        const size_t __src_size_0 = (__size);
                        for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                            const size_t ava_index = __src_index_0;

                            char *__src_a_0;
                            __src_a_0 = (src) + __src_index_0;

                            char *__src_b_0;
                            __src_b_0 = (__tmp_src_0) + __src_index_0;

                            *__src_a_0 = *__src_b_0;
                            *__src_a_0 = *__src_b_0;
                    }}
                } else {
                    src = NULL;
                }
            } else {
                src = __call->src;
            }
        }

        /* Input: void * dst */
        void *dst;
        dst = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->dst))) : (__call->dst);
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (__call->dst != NULL) {
                const size_t __size = ((size_t) (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                dst = (void *)calloc(__size, sizeof(void));
                g_ptr_array_add(__ava_alloc_list_nw_hipMemcpy, dst);
            } else {
                dst = NULL;
            }
        } else {
            dst = __call->dst;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipMemcpy(sizeBytes, kind, src, dst);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                        && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                        __total_buffer_size +=
                            command_channel_buffer_size(__chan,
                            ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                    }
                }
        }}
        struct hip_nw_hip_memcpy_ret *__ret =
            (struct hip_nw_hip_memcpy_ret *)command_channel_new_command(__chan, sizeof(struct hip_nw_hip_memcpy_ret),
            __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_MEMCPY;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __ret->dst =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dst,
                    ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
            } else {
                __ret->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    const size_t __size = (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0);
                    void *__tmp_dst_0;
                    __tmp_dst_0 = (void *)calloc(__size, sizeof(void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpy, __tmp_dst_0);
                    const size_t __dst_size_0 = (__size);
                    for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                        const size_t ava_index = __dst_index_0;

                        char *__dst_a_0;
                        __dst_a_0 = (__tmp_dst_0) + __dst_index_0;

                        char *__dst_b_0;
                        __dst_b_0 = (dst) + __dst_index_0;

                        *__dst_a_0 = *__dst_b_0;
                    }
                    __ret->dst =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_dst_0,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                } else {
                    __ret->dst = NULL;
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                    __ret->dst = dst;
                }
            }
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpy);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_MEMCPY_PEER_ASYNC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipMemcpyPeerAsync = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_memcpy_peer_async_call *__call = (struct hip_nw_hip_memcpy_peer_async_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_memcpy_peer_async_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: void * dst */
        void *dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: int dstDeviceId */
        int dstDeviceId;
        dstDeviceId = __call->dstDeviceId;
        dstDeviceId = __call->dstDeviceId;

        /* Input: const void * src */
        void *src;
        src = __call->src;
        src = __call->src;

        /* Input: int srcDevice */
        int srcDevice;
        srcDevice = __call->srcDevice;
        srcDevice = __call->srcDevice;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_nw_hip_memcpy_peer_async_ret *__ret =
            (struct hip_nw_hip_memcpy_peer_async_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_memcpy_peer_async_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_MEMCPY_PEER_ASYNC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpyPeerAsync);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_HTO_D_ASYNC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyHtoDAsync = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_hto_d_async_call *__call = (struct hip_hip_memcpy_hto_d_async_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_hto_d_async_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t dst */
        hipDeviceptr_t dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: void * src */
        void *src;
        src =
            ((__call->src) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->src))) : (__call->src);
        if (__call->src != NULL)
            src =
                ((__call->src) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                        __call->src))) : (__call->src);
        else
            src = NULL;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyHtoDAsync(dst, sizeBytes, stream, src);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_memcpy_hto_d_async_ret *__ret =
            (struct hip_hip_memcpy_hto_d_async_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_hto_d_async_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_HTO_D_ASYNC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyHtoDAsync); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_DTO_H_ASYNC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyDtoHAsync = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_dto_h_async_call *__call = (struct hip_hip_memcpy_dto_h_async_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_dto_h_async_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t src */
        hipDeviceptr_t src;
        src = __call->src;
        src = __call->src;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: void * dst */
        void *dst;
        dst =
            ((__call->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->dst))) : (__call->dst);
        if (__call->dst != NULL) {
            const size_t __size = ((size_t) sizeBytes);
            dst = (void *)calloc(__size, sizeof(void));
            g_ptr_array_add(__ava_alloc_list_hipMemcpyDtoHAsync, dst);
        } else {
            dst = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyDtoHAsync(src, sizeBytes, stream, dst);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * dst */
            if ((dst) != (NULL) && (sizeBytes) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (sizeBytes) * sizeof(void));
            }
        }
        struct hip_hip_memcpy_dto_h_async_ret *__ret =
            (struct hip_hip_memcpy_dto_h_async_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_dto_h_async_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_DTO_H_ASYNC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void * dst */
        if ((dst) != (NULL) && (sizeBytes) > (0)) {
            __ret->dst =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dst,
                (sizeBytes) * sizeof(void));
        } else {
            __ret->dst = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoHAsync); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEMCPY_DTO_D_ASYNC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemcpyDtoDAsync = g_ptr_array_new_full(0, free);
        struct hip_hip_memcpy_dto_d_async_call *__call = (struct hip_hip_memcpy_dto_d_async_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memcpy_dto_d_async_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDeviceptr_t dst */
        hipDeviceptr_t dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: hipDeviceptr_t src */
        hipDeviceptr_t src;
        src = __call->src;
        src = __call->src;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemcpyDtoDAsync(dst, src, sizeBytes, stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_memcpy_dto_d_async_ret *__ret =
            (struct hip_hip_memcpy_dto_d_async_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_memcpy_dto_d_async_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMCPY_DTO_D_ASYNC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoDAsync); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_MEMCPY_SYNC:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipMemcpySync = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_memcpy_sync_call *__call = (struct hip_nw_hip_memcpy_sync_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_memcpy_sync_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: hipMemcpyKind kind */
        hipMemcpyKind kind;
        kind = __call->kind;
        kind = __call->kind;

        /* Input: const void * src */
        void *src;
        src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                    __call->src))) : (__call->src);
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && kind == hipMemcpyHostToDevice) {
            if (__call->src != NULL)
                src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                            __call->src))) : (__call->src);
            else
                src = NULL;
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (__call->src != NULL) {
                    const size_t __size = ((size_t) (kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    src = (void *)calloc(__size, sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpySync, src);
                    if (kind == hipMemcpyHostToDevice) {
                        void *__tmp_src_0;
                        __tmp_src_0 = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                                    __call->src))) : (__call->src);
                        const size_t __src_size_0 = (__size);
                        for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                            const size_t ava_index = __src_index_0;

                            char *__src_a_0;
                            __src_a_0 = (src) + __src_index_0;

                            char *__src_b_0;
                            __src_b_0 = (__tmp_src_0) + __src_index_0;

                            *__src_a_0 = *__src_b_0;
                            *__src_a_0 = *__src_b_0;
                    }}
                } else {
                    src = NULL;
                }
            } else {
                src = __call->src;
            }
        }

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: void * dst */
        void *dst;
        dst = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->dst))) : (__call->dst);
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (__call->dst != NULL) {
                const size_t __size = ((size_t) (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                dst = (void *)calloc(__size, sizeof(void));
                g_ptr_array_add(__ava_alloc_list_nw_hipMemcpySync, dst);
            } else {
                dst = NULL;
            }
        } else {
            dst = __call->dst;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipMemcpySync(sizeBytes, kind, src, dst, stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                        && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                        __total_buffer_size +=
                            command_channel_buffer_size(__chan,
                            ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                    }
                }
        }}
        struct hip_nw_hip_memcpy_sync_ret *__ret =
            (struct hip_nw_hip_memcpy_sync_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_memcpy_sync_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_MEMCPY_SYNC;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __ret->dst =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dst,
                    ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
            } else {
                __ret->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    const size_t __size = (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0);
                    void *__tmp_dst_0;
                    __tmp_dst_0 = (void *)calloc(__size, sizeof(void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpySync, __tmp_dst_0);
                    const size_t __dst_size_0 = (__size);
                    for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                        const size_t ava_index = __dst_index_0;

                        char *__dst_a_0;
                        __dst_a_0 = (__tmp_dst_0) + __dst_index_0;

                        char *__dst_b_0;
                        __dst_b_0 = (dst) + __dst_index_0;

                        *__dst_a_0 = *__dst_b_0;
                    }
                    __ret->dst =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_dst_0,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                } else {
                    __ret->dst = NULL;
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                    __ret->dst = dst;
                }
            }
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpySync);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_GET_DEVICE_COUNT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipGetDeviceCount = g_ptr_array_new_full(0, free);
        struct hip_hip_get_device_count_call *__call = (struct hip_hip_get_device_count_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_get_device_count_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int * count */
        int *count;
        count =
            ((__call->count) != (NULL)) ? (((int *)command_channel_get_buffer(__chan, __cmd,
                    __call->count))) : (__call->count);
        if (__call->count != NULL) {
            const size_t __size = ((size_t) 1);
            count = (int *)calloc(__size, sizeof(int));
            g_ptr_array_add(__ava_alloc_list_hipGetDeviceCount, count);
        } else {
            count = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipGetDeviceCount(count);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: int * count */
            if ((count) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(int));
            }
        }
        struct hip_hip_get_device_count_ret *__ret =
            (struct hip_hip_get_device_count_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_get_device_count_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_GET_DEVICE_COUNT;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: int * count */
        if ((count) != (NULL)) {
            __ret->count =
                (int *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, count, (1) * sizeof(int));
        } else {
            __ret->count = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipGetDeviceCount);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_SET_DEVICE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipSetDevice = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_set_device_call *__call = (struct hip_nw_hip_set_device_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_set_device_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int deviceId */
        int deviceId;
        deviceId = __call->deviceId;
        deviceId = __call->deviceId;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipSetDevice(deviceId);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_nw_hip_set_device_ret *__ret =
            (struct hip_nw_hip_set_device_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_set_device_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_SET_DEVICE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipSetDevice);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MEM_GET_INFO:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemGetInfo = g_ptr_array_new_full(0, free);
        struct hip_hip_mem_get_info_call *__call = (struct hip_hip_mem_get_info_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_mem_get_info_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t * __free */
        size_t *__free;
        __free =
            ((__call->__free) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->__free))) : (__call->__free);
        if (__call->__free != NULL) {
            const size_t __size = ((size_t) 1);
            __free = (size_t *) calloc(__size, sizeof(size_t));
            g_ptr_array_add(__ava_alloc_list_hipMemGetInfo, __free);
        } else {
            __free = NULL;
        }

        /* Input: size_t * total */
        size_t *total;
        total =
            ((__call->total) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->total))) : (__call->total);
        if (__call->total != NULL) {
            const size_t __size = ((size_t) 1);
            total = (size_t *) calloc(__size, sizeof(size_t));
            g_ptr_array_add(__ava_alloc_list_hipMemGetInfo, total);
        } else {
            total = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemGetInfo(__free, total);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: size_t * __free */
            if ((__free) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(size_t));
            }

            /* Size: size_t * total */
            if ((total) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(size_t));
            }
        }
        struct hip_hip_mem_get_info_ret *__ret =
            (struct hip_hip_mem_get_info_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_mem_get_info_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEM_GET_INFO;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: size_t * __free */
        if ((__free) != (NULL)) {
            __ret->__free =
                (size_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, __free,
                (1) * sizeof(size_t));
        } else {
            __ret->__free = NULL;
        }

        /* Output: size_t * total */
        if ((total) != (NULL)) {
            __ret->total =
                (size_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, total,
                (1) * sizeof(size_t));
        } else {
            __ret->total = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemGetInfo);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_STREAM_CREATE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipStreamCreate = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_stream_create_call *__call = (struct hip_nw_hip_stream_create_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_stream_create_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipStream_t * stream */
        hipStream_t *stream;
        stream =
            ((__call->stream) != (NULL)) ? (((hipStream_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->stream))) : (__call->stream);
        if (__call->stream != NULL) {
            const size_t __size = ((size_t) 1);
            stream = (hipStream_t *) calloc(__size, sizeof(hipStream_t));
            g_ptr_array_add(__ava_alloc_list_nw_hipStreamCreate, stream);
        } else {
            stream = NULL;
        }

        /* Input: hsa_agent_t * agent */
        hsa_agent_t *agent;
        agent =
            ((__call->agent) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->agent))) : (__call->agent);
        if (__call->agent != NULL) {
            const size_t __size = ((size_t) 1);
            agent = (hsa_agent_t *) calloc(__size, sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list_nw_hipStreamCreate, agent);
        } else {
            agent = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipStreamCreate(stream, agent);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipStream_t * stream */
            if ((stream) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipStream_t));
            }

            /* Size: hsa_agent_t * agent */
            if ((agent) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_agent_t));
            }
        }
        struct hip_nw_hip_stream_create_ret *__ret =
            (struct hip_nw_hip_stream_create_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_stream_create_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_STREAM_CREATE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipStream_t * stream */
        if ((stream) != (NULL)) {
            __ret->stream =
                (hipStream_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, stream,
                (1) * sizeof(hipStream_t));
        } else {
            __ret->stream = NULL;
        }

        /* Output: hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            const size_t __size = 1;
            hsa_agent_t *__tmp_agent_0;
            __tmp_agent_0 = (hsa_agent_t *) calloc(__size, sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list_nw_hipStreamCreate, __tmp_agent_0);
            const size_t __agent_size_0 = (__size);
            for (size_t __agent_index_0 = 0; __agent_index_0 < __agent_size_0; __agent_index_0++) {
                const size_t ava_index = __agent_index_0;

                hsa_agent_t *__agent_a_0;
                __agent_a_0 = (__tmp_agent_0) + __agent_index_0;

                hsa_agent_t *__agent_b_0;
                __agent_b_0 = (agent) + __agent_index_0;

                hsa_agent_t *ava_self;
                ava_self = &*__agent_a_0;
                uint64_t *__agent_a_1_handle;
                __agent_a_1_handle = &(*__agent_a_0).handle;
                uint64_t *__agent_b_1_handle;
                __agent_b_1_handle = &(*__agent_b_0).handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
            }
            __ret->agent =
                (hsa_agent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_agent_0,
                (1) * sizeof(hsa_agent_t));
        } else {
            __ret->agent = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipStreamCreate); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_GET_DEVICE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipGetDevice = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_get_device_call *__call = (struct hip_nw_hip_get_device_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_get_device_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int * deviceId */
        int *deviceId;
        deviceId =
            ((__call->deviceId) != (NULL)) ? (((int *)command_channel_get_buffer(__chan, __cmd,
                    __call->deviceId))) : (__call->deviceId);
        if (__call->deviceId != NULL) {
            const size_t __size = ((size_t) 1);
            deviceId = (int *)calloc(__size, sizeof(int));
            g_ptr_array_add(__ava_alloc_list_nw_hipGetDevice, deviceId);
        } else {
            deviceId = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipGetDevice(deviceId);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: int * deviceId */
            if ((deviceId) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(int));
            }
        }
        struct hip_nw_hip_get_device_ret *__ret =
            (struct hip_nw_hip_get_device_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_get_device_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_GET_DEVICE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: int * deviceId */
        if ((deviceId) != (NULL)) {
            __ret->deviceId =
                (int *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, deviceId, (1) * sizeof(int));
        } else {
            __ret->deviceId = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipGetDevice);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_INIT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipInit = g_ptr_array_new_full(0, free);
        struct hip_hip_init_call *__call = (struct hip_hip_init_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_init_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: unsigned int flags */
        unsigned int flags;
        flags = __call->flags;
        flags = __call->flags;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipInit(flags);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_init_ret *__ret =
            (struct hip_hip_init_ret *)command_channel_new_command(__chan, sizeof(struct hip_hip_init_ret),
            __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_INIT;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipInit);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_CTX_GET_CURRENT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipCtxGetCurrent = g_ptr_array_new_full(0, free);
        struct hip_hip_ctx_get_current_call *__call = (struct hip_hip_ctx_get_current_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_ctx_get_current_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipCtx_t * ctx */
        hipCtx_t *ctx;
        ctx =
            ((__call->ctx) != (NULL)) ? (((hipCtx_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->ctx))) : (__call->ctx);
        if (__call->ctx != NULL) {
            const size_t __size = ((size_t) 1);
            ctx = (hipCtx_t *) calloc(__size, sizeof(hipCtx_t));
            g_ptr_array_add(__ava_alloc_list_hipCtxGetCurrent, ctx);
        } else {
            ctx = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipCtxGetCurrent(ctx);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipCtx_t * ctx */
            if ((ctx) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipCtx_t));
            }
        }
        struct hip_hip_ctx_get_current_ret *__ret =
            (struct hip_hip_ctx_get_current_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_ctx_get_current_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_CTX_GET_CURRENT;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipCtx_t * ctx */
        if ((ctx) != (NULL)) {
            __ret->ctx =
                (hipCtx_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, ctx,
                (1) * sizeof(hipCtx_t));
        } else {
            __ret->ctx = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipCtxGetCurrent);   /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_STREAM_SYNCHRONIZE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipStreamSynchronize = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_stream_synchronize_call *__call = (struct hip_nw_hip_stream_synchronize_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_stream_synchronize_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipStreamSynchronize(stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_nw_hip_stream_synchronize_ret *__ret =
            (struct hip_nw_hip_stream_synchronize_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_stream_synchronize_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_STREAM_SYNCHRONIZE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipStreamSynchronize);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hipGetDeviceProperties = g_ptr_array_new_full(0, free);
        struct hip___do_c_hip_get_device_properties_call *__call =
            (struct hip___do_c_hip_get_device_properties_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_hip_get_device_properties_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: char * prop */
        char *prop;
        prop =
            ((__call->prop) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->prop))) : (__call->prop);
        if (__call->prop != NULL) {
            const size_t __size = ((size_t) sizeof(hipDeviceProp_t));
            prop = (char *)calloc(__size, sizeof(char));
            g_ptr_array_add(__ava_alloc_list___do_c_hipGetDeviceProperties, prop);
        } else {
            prop = NULL;
        }

        /* Input: int deviceId */
        int deviceId;
        deviceId = __call->deviceId;
        deviceId = __call->deviceId;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper___do_c_hipGetDeviceProperties(prop, deviceId);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: char * prop */
            if ((prop) != (NULL) && (sizeof(hipDeviceProp_t)) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (sizeof(hipDeviceProp_t)) * sizeof(char));
            }
        }
        struct hip___do_c_hip_get_device_properties_ret *__ret =
            (struct hip___do_c_hip_get_device_properties_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hip_get_device_properties_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: char * prop */
        if ((prop) != (NULL) && (sizeof(hipDeviceProp_t)) > (0)) {
            __ret->prop =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, prop,
                (sizeof(hipDeviceProp_t)) * sizeof(char));
        } else {
            __ret->prop = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hipGetDeviceProperties);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchKernel = g_ptr_array_new_full(0, free);
        struct hip___do_c_hip_hcc_module_launch_kernel_call *__call =
            (struct hip___do_c_hip_hcc_module_launch_kernel_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_hip_hcc_module_launch_kernel_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_kernel_dispatch_packet_t * aql */
        hsa_kernel_dispatch_packet_t *aql;
        aql =
            ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->aql))) : (__call->aql);
        if (__call->aql != NULL) {
            const size_t __size = ((size_t) 1);
            aql = (hsa_kernel_dispatch_packet_t *) calloc(__size, sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchKernel, aql);
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 =
                ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->aql))) : (__call->aql);
            const size_t __aql_size_0 = (__size);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_b_0;
                __aql_b_0 = (__tmp_aql_0) + __aql_index_0;

                memcpy(__aql_a_0, __aql_b_0, sizeof(hsa_kernel_dispatch_packet_t));
            }
        } else {
            aql = NULL;
        }

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: void ** kernelParams */
        void **kernelParams;
        kernelParams =
            ((__call->kernelParams) != (NULL)) ? (((void **)command_channel_get_buffer(__chan, __cmd,
                    __call->kernelParams))) : (__call->kernelParams);
        if (__call->kernelParams != NULL)
            kernelParams =
                ((__call->kernelParams) != (NULL)) ? (((void **)command_channel_get_buffer(__chan, __cmd,
                        __call->kernelParams))) : (__call->kernelParams);
        else
            kernelParams = NULL;

        /* Input: size_t extra_size */
        size_t extra_size;
        extra_size = __call->extra_size;
        extra_size = __call->extra_size;

        /* Input: char * extra */
        char *extra;
        extra =
            ((__call->extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->extra))) : (__call->extra);
        if (__call->extra != NULL)
            extra =
                ((__call->extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                        __call->extra))) : (__call->extra);
        else
            extra = NULL;

        /* Input: hipEvent_t start */
        hipEvent_t start;
        start = __call->start;
        start = __call->start;

        /* Input: hipEvent_t stop */
        hipEvent_t stop;
        stop = __call->stop;
        stop = __call->stop;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper___do_c_hipHccModuleLaunchKernel(aql, stream, kernelParams, extra_size, start, extra, stop);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip___do_c_hip_hcc_module_launch_kernel_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_kernel_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hip_hcc_module_launch_kernel_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchKernel);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel = g_ptr_array_new_full(0, free);
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_call *__call =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int numKernels */
        int numKernels;
        numKernels = __call->numKernels;
        numKernels = __call->numKernels;

        /* Input: hipEvent_t * stop */
        hipEvent_t *stop;
        stop =
            ((__call->stop) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->stop))) : (__call->stop);
        if (__call->stop != NULL)
            stop =
                ((__call->stop) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->stop))) : (__call->stop);
        else
            stop = NULL;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: hsa_kernel_dispatch_packet_t * aql */
        hsa_kernel_dispatch_packet_t *aql;
        aql =
            ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->aql))) : (__call->aql);
        if (__call->aql != NULL) {
            const size_t __size = ((size_t) numKernels);
            aql = (hsa_kernel_dispatch_packet_t *) calloc(__size, sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel, aql);
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 =
                ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->aql))) : (__call->aql);
            const size_t __aql_size_0 = (__size);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_b_0;
                __aql_b_0 = (__tmp_aql_0) + __aql_index_0;
                memcpy(__aql_a_0, __aql_b_0, sizeof(hsa_kernel_dispatch_packet_t));
            }
        } else {
            aql = NULL;
        }

        /* Input: hipEvent_t * start */
        hipEvent_t *start;
        start =
            ((__call->start) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->start))) : (__call->start);
        if (__call->start != NULL)
            start =
                ((__call->start) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->start))) : (__call->start);
        else
            start = NULL;

        /* Input: size_t * extra_size */
        size_t *extra_size;
        extra_size =
            ((__call->extra_size) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->extra_size))) : (__call->extra_size);
        if (__call->extra_size != NULL)
            extra_size =
                ((__call->extra_size) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->extra_size))) : (__call->extra_size);
        else
            extra_size = NULL;

        /* Input: size_t total_extra_size */
        size_t total_extra_size;
        total_extra_size = __call->total_extra_size;
        total_extra_size = __call->total_extra_size;

        /* Input: char * all_extra */
        char *all_extra;
        all_extra =
            ((__call->all_extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->all_extra))) : (__call->all_extra);
        if (__call->all_extra != NULL)
            all_extra =
                ((__call->all_extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                        __call->all_extra))) : (__call->all_extra);
        else
            all_extra = NULL;

        /* Perform Call */
        hipError_t ret;
        ret =
            __wrapper___do_c_hipHccModuleLaunchMultiKernel(numKernels, extra_size, stop, stream, aql, start,
            total_extra_size, all_extra);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL_AND_MEMCPY:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy = g_ptr_array_new_full(0, free);
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call *__call =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size ==
            sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int numKernels */
        int numKernels;
        numKernels = __call->numKernels;
        numKernels = __call->numKernels;

        /* Input: hsa_kernel_dispatch_packet_t * aql */
        hsa_kernel_dispatch_packet_t *aql;
        aql =
            ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->aql))) : (__call->aql);
        if (__call->aql != NULL) {
            const size_t __size = ((size_t) numKernels);
            aql = (hsa_kernel_dispatch_packet_t *) calloc(__size, sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, aql);
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 =
                ((__call->aql) != (NULL)) ? (((hsa_kernel_dispatch_packet_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->aql))) : (__call->aql);
            const size_t __aql_size_0 = (__size);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_b_0;
                __aql_b_0 = (__tmp_aql_0) + __aql_index_0;
                memcpy(__aql_a_0, __aql_b_0, sizeof(hsa_kernel_dispatch_packet_t));
            }
        } else {
            aql = NULL;
        }

        /* Input: size_t * extra_size */
        size_t *extra_size;
        extra_size =
            ((__call->extra_size) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->extra_size))) : (__call->extra_size);
        if (__call->extra_size != NULL)
            extra_size =
                ((__call->extra_size) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->extra_size))) : (__call->extra_size);
        else
            extra_size = NULL;

        /* Input: hipEvent_t * stop */
        hipEvent_t *stop;
        stop =
            ((__call->stop) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->stop))) : (__call->stop);
        if (__call->stop != NULL)
            stop =
                ((__call->stop) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->stop))) : (__call->stop);
        else
            stop = NULL;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: hipEvent_t * start */
        hipEvent_t *start;
        start =
            ((__call->start) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->start))) : (__call->start);
        if (__call->start != NULL)
            start =
                ((__call->start) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->start))) : (__call->start);
        else
            start = NULL;

        /* Input: size_t total_extra_size */
        size_t total_extra_size;
        total_extra_size = __call->total_extra_size;
        total_extra_size = __call->total_extra_size;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Input: char * all_extra */
        char *all_extra;
        all_extra =
            ((__call->all_extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->all_extra))) : (__call->all_extra);
        if (__call->all_extra != NULL)
            all_extra =
                ((__call->all_extra) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                        __call->all_extra))) : (__call->all_extra);
        else
            all_extra = NULL;

        /* Input: hipMemcpyKind kind */
        hipMemcpyKind kind;
        kind = __call->kind;
        kind = __call->kind;

        /* Input: void * dst */
        void *dst;
        dst = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->dst))) : (__call->dst);
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (__call->dst != NULL) {
                const size_t __size = ((size_t) (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                dst = (void *)calloc(__size, sizeof(void));
                g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, dst);
            } else {
                dst = NULL;
            }
        } else {
            dst = __call->dst;
        }

        /* Input: const void * src */
        void *src;
        src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                    __call->src))) : (__call->src);
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
            && kind == hipMemcpyHostToDevice) {
            if (__call->src != NULL)
                src = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                            __call->src))) : (__call->src);
            else
                src = NULL;
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (__call->src != NULL) {
                    const size_t __size = ((size_t) (kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    src = (void *)calloc(__size, sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, src);
                    if (kind == hipMemcpyHostToDevice) {
                        void *__tmp_src_0;
                        __tmp_src_0 = (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__call->src) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                                    __call->src))) : (__call->src);
                        const size_t __src_size_0 = (__size);
                        for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                            const size_t ava_index = __src_index_0;

                            char *__src_a_0;
                            __src_a_0 = (src) + __src_index_0;

                            char *__src_b_0;
                            __src_b_0 = (__tmp_src_0) + __src_index_0;

                            *__src_a_0 = *__src_b_0;
                            *__src_a_0 = *__src_b_0;
                    }}
                } else {
                    src = NULL;
                }
            } else {
                src = __call->src;
            }
        }

        /* Perform Call */
        hipError_t ret;
        ret =
            __wrapper___do_c_hipHccModuleLaunchMultiKernel_and_memcpy(numKernels, aql, extra_size, stop, stream, start,
            total_extra_size, sizeBytes, all_extra, kind, dst, src);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                        && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                        __total_buffer_size +=
                            command_channel_buffer_size(__chan,
                            ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                    }
                }
        }}
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL_AND_MEMCPY;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __ret->dst =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, dst,
                    ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
            } else {
                __ret->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    const size_t __size = (kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0);
                    void *__tmp_dst_0;
                    __tmp_dst_0 = (void *)calloc(__size, sizeof(void));
                    g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, __tmp_dst_0);
                    const size_t __dst_size_0 = (__size);
                    for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                        const size_t ava_index = __dst_index_0;

                        char *__dst_a_0;
                        __dst_a_0 = (__tmp_dst_0) + __dst_index_0;

                        char *__dst_b_0;
                        __dst_b_0 = (dst) + __dst_index_0;

                        *__dst_a_0 = *__dst_b_0;
                    }
                    __ret->dst =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_dst_0,
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                } else {
                    __ret->dst = NULL;
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                    __ret->dst = dst;
                }
            }
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hsa_system_major_extension_supported = g_ptr_array_new_full(0, free);
        struct hip_nw_hsa_system_major_extension_supported_call *__call =
            (struct hip_nw_hsa_system_major_extension_supported_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hsa_system_major_extension_supported_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: uint16_t extension */
        uint16_t extension;
        extension = __call->extension;
        extension = __call->extension;

        /* Input: uint16_t version_major */
        uint16_t version_major;
        version_major = __call->version_major;
        version_major = __call->version_major;

        /* Input: uint16_t * version_minor */
        uint16_t *version_minor;
        version_minor =
            ((__call->version_minor) != (NULL)) ? (((uint16_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->version_minor))) : (__call->version_minor);
        if (__call->version_minor != NULL)
            version_minor =
                ((__call->version_minor) != (NULL)) ? (((uint16_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->version_minor))) : (__call->version_minor);
        else
            version_minor = NULL;

        /* Input: _Bool * result */
        _Bool *result;
        result =
            ((__call->result) != (NULL)) ? (((_Bool *) command_channel_get_buffer(__chan, __cmd,
                    __call->result))) : (__call->result);
        if (__call->result != NULL) {
            const size_t __size = ((size_t) 1);
            result = (_Bool *) calloc(__size, sizeof(_Bool));
            g_ptr_array_add(__ava_alloc_list_nw_hsa_system_major_extension_supported, result);
        } else {
            result = NULL;
        }

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper_nw_hsa_system_major_extension_supported(extension, version_major, version_minor, result);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: uint16_t * version_minor */
            if ((version_minor) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(uint16_t));
            }

            /* Size: _Bool * result */
            if ((result) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(_Bool));
            }
        }
        struct hip_nw_hsa_system_major_extension_supported_ret *__ret =
            (struct hip_nw_hsa_system_major_extension_supported_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hsa_system_major_extension_supported_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: uint16_t * version_minor */
        if ((version_minor) != (NULL)) {
            __ret->version_minor =
                (uint16_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, version_minor,
                (1) * sizeof(uint16_t));
        } else {
            __ret->version_minor = NULL;
        }

        /* Output: _Bool * result */
        if ((result) != (NULL)) {
            __ret->result =
                (_Bool *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, result,
                (1) * sizeof(_Bool));
        } else {
            __ret->result = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hsa_system_major_extension_supported);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HSA_EXECUTABLE_CREATE_ALT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hsa_executable_create_alt = g_ptr_array_new_full(0, free);
        struct hip_nw_hsa_executable_create_alt_call *__call = (struct hip_nw_hsa_executable_create_alt_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hsa_executable_create_alt_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_profile_t profile */
        hsa_profile_t profile;
        profile = __call->profile;
        profile = __call->profile;

        /* Input: const char * options */
        char *options;
        options =
            ((__call->options) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->options))) : (__call->options);
        if (__call->options != NULL)
            options =
                ((__call->options) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->options))) : (__call->options);
        else
            options = NULL;

        /* Input: hsa_default_float_rounding_mode_t default_float_rounding_mode */
        hsa_default_float_rounding_mode_t default_float_rounding_mode;
        default_float_rounding_mode = __call->default_float_rounding_mode;
        default_float_rounding_mode = __call->default_float_rounding_mode;

        /* Input: hsa_executable_t * executable */
        hsa_executable_t *executable;
        executable =
            ((__call->executable) != (NULL)) ? (((hsa_executable_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->executable))) : (__call->executable);
        if (__call->executable != NULL) {
            const size_t __size = ((size_t) 1);
            executable = (hsa_executable_t *) calloc(__size, sizeof(hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list_nw_hsa_executable_create_alt, executable);
        } else {
            executable = NULL;
        }

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper_nw_hsa_executable_create_alt(profile, options, default_float_rounding_mode, executable);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_executable_t * executable */
            if ((executable) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_executable_t));
            }
        }
        struct hip_nw_hsa_executable_create_alt_ret *__ret =
            (struct hip_nw_hsa_executable_create_alt_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hsa_executable_create_alt_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HSA_EXECUTABLE_CREATE_ALT;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: hsa_executable_t * executable */
        if ((executable) != (NULL)) {
            const size_t __size = 1;
            hsa_executable_t *__tmp_executable_0;
            __tmp_executable_0 = (hsa_executable_t *) calloc(__size, sizeof(hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list_nw_hsa_executable_create_alt, __tmp_executable_0);
            const size_t __executable_size_0 = (__size);
            for (size_t __executable_index_0 = 0; __executable_index_0 < __executable_size_0; __executable_index_0++) {
                const size_t ava_index = __executable_index_0;

                hsa_executable_t *__executable_a_0;
                __executable_a_0 = (__tmp_executable_0) + __executable_index_0;

                hsa_executable_t *__executable_b_0;
                __executable_b_0 = (executable) + __executable_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__executable_a_0;
                uint64_t *__executable_a_1_handle;
                __executable_a_1_handle = &(*__executable_a_0).handle;
                uint64_t *__executable_b_1_handle;
                __executable_b_1_handle = &(*__executable_b_0).handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
            }
            __ret->executable =
                (hsa_executable_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret,
                __tmp_executable_0, (1) * sizeof(hsa_executable_t));
        } else {
            __ret->executable = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hsa_executable_create_alt);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HSA_ISA_FROM_NAME:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hsa_isa_from_name = g_ptr_array_new_full(0, free);
        struct hip_nw_hsa_isa_from_name_call *__call = (struct hip_nw_hsa_isa_from_name_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hsa_isa_from_name_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_isa_t * isa */
        hsa_isa_t *isa;
        isa =
            ((__call->isa) != (NULL)) ? (((hsa_isa_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->isa))) : (__call->isa);
        if (__call->isa != NULL) {
            const size_t __size = ((size_t) 1);
            isa = (hsa_isa_t *) calloc(__size, sizeof(hsa_isa_t));
            g_ptr_array_add(__ava_alloc_list_nw_hsa_isa_from_name, isa);
        } else {
            isa = NULL;
        }

        /* Input: const char * name */
        char *name;
        name =
            ((__call->name) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->name))) : (__call->name);
        if (__call->name != NULL)
            name =
                ((__call->name) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->name))) : (__call->name);
        else
            name = NULL;

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper_nw_hsa_isa_from_name(isa, name);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_isa_t * isa */
            if ((isa) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_isa_t));
            }
        }
        struct hip_nw_hsa_isa_from_name_ret *__ret =
            (struct hip_nw_hsa_isa_from_name_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hsa_isa_from_name_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HSA_ISA_FROM_NAME;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: hsa_isa_t * isa */
        if ((isa) != (NULL)) {
            const size_t __size = 1;
            hsa_isa_t *__tmp_isa_0;
            __tmp_isa_0 = (hsa_isa_t *) calloc(__size, sizeof(hsa_isa_t));
            g_ptr_array_add(__ava_alloc_list_nw_hsa_isa_from_name, __tmp_isa_0);
            const size_t __isa_size_0 = (__size);
            for (size_t __isa_index_0 = 0; __isa_index_0 < __isa_size_0; __isa_index_0++) {
                const size_t ava_index = __isa_index_0;

                hsa_isa_t *__isa_a_0;
                __isa_a_0 = (__tmp_isa_0) + __isa_index_0;

                hsa_isa_t *__isa_b_0;
                __isa_b_0 = (isa) + __isa_index_0;

                hsa_isa_t *ava_self;
                ava_self = &*__isa_a_0;
                uint64_t *__isa_a_1_handle;
                __isa_a_1_handle = &(*__isa_a_0).handle;
                uint64_t *__isa_b_1_handle;
                __isa_b_1_handle = &(*__isa_b_0).handle;
                *__isa_a_1_handle = *__isa_b_1_handle;
            }
            __ret->isa =
                (hsa_isa_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_isa_0,
                (1) * sizeof(hsa_isa_t));
        } else {
            __ret->isa = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hsa_isa_from_name);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_PEEK_AT_LAST_ERROR:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipPeekAtLastError = g_ptr_array_new_full(0, free);
        struct hip_hip_peek_at_last_error_call *__call = (struct hip_hip_peek_at_last_error_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_peek_at_last_error_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipPeekAtLastError();

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_peek_at_last_error_ret *__ret =
            (struct hip_hip_peek_at_last_error_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_peek_at_last_error_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_PEEK_AT_LAST_ERROR;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipPeekAtLastError); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipDeviceGetAttribute = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_device_get_attribute_call *__call = (struct hip_nw_hip_device_get_attribute_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_device_get_attribute_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: int * pi */
        int *pi;
        pi = ((__call->pi) != (NULL)) ? (((int *)command_channel_get_buffer(__chan, __cmd, __call->pi))) : (__call->pi);
        if (__call->pi != NULL) {
            const size_t __size = ((size_t) 1);
            pi = (int *)calloc(__size, sizeof(int));
            g_ptr_array_add(__ava_alloc_list_nw_hipDeviceGetAttribute, pi);
        } else {
            pi = NULL;
        }

        /* Input: hipDeviceAttribute_t attr */
        hipDeviceAttribute_t attr;
        attr = __call->attr;
        attr = __call->attr;

        /* Input: int deviceId */
        int deviceId;
        deviceId = __call->deviceId;
        deviceId = __call->deviceId;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipDeviceGetAttribute(pi, attr, deviceId);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: int * pi */
            if ((pi) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(int));
            }
        }
        struct hip_nw_hip_device_get_attribute_ret *__ret =
            (struct hip_nw_hip_device_get_attribute_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_device_get_attribute_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: int * pi */
        if ((pi) != (NULL)) {
            __ret->pi =
                (int *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, pi, (1) * sizeof(int));
        } else {
            __ret->pi = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipDeviceGetAttribute);   /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MODULE_LOAD_DATA:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipModuleLoadData = g_ptr_array_new_full(0, free);
        struct hip_hip_module_load_data_call *__call = (struct hip_hip_module_load_data_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_module_load_data_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: const void * image */
        void *image;
        image =
            ((__call->image) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                    __call->image))) : (__call->image);
        if (__call->image != NULL)
            image =
                ((__call->image) != (NULL)) ? (((const void *)command_channel_get_buffer(__chan, __cmd,
                        __call->image))) : (__call->image);
        else
            image = NULL;

        /* Input: hipModule_t * module */
        hipModule_t *module;
        module =
            ((__call->module) != (NULL)) ? (((hipModule_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->module))) : (__call->module);
        if (__call->module != NULL) {
            const size_t __size = ((size_t) 1);
            module = (hipModule_t *) calloc(__size, sizeof(hipModule_t));
            g_ptr_array_add(__ava_alloc_list_hipModuleLoadData, module);
        } else {
            module = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipModuleLoadData(image, module);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipModule_t * module */
            if ((module) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipModule_t));
            }
        }
        struct hip_hip_module_load_data_ret *__ret =
            (struct hip_hip_module_load_data_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_module_load_data_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MODULE_LOAD_DATA;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipModule_t * module */
        if ((module) != (NULL)) {
            __ret->module =
                (hipModule_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, module,
                (1) * sizeof(hipModule_t));
        } else {
            __ret->module = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipModuleLoadData);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hsa_executable_symbol_get_info = g_ptr_array_new_full(0, free);
        struct hip___do_c_hsa_executable_symbol_get_info_call *__call =
            (struct hip___do_c_hsa_executable_symbol_get_info_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_hsa_executable_symbol_get_info_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_executable_symbol_t executable_symbol */
        hsa_executable_symbol_t executable_symbol;
        hsa_executable_symbol_t *ava_self;
        ava_self = &executable_symbol;
        uint64_t *__executable_symbol_a_0_handle;
        __executable_symbol_a_0_handle = &(executable_symbol).handle;
        uint64_t *__executable_symbol_b_0_handle;
        __executable_symbol_b_0_handle = &(__call->executable_symbol).handle;
        *__executable_symbol_a_0_handle = *__executable_symbol_b_0_handle;
        *__executable_symbol_a_0_handle = *__executable_symbol_b_0_handle;

        /* Input: hsa_executable_symbol_info_t attribute */
        hsa_executable_symbol_info_t attribute;
        attribute = __call->attribute;
        attribute = __call->attribute;

        /* Input: size_t max_value */
        size_t max_value;
        max_value = __call->max_value;
        max_value = __call->max_value;

        /* Input: char * value */
        char *value;
        value =
            ((__call->value) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->value))) : (__call->value);
        if (__call->value != NULL) {
            const size_t __size = ((size_t) max_value);
            value = (char *)calloc(__size, sizeof(char));
            g_ptr_array_add(__ava_alloc_list___do_c_hsa_executable_symbol_get_info, value);
        } else {
            value = NULL;
        }

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper___do_c_hsa_executable_symbol_get_info(executable_symbol, attribute, max_value, value);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: char * value */
            if ((value) != (NULL) && (max_value) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (max_value) * sizeof(char));
            }
        }
        struct hip___do_c_hsa_executable_symbol_get_info_ret *__ret =
            (struct hip___do_c_hsa_executable_symbol_get_info_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hsa_executable_symbol_get_info_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: char * value */
        if ((value) != (NULL) && (max_value) > (0)) {
            __ret->value =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, value,
                (max_value) * sizeof(char));
        } else {
            __ret->value = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hsa_executable_symbol_get_info);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_CTX_SET_CURRENT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipCtxSetCurrent = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_ctx_set_current_call *__call = (struct hip_nw_hip_ctx_set_current_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_ctx_set_current_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipCtx_t ctx */
        hipCtx_t ctx;
        ctx = __call->ctx;
        ctx = __call->ctx;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipCtxSetCurrent(ctx);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_nw_hip_ctx_set_current_ret *__ret =
            (struct hip_nw_hip_ctx_set_current_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_ctx_set_current_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_CTX_SET_CURRENT;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipCtxSetCurrent);        /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_EVENT_CREATE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipEventCreate = g_ptr_array_new_full(0, free);
        struct hip_hip_event_create_call *__call = (struct hip_hip_event_create_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_event_create_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipEvent_t * event */
        hipEvent_t *event;
        event =
            ((__call->event) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->event))) : (__call->event);
        if (__call->event != NULL) {
            const size_t __size = ((size_t) 1);
            event = (hipEvent_t *) calloc(__size, sizeof(hipEvent_t));
            g_ptr_array_add(__ava_alloc_list_hipEventCreate, event);
        } else {
            event = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipEventCreate(event);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipEvent_t * event */
            if ((event) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipEvent_t));
            }
        }
        struct hip_hip_event_create_ret *__ret =
            (struct hip_hip_event_create_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_event_create_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_EVENT_CREATE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipEvent_t * event */
        if ((event) != (NULL)) {
            __ret->event =
                (hipEvent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, event,
                (1) * sizeof(hipEvent_t));
        } else {
            __ret->event = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipEventCreate);     /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_EVENT_RECORD:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipEventRecord = g_ptr_array_new_full(0, free);
        struct hip_hip_event_record_call *__call = (struct hip_hip_event_record_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_event_record_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipEvent_t event */
        hipEvent_t event;
        event = __call->event;
        event = __call->event;

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipEventRecord(event, stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_event_record_ret *__ret =
            (struct hip_hip_event_record_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_event_record_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_EVENT_RECORD;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipEventRecord);     /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_EVENT_SYNCHRONIZE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipEventSynchronize = g_ptr_array_new_full(0, free);
        struct hip_hip_event_synchronize_call *__call = (struct hip_hip_event_synchronize_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_event_synchronize_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipEvent_t event */
        hipEvent_t event;
        event = __call->event;
        event = __call->event;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipEventSynchronize(event);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_event_synchronize_ret *__ret =
            (struct hip_hip_event_synchronize_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_event_synchronize_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_EVENT_SYNCHRONIZE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipEventSynchronize);        /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_EVENT_DESTROY:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipEventDestroy = g_ptr_array_new_full(0, free);
        struct hip_hip_event_destroy_call *__call = (struct hip_hip_event_destroy_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_event_destroy_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipEvent_t event */
        hipEvent_t event;
        event = __call->event;
        event = __call->event;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipEventDestroy(event);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_event_destroy_ret *__ret =
            (struct hip_hip_event_destroy_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_event_destroy_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_EVENT_DESTROY;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipEventDestroy);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_EVENT_ELAPSED_TIME:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipEventElapsedTime = g_ptr_array_new_full(0, free);
        struct hip_hip_event_elapsed_time_call *__call = (struct hip_hip_event_elapsed_time_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_event_elapsed_time_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: float * ms */
        float *ms;
        ms = ((__call->ms) != (NULL)) ? (((float *)command_channel_get_buffer(__chan, __cmd,
                    __call->ms))) : (__call->ms);
        if (__call->ms != NULL) {
            const size_t __size = ((size_t) 1);
            ms = (float *)calloc(__size, sizeof(float));
            g_ptr_array_add(__ava_alloc_list_hipEventElapsedTime, ms);
        } else {
            ms = NULL;
        }

        /* Input: hipEvent_t start */
        hipEvent_t start;
        start = __call->start;
        start = __call->start;

        /* Input: hipEvent_t stop */
        hipEvent_t stop;
        stop = __call->stop;
        stop = __call->stop;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipEventElapsedTime(ms, start, stop);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: float * ms */
            if ((ms) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(float));
            }
        }
        struct hip_hip_event_elapsed_time_ret *__ret =
            (struct hip_hip_event_elapsed_time_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_event_elapsed_time_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_EVENT_ELAPSED_TIME;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: float * ms */
        if ((ms) != (NULL)) {
            __ret->ms =
                (float *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, ms, (1) * sizeof(float));
        } else {
            __ret->ms = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipEventElapsedTime);        /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MODULE_LOAD:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipModuleLoad = g_ptr_array_new_full(0, free);
        struct hip_hip_module_load_call *__call = (struct hip_hip_module_load_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_module_load_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: const char * fname */
        char *fname;
        fname =
            ((__call->fname) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->fname))) : (__call->fname);
        if (__call->fname != NULL)
            fname =
                ((__call->fname) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->fname))) : (__call->fname);
        else
            fname = NULL;

        /* Input: hipModule_t * module */
        hipModule_t *module;
        module =
            ((__call->module) != (NULL)) ? (((hipModule_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->module))) : (__call->module);
        if (__call->module != NULL) {
            const size_t __size = ((size_t) 1);
            module = (hipModule_t *) calloc(__size, sizeof(hipModule_t));
            g_ptr_array_add(__ava_alloc_list_hipModuleLoad, module);
        } else {
            module = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipModuleLoad(fname, module);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipModule_t * module */
            if ((module) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipModule_t));
            }
        }
        struct hip_hip_module_load_ret *__ret =
            (struct hip_hip_module_load_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_module_load_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MODULE_LOAD;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipModule_t * module */
        if ((module) != (NULL)) {
            __ret->module =
                (hipModule_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, module,
                (1) * sizeof(hipModule_t));
        } else {
            __ret->module = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipModuleLoad);      /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MODULE_UNLOAD:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipModuleUnload = g_ptr_array_new_full(0, free);
        struct hip_hip_module_unload_call *__call = (struct hip_hip_module_unload_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_module_unload_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipModule_t module */
        hipModule_t module;
        module = __call->module;
        module = __call->module;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipModuleUnload(module);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_module_unload_ret *__ret =
            (struct hip_hip_module_unload_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_module_unload_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MODULE_UNLOAD;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipModuleUnload);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_STREAM_DESTROY:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipStreamDestroy = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_stream_destroy_call *__call = (struct hip_nw_hip_stream_destroy_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_stream_destroy_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipStreamDestroy(stream);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_nw_hip_stream_destroy_ret *__ret =
            (struct hip_nw_hip_stream_destroy_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_stream_destroy_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_STREAM_DESTROY;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipStreamDestroy);        /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_HIP_MODULE_GET_FUNCTION:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipModuleGetFunction = g_ptr_array_new_full(0, free);
        struct hip_hip_module_get_function_call *__call = (struct hip_hip_module_get_function_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_module_get_function_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipFunction_t * function */
        hipFunction_t *function;
        function =
            ((__call->function) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->function))) : (__call->function);
        if (__call->function != NULL) {
            const size_t __size = ((size_t) 1);
            function = (hipFunction_t *) calloc(__size, sizeof(hipFunction_t));
            g_ptr_array_add(__ava_alloc_list_hipModuleGetFunction, function);
        } else {
            function = NULL;
        }

        /* Input: const char * kname */
        char *kname;
        kname =
            ((__call->kname) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->kname))) : (__call->kname);
        if (__call->kname != NULL)
            kname =
                ((__call->kname) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->kname))) : (__call->kname);
        else
            kname = NULL;

        /* Input: hipModule_t module */
        hipModule_t module;
        module = __call->module;
        module = __call->module;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipModuleGetFunction(function, kname, module);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipFunction_t * function */
            if ((function) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipFunction_t));
            }
        }
        struct hip_hip_module_get_function_ret *__ret =
            (struct hip_hip_module_get_function_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_module_get_function_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MODULE_GET_FUNCTION;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipFunction_t * function */
        if ((function) != (NULL)) {
            __ret->function =
                (hipFunction_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, function,
                (1) * sizeof(hipFunction_t));
        } else {
            __ret->function = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipModuleGetFunction);       /* Deallocate all memory in the alloc list */
        break;
    }
    case CALL_HIP_HIP_GET_LAST_ERROR:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipGetLastError = g_ptr_array_new_full(0, free);
        struct hip_hip_get_last_error_call *__call = (struct hip_hip_get_last_error_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_get_last_error_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipGetLastError();

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_get_last_error_ret *__ret =
            (struct hip_hip_get_last_error_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_get_last_error_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_GET_LAST_ERROR;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipGetLastError);    /* Deallocate all memory in the alloc list */
        break;
    }
    case CALL_HIP_HIP_MEMSET:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipMemset = g_ptr_array_new_full(0, free);
        struct hip_hip_memset_call *__call = (struct hip_hip_memset_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_memset_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: void * dst */
        void *dst;
        dst = __call->dst;
        dst = __call->dst;

        /* Input: int value */
        int value;
        value = __call->value;
        value = __call->value;

        /* Input: size_t sizeBytes */
        size_t sizeBytes;
        sizeBytes = __call->sizeBytes;
        sizeBytes = __call->sizeBytes;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipMemset(dst, value, sizeBytes);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_memset_ret *__ret =
            (struct hip_hip_memset_ret *)command_channel_new_command(__chan, sizeof(struct hip_hip_memset_ret),
            __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_MEMSET;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipMemset);  /* Deallocate all memory in the alloc list */
        break;
    }
    case CALL_HIP_HIP_STREAM_WAIT_EVENT:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_hipStreamWaitEvent = g_ptr_array_new_full(0, free);
        struct hip_hip_stream_wait_event_call *__call = (struct hip_hip_stream_wait_event_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_hip_stream_wait_event_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipStream_t stream */
        hipStream_t stream;
        stream = __call->stream;
        stream = __call->stream;

        /* Input: hipEvent_t event */
        hipEvent_t event;
        event = __call->event;
        event = __call->event;

        /* Input: unsigned int flags */
        unsigned int flags;
        flags = __call->flags;
        flags = __call->flags;

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_hipStreamWaitEvent(stream, event, flags);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
        }
        struct hip_hip_stream_wait_event_ret *__ret =
            (struct hip_hip_stream_wait_event_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_hip_stream_wait_event_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_HIP_STREAM_WAIT_EVENT;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_hipStreamWaitEvent); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_HSA_AGENT_GET_INFO:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_hsa_agent_get_info = g_ptr_array_new_full(0, free);
        struct hip___do_c_hsa_agent_get_info_call *__call = (struct hip___do_c_hsa_agent_get_info_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_hsa_agent_get_info_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_agent_t agent */
        hsa_agent_t agent;
        hsa_agent_t *ava_self;
        ava_self = &agent;
        uint64_t *__agent_a_0_handle;
        __agent_a_0_handle = &(agent).handle;
        uint64_t *__agent_b_0_handle;
        __agent_b_0_handle = &(__call->agent).handle;
        *__agent_a_0_handle = *__agent_b_0_handle;
        *__agent_a_0_handle = *__agent_b_0_handle;

        /* Input: hsa_agent_info_t attribute */
        hsa_agent_info_t attribute;
        attribute = __call->attribute;
        attribute = __call->attribute;

        /* Input: size_t max_value */
        size_t max_value;
        max_value = __call->max_value;
        max_value = __call->max_value;

        /* Input: void * value */
        void *value;
        value =
            ((__call->value) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                    __call->value))) : (__call->value);
        if (__call->value != NULL) {
            const size_t __size = ((size_t) max_value);
            value = (void *)calloc(__size, sizeof(void));
            g_ptr_array_add(__ava_alloc_list___do_c_hsa_agent_get_info, value);
        } else {
            value = NULL;
        }

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper___do_c_hsa_agent_get_info(agent, attribute, max_value, value);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: void * value */
            if ((value) != (NULL) && (max_value) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (max_value) * sizeof(void));
            }
        }
        struct hip___do_c_hsa_agent_get_info_ret *__ret =
            (struct hip___do_c_hsa_agent_get_info_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_hsa_agent_get_info_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_HSA_AGENT_GET_INFO;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: void * value */
        if ((value) != (NULL) && (max_value) > (0)) {
            __ret->value =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, value,
                (max_value) * sizeof(void));
        } else {
            __ret->value = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_hsa_agent_get_info);  /* Deallocate all memory in the alloc list */
        break;
    }
    case CALL_HIP___DO_C_LOAD_EXECUTABLE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_load_executable = g_ptr_array_new_full(0, free);
        struct hip___do_c_load_executable_call *__call = (struct hip___do_c_load_executable_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_load_executable_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t file_len */
        size_t file_len;
        file_len = __call->file_len;
        file_len = __call->file_len;

        /* Input: const char * file_buf */
        char *file_buf;
        file_buf =
            ((__call->file_buf) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->file_buf))) : (__call->file_buf);
        if (__call->file_buf != NULL)
            file_buf =
                ((__call->file_buf) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->file_buf))) : (__call->file_buf);
        else
            file_buf = NULL;

        /* Input: hsa_executable_t * executable */
        hsa_executable_t *executable;
        executable =
            ((__call->executable) != (NULL)) ? (((hsa_executable_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->executable))) : (__call->executable);
        if (__call->executable != NULL) {
            const size_t __size = ((size_t) 1);
            executable = (hsa_executable_t *) calloc(__size, sizeof(hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list___do_c_load_executable, executable);
            hsa_executable_t *__tmp_executable_0;
            __tmp_executable_0 =
                ((__call->executable) != (NULL)) ? (((hsa_executable_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->executable))) : (__call->executable);
            const size_t __executable_size_0 = (__size);
            for (size_t __executable_index_0 = 0; __executable_index_0 < __executable_size_0; __executable_index_0++) {
                const size_t ava_index = __executable_index_0;

                hsa_executable_t *__executable_a_0;
                __executable_a_0 = (executable) + __executable_index_0;

                hsa_executable_t *__executable_b_0;
                __executable_b_0 = (__tmp_executable_0) + __executable_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__executable_a_0;
                uint64_t *__executable_a_1_handle;
                __executable_a_1_handle = &(*__executable_a_0).handle;
                uint64_t *__executable_b_1_handle;
                __executable_b_1_handle = &(*__executable_b_0).handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
            }
        } else {
            executable = NULL;
        }

        /* Input: hsa_agent_t * agent */
        hsa_agent_t *agent;
        agent =
            ((__call->agent) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->agent))) : (__call->agent);
        if (__call->agent != NULL) {
            const size_t __size = ((size_t) 1);
            agent = (hsa_agent_t *) calloc(__size, sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_load_executable, agent);
            hsa_agent_t *__tmp_agent_0;
            __tmp_agent_0 =
                ((__call->agent) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                        __call->agent))) : (__call->agent);
            const size_t __agent_size_0 = (__size);
            for (size_t __agent_index_0 = 0; __agent_index_0 < __agent_size_0; __agent_index_0++) {
                const size_t ava_index = __agent_index_0;

                hsa_agent_t *__agent_a_0;
                __agent_a_0 = (agent) + __agent_index_0;

                hsa_agent_t *__agent_b_0;
                __agent_b_0 = (__tmp_agent_0) + __agent_index_0;

                hsa_agent_t *ava_self;
                ava_self = &*__agent_a_0;
                uint64_t *__agent_a_1_handle;
                __agent_a_1_handle = &(*__agent_a_0).handle;
                uint64_t *__agent_b_1_handle;
                __agent_b_1_handle = &(*__agent_b_0).handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
            }
        } else {
            agent = NULL;
        }

        /* Perform Call */
        int ret;
        ret = __wrapper___do_c_load_executable(file_len, file_buf, executable, agent);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_executable_t * executable */
            if ((executable) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_executable_t));
            }
        }
        struct hip___do_c_load_executable_ret *__ret =
            (struct hip___do_c_load_executable_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_load_executable_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_LOAD_EXECUTABLE;
        __ret->__call_id = __call->__call_id;

        /* Output: int ret */
        __ret->ret = ret;

        /* Output: hsa_executable_t * executable */
        if ((executable) != (NULL)) {
            const size_t __size = 1;
            hsa_executable_t *__tmp_executable_0;
            __tmp_executable_0 = (hsa_executable_t *) calloc(__size, sizeof(hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list___do_c_load_executable, __tmp_executable_0);
            const size_t __executable_size_0 = (__size);
            for (size_t __executable_index_0 = 0; __executable_index_0 < __executable_size_0; __executable_index_0++) {
                const size_t ava_index = __executable_index_0;

                hsa_executable_t *__executable_a_0;
                __executable_a_0 = (__tmp_executable_0) + __executable_index_0;

                hsa_executable_t *__executable_b_0;
                __executable_b_0 = (executable) + __executable_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__executable_a_0;
                uint64_t *__executable_a_1_handle;
                __executable_a_1_handle = &(*__executable_a_0).handle;
                uint64_t *__executable_b_1_handle;
                __executable_b_1_handle = &(*__executable_b_0).handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
            }
            __ret->executable =
                (hsa_executable_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret,
                __tmp_executable_0, (1) * sizeof(hsa_executable_t));
        } else {
            __ret->executable = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_load_executable);     /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_GET_AGENTS:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_get_agents = g_ptr_array_new_full(0, free);
        struct hip___do_c_get_agents_call *__call = (struct hip___do_c_get_agents_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_get_agents_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t max_agents */
        size_t max_agents;
        max_agents = __call->max_agents;
        max_agents = __call->max_agents;

        /* Input: hsa_agent_t * agents */
        hsa_agent_t *agents;
        agents =
            ((__call->agents) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->agents))) : (__call->agents);
        if (__call->agents != NULL) {
            const size_t __size = ((size_t) max_agents);
            agents = (hsa_agent_t *) calloc(__size, sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_agents, agents);
        } else {
            agents = NULL;
        }

        /* Perform Call */
        size_t ret;
        ret = __wrapper___do_c_get_agents(max_agents, agents);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_agent_t * agents */
            if ((agents) != (NULL) && (max_agents) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (max_agents) * sizeof(hsa_agent_t));
            }
        }
        struct hip___do_c_get_agents_ret *__ret =
            (struct hip___do_c_get_agents_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_get_agents_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_GET_AGENTS;
        __ret->__call_id = __call->__call_id;

        /* Output: size_t ret */
        __ret->ret = ret;

        /* Output: hsa_agent_t * agents */
        if ((agents) != (NULL) && (max_agents) > (0)) {
            const size_t __size = max_agents;
            hsa_agent_t *__tmp_agents_0;
            __tmp_agents_0 = (hsa_agent_t *) calloc(__size, sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_agents, __tmp_agents_0);
            const size_t __agents_size_0 = (__size);
            for (size_t __agents_index_0 = 0; __agents_index_0 < __agents_size_0; __agents_index_0++) {
                const size_t ava_index = __agents_index_0;

                hsa_agent_t *__agents_a_0;
                __agents_a_0 = (__tmp_agents_0) + __agents_index_0;

                hsa_agent_t *__agents_b_0;
                __agents_b_0 = (agents) + __agents_index_0;

                hsa_agent_t *ava_self;
                ava_self = &*__agents_a_0;
                uint64_t *__agents_a_1_handle;
                __agents_a_1_handle = &(*__agents_a_0).handle;
                uint64_t *__agents_b_1_handle;
                __agents_b_1_handle = &(*__agents_b_0).handle;
                *__agents_a_1_handle = *__agents_b_1_handle;
            }
            __ret->agents =
                (hsa_agent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_agents_0,
                (max_agents) * sizeof(hsa_agent_t));
        } else {
            __ret->agents = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_get_agents);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_GET_ISAS:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_get_isas = g_ptr_array_new_full(0, free);
        struct hip___do_c_get_isas_call *__call = (struct hip___do_c_get_isas_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_get_isas_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hsa_agent_t agents */
        hsa_agent_t agents;
        hsa_agent_t *ava_self;
        ava_self = &agents;
        uint64_t *__agents_a_0_handle;
        __agents_a_0_handle = &(agents).handle;
        uint64_t *__agents_b_0_handle;
        __agents_b_0_handle = &(__call->agents).handle;
        *__agents_a_0_handle = *__agents_b_0_handle;
        *__agents_a_0_handle = *__agents_b_0_handle;

        /* Input: size_t max_isas */
        size_t max_isas;
        max_isas = __call->max_isas;
        max_isas = __call->max_isas;

        /* Input: hsa_isa_t * isas */
        hsa_isa_t *isas;
        isas =
            ((__call->isas) != (NULL)) ? (((hsa_isa_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->isas))) : (__call->isas);
        if (__call->isas != NULL) {
            const size_t __size = ((size_t) max_isas);
            isas = (hsa_isa_t *) calloc(__size, sizeof(hsa_isa_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_isas, isas);
        } else {
            isas = NULL;
        }

        /* Perform Call */
        size_t ret;
        ret = __wrapper___do_c_get_isas(agents, max_isas, isas);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_isa_t * isas */
            if ((isas) != (NULL) && (max_isas) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (max_isas) * sizeof(hsa_isa_t));
            }
        }
        struct hip___do_c_get_isas_ret *__ret =
            (struct hip___do_c_get_isas_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_get_isas_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_GET_ISAS;
        __ret->__call_id = __call->__call_id;

        /* Output: size_t ret */
        __ret->ret = ret;

        /* Output: hsa_isa_t * isas */
        if ((isas) != (NULL) && (max_isas) > (0)) {
            const size_t __size = max_isas;
            hsa_isa_t *__tmp_isas_0;
            __tmp_isas_0 = (hsa_isa_t *) calloc(__size, sizeof(hsa_isa_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_isas, __tmp_isas_0);
            const size_t __isas_size_0 = (__size);
            for (size_t __isas_index_0 = 0; __isas_index_0 < __isas_size_0; __isas_index_0++) {
                const size_t ava_index = __isas_index_0;

                hsa_isa_t *__isas_a_0;
                __isas_a_0 = (__tmp_isas_0) + __isas_index_0;

                hsa_isa_t *__isas_b_0;
                __isas_b_0 = (isas) + __isas_index_0;

                hsa_isa_t *ava_self;
                ava_self = &*__isas_a_0;
                uint64_t *__isas_a_1_handle;
                __isas_a_1_handle = &(*__isas_a_0).handle;
                uint64_t *__isas_b_1_handle;
                __isas_b_1_handle = &(*__isas_b_0).handle;
                *__isas_a_1_handle = *__isas_b_1_handle;
            }
            __ret->isas =
                (hsa_isa_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_isas_0,
                (max_isas) * sizeof(hsa_isa_t));
        } else {
            __ret->isas = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_get_isas);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_GET_KERENEL_SYMBOLS:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_get_kerenel_symbols = g_ptr_array_new_full(0, free);
        struct hip___do_c_get_kerenel_symbols_call *__call = (struct hip___do_c_get_kerenel_symbols_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_get_kerenel_symbols_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: const hsa_executable_t * exec */
        hsa_executable_t *exec;
        exec =
            ((__call->exec) != (NULL)) ? (((const hsa_executable_t *)command_channel_get_buffer(__chan, __cmd,
                    __call->exec))) : (__call->exec);
        if (__call->exec != NULL) {
            const size_t __size = ((size_t) 1);
            exec = (hsa_executable_t *) calloc(__size, sizeof(const hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, exec);
            hsa_executable_t *__tmp_exec_0;
            __tmp_exec_0 =
                ((__call->exec) != (NULL)) ? (((const hsa_executable_t *)command_channel_get_buffer(__chan, __cmd,
                        __call->exec))) : (__call->exec);
            const size_t __exec_size_0 = (__size);
            for (size_t __exec_index_0 = 0; __exec_index_0 < __exec_size_0; __exec_index_0++) {
                const size_t ava_index = __exec_index_0;

                hsa_executable_t *__exec_a_0;
                __exec_a_0 = (exec) + __exec_index_0;

                hsa_executable_t *__exec_b_0;
                __exec_b_0 = (__tmp_exec_0) + __exec_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__exec_a_0;
                uint64_t *__exec_a_1_handle;
                __exec_a_1_handle = &(*__exec_a_0).handle;
                uint64_t *__exec_b_1_handle;
                __exec_b_1_handle = &(*__exec_b_0).handle;
                *__exec_a_1_handle = *__exec_b_1_handle;
                *__exec_a_1_handle = *__exec_b_1_handle;
            }
        } else {
            exec = NULL;
        }

        /* Input: const hsa_agent_t * agent */
        hsa_agent_t *agent;
        agent =
            ((__call->agent) != (NULL)) ? (((const hsa_agent_t *)command_channel_get_buffer(__chan, __cmd,
                    __call->agent))) : (__call->agent);
        if (__call->agent != NULL) {
            const size_t __size = ((size_t) 1);
            agent = (hsa_agent_t *) calloc(__size, sizeof(const hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, agent);
            hsa_agent_t *__tmp_agent_0;
            __tmp_agent_0 =
                ((__call->agent) != (NULL)) ? (((const hsa_agent_t *)command_channel_get_buffer(__chan, __cmd,
                        __call->agent))) : (__call->agent);
            const size_t __agent_size_0 = (__size);
            for (size_t __agent_index_0 = 0; __agent_index_0 < __agent_size_0; __agent_index_0++) {
                const size_t ava_index = __agent_index_0;

                hsa_agent_t *__agent_a_0;
                __agent_a_0 = (agent) + __agent_index_0;

                hsa_agent_t *__agent_b_0;
                __agent_b_0 = (__tmp_agent_0) + __agent_index_0;

                hsa_agent_t *ava_self;
                ava_self = &*__agent_a_0;
                uint64_t *__agent_a_1_handle;
                __agent_a_1_handle = &(*__agent_a_0).handle;
                uint64_t *__agent_b_1_handle;
                __agent_b_1_handle = &(*__agent_b_0).handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
            }
        } else {
            agent = NULL;
        }

        /* Input: size_t max_symbols */
        size_t max_symbols;
        max_symbols = __call->max_symbols;
        max_symbols = __call->max_symbols;

        /* Input: hsa_executable_symbol_t * symbols */
        hsa_executable_symbol_t *symbols;
        symbols =
            ((__call->symbols) != (NULL)) ? (((hsa_executable_symbol_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->symbols))) : (__call->symbols);
        if (__call->symbols != NULL) {
            const size_t __size = ((size_t) max_symbols);
            symbols = (hsa_executable_symbol_t *) calloc(__size, sizeof(hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, symbols);
        } else {
            symbols = NULL;
        }

        /* Perform Call */
        size_t ret;
        ret = __wrapper___do_c_get_kerenel_symbols(exec, agent, max_symbols, symbols);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hsa_executable_symbol_t * symbols */
            if ((symbols) != (NULL) && (max_symbols) > (0)) {
                __total_buffer_size +=
                    command_channel_buffer_size(__chan, (max_symbols) * sizeof(hsa_executable_symbol_t));
            }
        }
        struct hip___do_c_get_kerenel_symbols_ret *__ret =
            (struct hip___do_c_get_kerenel_symbols_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_get_kerenel_symbols_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_GET_KERENEL_SYMBOLS;
        __ret->__call_id = __call->__call_id;

        /* Output: size_t ret */
        __ret->ret = ret;

        /* Output: hsa_executable_symbol_t * symbols */
        if ((symbols) != (NULL) && (max_symbols) > (0)) {
            const size_t __size = max_symbols;
            hsa_executable_symbol_t *__tmp_symbols_0;
            __tmp_symbols_0 = (hsa_executable_symbol_t *) calloc(__size, sizeof(hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, __tmp_symbols_0);
            const size_t __symbols_size_0 = (__size);
            for (size_t __symbols_index_0 = 0; __symbols_index_0 < __symbols_size_0; __symbols_index_0++) {
                const size_t ava_index = __symbols_index_0;

                hsa_executable_symbol_t *__symbols_a_0;
                __symbols_a_0 = (__tmp_symbols_0) + __symbols_index_0;

                hsa_executable_symbol_t *__symbols_b_0;
                __symbols_b_0 = (symbols) + __symbols_index_0;

                hsa_executable_symbol_t *ava_self;
                ava_self = &*__symbols_a_0;
                uint64_t *__symbols_a_1_handle;
                __symbols_a_1_handle = &(*__symbols_a_0).handle;
                uint64_t *__symbols_b_1_handle;
                __symbols_b_1_handle = &(*__symbols_b_0).handle;
                *__symbols_a_1_handle = *__symbols_b_1_handle;
            }
            __ret->symbols =
                (hsa_executable_symbol_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret,
                __tmp_symbols_0, (max_symbols) * sizeof(hsa_executable_symbol_t));
        } else {
            __ret->symbols = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_get_kerenel_symbols); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_QUERY_HOST_ADDRESS:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_query_host_address = g_ptr_array_new_full(0, free);
        struct hip___do_c_query_host_address_call *__call = (struct hip___do_c_query_host_address_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_query_host_address_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: uint64_t kernel_object_ */
        uint64_t kernel_object_;
        kernel_object_ = __call->kernel_object_;
        kernel_object_ = __call->kernel_object_;

        /* Input: char * kernel_header_ */
        char *kernel_header_;
        kernel_header_ =
            ((__call->kernel_header_) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->kernel_header_))) : (__call->kernel_header_);
        if (__call->kernel_header_ != NULL) {
            const size_t __size = ((size_t) sizeof(amd_kernel_code_t));
            kernel_header_ = (char *)calloc(__size, sizeof(char));
            g_ptr_array_add(__ava_alloc_list___do_c_query_host_address, kernel_header_);
        } else {
            kernel_header_ = NULL;
        }

        /* Perform Call */
        hsa_status_t ret;
        ret = __wrapper___do_c_query_host_address(kernel_object_, kernel_header_);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: char * kernel_header_ */
            if ((kernel_header_) != (NULL) && (sizeof(amd_kernel_code_t)) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (sizeof(amd_kernel_code_t)) * sizeof(char));
            }
        }
        struct hip___do_c_query_host_address_ret *__ret =
            (struct hip___do_c_query_host_address_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_query_host_address_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_QUERY_HOST_ADDRESS;
        __ret->__call_id = __call->__call_id;

        /* Output: hsa_status_t ret */
        __ret->ret = ret;

        /* Output: char * kernel_header_ */
        if ((kernel_header_) != (NULL) && (sizeof(amd_kernel_code_t)) > (0)) {
            __ret->kernel_header_ =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, kernel_header_,
                (sizeof(amd_kernel_code_t)) * sizeof(char));
        } else {
            __ret->kernel_header_ = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_query_host_address);  /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_GET_KERNEL_DESCRIPTOR:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_get_kernel_descriptor = g_ptr_array_new_full(0, free);
        struct hip___do_c_get_kernel_descriptor_call *__call = (struct hip___do_c_get_kernel_descriptor_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_get_kernel_descriptor_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: const char * name */
        char *name;
        name =
            ((__call->name) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                    __call->name))) : (__call->name);
        if (__call->name != NULL)
            name =
                ((__call->name) != (NULL)) ? (((const char *)command_channel_get_buffer(__chan, __cmd,
                        __call->name))) : (__call->name);
        else
            name = NULL;

        /* Input: const hsa_executable_symbol_t * symbol */
        hsa_executable_symbol_t *symbol;
        symbol =
            ((__call->symbol) != (NULL)) ? (((const hsa_executable_symbol_t *)command_channel_get_buffer(__chan, __cmd,
                    __call->symbol))) : (__call->symbol);
        if (__call->symbol != NULL) {
            const size_t __size = ((size_t) 1);
            symbol = (hsa_executable_symbol_t *) calloc(__size, sizeof(const hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kernel_descriptor, symbol);
            hsa_executable_symbol_t *__tmp_symbol_0;
            __tmp_symbol_0 =
                ((__call->symbol) != (NULL)) ? (((const hsa_executable_symbol_t *)command_channel_get_buffer(__chan,
                        __cmd, __call->symbol))) : (__call->symbol);
            const size_t __symbol_size_0 = (__size);
            for (size_t __symbol_index_0 = 0; __symbol_index_0 < __symbol_size_0; __symbol_index_0++) {
                const size_t ava_index = __symbol_index_0;

                hsa_executable_symbol_t *__symbol_a_0;
                __symbol_a_0 = (symbol) + __symbol_index_0;

                hsa_executable_symbol_t *__symbol_b_0;
                __symbol_b_0 = (__tmp_symbol_0) + __symbol_index_0;

                hsa_executable_symbol_t *ava_self;
                ava_self = &*__symbol_a_0;
                uint64_t *__symbol_a_1_handle;
                __symbol_a_1_handle = &(*__symbol_a_0).handle;
                uint64_t *__symbol_b_1_handle;
                __symbol_b_1_handle = &(*__symbol_b_0).handle;
                *__symbol_a_1_handle = *__symbol_b_1_handle;
                *__symbol_a_1_handle = *__symbol_b_1_handle;
            }
        } else {
            symbol = NULL;
        }

        /* Input: hipFunction_t * f */
        hipFunction_t *f;
        f = ((__call->f) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->f))) : (__call->f);
        if (__call->f != NULL) {
            const size_t __size = ((size_t) 1);
            f = (hipFunction_t *) calloc(__size, sizeof(hipFunction_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kernel_descriptor, f);
        } else {
            f = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper___do_c_get_kernel_descriptor(name, symbol, f);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipFunction_t * f */
            if ((f) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipFunction_t));
            }
        }
        struct hip___do_c_get_kernel_descriptor_ret *__ret =
            (struct hip___do_c_get_kernel_descriptor_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_get_kernel_descriptor_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_GET_KERNEL_DESCRIPTOR;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipFunction_t * f */
        if ((f) != (NULL)) {
            __ret->f =
                (hipFunction_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, f,
                (1) * sizeof(hipFunction_t));
        } else {
            __ret->f = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_get_kernel_descriptor);       /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_HIP_CTX_GET_DEVICE:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_hipCtxGetDevice = g_ptr_array_new_full(0, free);
        struct hip_nw_hip_ctx_get_device_call *__call = (struct hip_nw_hip_ctx_get_device_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_hip_ctx_get_device_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipDevice_t * device */
        hipDevice_t *device;
        device =
            ((__call->device) != (NULL)) ? (((hipDevice_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->device))) : (__call->device);
        if (__call->device != NULL) {
            const size_t __size = ((size_t) 1);
            device = (hipDevice_t *) calloc(__size, sizeof(hipDevice_t));
            g_ptr_array_add(__ava_alloc_list_nw_hipCtxGetDevice, device);
        } else {
            device = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_hipCtxGetDevice(device);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: hipDevice_t * device */
            if ((device) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hipDevice_t));
            }
        }
        struct hip_nw_hip_ctx_get_device_ret *__ret =
            (struct hip_nw_hip_ctx_get_device_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_hip_ctx_get_device_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_HIP_CTX_GET_DEVICE;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: hipDevice_t * device */
        if ((device) != (NULL)) {
            __ret->device =
                (hipDevice_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, device,
                (1) * sizeof(hipDevice_t));
        } else {
            __ret->device = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_hipCtxGetDevice); /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP_NW_LOOKUP_KERN_INFO:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list_nw_lookup_kern_info = g_ptr_array_new_full(0, free);
        struct hip_nw_lookup_kern_info_call *__call = (struct hip_nw_lookup_kern_info_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip_nw_lookup_kern_info_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: hipFunction_t f */
        hipFunction_t f;
        f = __call->f;
        f = __call->f;

        /* Input: struct nw_kern_info * info */
        struct nw_kern_info *info;
        info =
            ((__call->info) != (NULL)) ? (((struct nw_kern_info *)command_channel_get_buffer(__chan, __cmd,
                    __call->info))) : (__call->info);
        if (__call->info != NULL) {
            const size_t __size = ((size_t) 1);
            info = (struct nw_kern_info *)calloc(__size, sizeof(struct nw_kern_info));
            g_ptr_array_add(__ava_alloc_list_nw_lookup_kern_info, info);
        } else {
            info = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper_nw_lookup_kern_info(f, info);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: struct nw_kern_info * info */
            if ((info) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(struct nw_kern_info));
            }
        }
        struct hip_nw_lookup_kern_info_ret *__ret =
            (struct hip_nw_lookup_kern_info_ret *)command_channel_new_command(__chan,
            sizeof(struct hip_nw_lookup_kern_info_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP_NW_LOOKUP_KERN_INFO;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: struct nw_kern_info * info */
        if ((info) != (NULL)) {
            const size_t __size = 1;
            struct nw_kern_info *__tmp_info_0;
            __tmp_info_0 = (struct nw_kern_info *)calloc(__size, sizeof(struct nw_kern_info));
            g_ptr_array_add(__ava_alloc_list_nw_lookup_kern_info, __tmp_info_0);
            const size_t __info_size_0 = (__size);
            for (size_t __info_index_0 = 0; __info_index_0 < __info_size_0; __info_index_0++) {
                const size_t ava_index = __info_index_0;

                struct nw_kern_info *__info_a_0;
                __info_a_0 = (__tmp_info_0) + __info_index_0;

                struct nw_kern_info *__info_b_0;
                __info_b_0 = (info) + __info_index_0;

                struct nw_kern_info *ava_self;
                ava_self = &*__info_a_0;
                uint64_t *__info_a_1__object;
                __info_a_1__object = &(*__info_a_0)._object;
                uint64_t *__info_b_1__object;
                __info_b_1__object = &(*__info_b_0)._object;
                *__info_a_1__object = *__info_b_1__object;
                uint64_t *__info_a_1_workgroup_group_segment_byte_size;
                __info_a_1_workgroup_group_segment_byte_size = &(*__info_a_0).workgroup_group_segment_byte_size;
                uint64_t *__info_b_1_workgroup_group_segment_byte_size;
                __info_b_1_workgroup_group_segment_byte_size = &(*__info_b_0).workgroup_group_segment_byte_size;
                *__info_a_1_workgroup_group_segment_byte_size = *__info_b_1_workgroup_group_segment_byte_size;
                uint64_t *__info_a_1_workitem_private_segment_byte_size;
                __info_a_1_workitem_private_segment_byte_size = &(*__info_a_0).workitem_private_segment_byte_size;
                uint64_t *__info_b_1_workitem_private_segment_byte_size;
                __info_b_1_workitem_private_segment_byte_size = &(*__info_b_0).workitem_private_segment_byte_size;
                *__info_a_1_workitem_private_segment_byte_size = *__info_b_1_workitem_private_segment_byte_size;
            }
            __ret->info =
                (struct nw_kern_info *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, __tmp_info_0,
                (1) * sizeof(struct nw_kern_info));
        } else {
            __ret->info = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list_nw_lookup_kern_info);        /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case CALL_HIP___DO_C_MASS_SYMBOL_INFO:{
        ava_is_in = 1;
        ava_is_out = 0;
        GPtrArray *__ava_alloc_list___do_c_mass_symbol_info = g_ptr_array_new_full(0, free);
        struct hip___do_c_mass_symbol_info_call *__call = (struct hip___do_c_mass_symbol_info_call *)__cmd;
        assert(__call->base.api_id == HIP_API);
        assert(__call->base.command_size == sizeof(struct hip___do_c_mass_symbol_info_call));
#ifdef AVA_RECORD_REPLAY

#endif

        /* Unpack and translate arguments */

        /* Input: size_t n */
        size_t n;
        n = __call->n;
        n = __call->n;

        /* Input: unsigned int * offsets */
        unsigned int *offsets;
        offsets =
            ((__call->offsets) != (NULL)) ? (((unsigned int *)command_channel_get_buffer(__chan, __cmd,
                    __call->offsets))) : (__call->offsets);
        if (__call->offsets != NULL) {
            const size_t __size = ((size_t) n);
            offsets = (unsigned int *)calloc(__size, sizeof(unsigned int));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, offsets);
        } else {
            offsets = NULL;
        }

        /* Input: size_t pool_size */
        size_t pool_size;
        pool_size = __call->pool_size;
        pool_size = __call->pool_size;

        /* Input: uint8_t * agents */
        uint8_t *agents;
        agents =
            ((__call->agents) != (NULL)) ? (((uint8_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->agents))) : (__call->agents);
        if (__call->agents != NULL) {
            const size_t __size = ((size_t) n * sizeof(agents));
            agents = (uint8_t *) calloc(__size, sizeof(uint8_t));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, agents);
        } else {
            agents = NULL;
        }

        /* Input: hsa_symbol_kind_t * types */
        hsa_symbol_kind_t *types;
        types =
            ((__call->types) != (NULL)) ? (((hsa_symbol_kind_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->types))) : (__call->types);
        if (__call->types != NULL) {
            const size_t __size = ((size_t) n);
            types = (hsa_symbol_kind_t *) calloc(__size, sizeof(hsa_symbol_kind_t));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, types);
        } else {
            types = NULL;
        }

        /* Input: hipFunction_t * descriptors */
        hipFunction_t *descriptors;
        descriptors =
            ((__call->descriptors) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan, __cmd,
                    __call->descriptors))) : (__call->descriptors);
        if (__call->descriptors != NULL) {
            const size_t __size = ((size_t) n);
            descriptors = (hipFunction_t *) calloc(__size, sizeof(hipFunction_t));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, descriptors);
        } else {
            descriptors = NULL;
        }

        /* Input: const hsa_executable_symbol_t * syms */
        hsa_executable_symbol_t *syms;
        syms =
            ((__call->syms) != (NULL)) ? (((const hsa_executable_symbol_t *)command_channel_get_buffer(__chan, __cmd,
                    __call->syms))) : (__call->syms);
        if (__call->syms != NULL) {
            const size_t __size = ((size_t) n);
            syms = (hsa_executable_symbol_t *) calloc(__size, sizeof(const hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, syms);
            hsa_executable_symbol_t *__tmp_syms_0;
            __tmp_syms_0 =
                ((__call->syms) != (NULL)) ? (((const hsa_executable_symbol_t *)command_channel_get_buffer(__chan,
                        __cmd, __call->syms))) : (__call->syms);
            const size_t __syms_size_0 = (__size);
            for (size_t __syms_index_0 = 0; __syms_index_0 < __syms_size_0; __syms_index_0++) {
                const size_t ava_index = __syms_index_0;

                hsa_executable_symbol_t *__syms_a_0;
                __syms_a_0 = (syms) + __syms_index_0;

                hsa_executable_symbol_t *__syms_b_0;
                __syms_b_0 = (__tmp_syms_0) + __syms_index_0;

                hsa_executable_symbol_t *ava_self;
                ava_self = &*__syms_a_0;
                uint64_t *__syms_a_1_handle;
                __syms_a_1_handle = &(*__syms_a_0).handle;
                uint64_t *__syms_b_1_handle;
                __syms_b_1_handle = &(*__syms_b_0).handle;
                *__syms_a_1_handle = *__syms_b_1_handle;
                *__syms_a_1_handle = *__syms_b_1_handle;
            }
        } else {
            syms = NULL;
        }

        /* Input: char * pool */
        char *pool;
        pool =
            ((__call->pool) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                    __call->pool))) : (__call->pool);
        if (__call->pool != NULL) {
            const size_t __size = ((size_t) pool_size);
            pool = (char *)calloc(__size, sizeof(char));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, pool);
        } else {
            pool = NULL;
        }

        /* Perform Call */
        hipError_t ret;
        ret = __wrapper___do_c_mass_symbol_info(n, pool_size, agents, types, descriptors, syms, offsets, pool);

        ava_is_in = 0;
        ava_is_out = 1;
        size_t __total_buffer_size = 0; {
            /* Size: unsigned int * offsets */
            if ((offsets) != (NULL) && (n) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (n) * sizeof(unsigned int));
            }

            /* Size: uint8_t * agents */
            if ((agents) != (NULL) && (n * sizeof(agents)) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (n * sizeof(agents)) * sizeof(uint8_t));
            }

            /* Size: hsa_symbol_kind_t * types */
            if ((types) != (NULL) && (n) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (n) * sizeof(hsa_symbol_kind_t));
            }

            /* Size: hipFunction_t * descriptors */
            if ((descriptors) != (NULL) && (n) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (n) * sizeof(hipFunction_t));
            }

            /* Size: char * pool */
            if ((pool) != (NULL) && (pool_size) > (0)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (pool_size) * sizeof(char));
            }
        }
        struct hip___do_c_mass_symbol_info_ret *__ret =
            (struct hip___do_c_mass_symbol_info_ret *)command_channel_new_command(__chan,
            sizeof(struct hip___do_c_mass_symbol_info_ret), __total_buffer_size);
        __ret->base.api_id = HIP_API;
        __ret->base.command_id = RET_HIP___DO_C_MASS_SYMBOL_INFO;
        __ret->__call_id = __call->__call_id;

        /* Output: hipError_t ret */
        __ret->ret = ret;

        /* Output: unsigned int * offsets */
        if ((offsets) != (NULL) && (n) > (0)) {
            __ret->offsets =
                (unsigned int *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, offsets,
                (n) * sizeof(unsigned int));
        } else {
            __ret->offsets = NULL;
        }

        /* Output: uint8_t * agents */
        if ((agents) != (NULL) && (n * sizeof(agents)) > (0)) {
            __ret->agents =
                (uint8_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, agents,
                (n * sizeof(agents)) * sizeof(uint8_t));
        } else {
            __ret->agents = NULL;
        }

        /* Output: hsa_symbol_kind_t * types */
        if ((types) != (NULL) && (n) > (0)) {
            __ret->types =
                (hsa_symbol_kind_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, types,
                (n) * sizeof(hsa_symbol_kind_t));
        } else {
            __ret->types = NULL;
        }

        /* Output: hipFunction_t * descriptors */
        if ((descriptors) != (NULL) && (n) > (0)) {
            __ret->descriptors =
                (hipFunction_t *) command_channel_attach_buffer(__chan, (struct command_base *)__ret, descriptors,
                (n) * sizeof(hipFunction_t));
        } else {
            __ret->descriptors = NULL;
        }

        /* Output: char * pool */
        if ((pool) != (NULL) && (pool_size) > (0)) {
            __ret->pool =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__ret, pool,
                (pool_size) * sizeof(char));
        } else {
            __ret->pool = NULL;
        }

#ifdef AVA_RECORD_REPLAY

#endif

        /* Send reply message */
        command_channel_send_command(__chan, (struct command_base *)__ret);

#ifdef AVA_RECORD_REPLAY
        /* Record call in object metadata */

#endif

        g_ptr_array_unref(__ava_alloc_list___do_c_mass_symbol_info);    /* Deallocate all memory in the alloc list */
        command_channel_free_command(__chan, (struct command_base *)__call);
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    default:
        abort_with_reason("Received unsupported command");
    }                                            // switch
}
