#define ava_is_worker 0
#define ava_is_guest 1

#include "guestlib.h"

#include "common/endpoint_lib.h"
#include "common/linkage.h"
#include "current_device.h"

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

static bool was_initted;
static pthread_once_t guestlib_init = PTHREAD_ONCE_INIT;

EXPORTED __thread int current_device = 0;

#define __chan nw_global_command_channel[get_ava_chan_no()]

/* DON'T CALL DIRECTLY! must be protected by pthread_once */
void init_hip_guestlib(void)
{
    was_initted = true;
    __handle_command_hip_init();
    nw_init_guestlib(HIP_API);
}

void __attribute__ ((destructor)) destroy_hip_guestlib(void)
{
    if (!was_initted)
       return;
    __handle_command_hip_destroy();
    nw_destroy_guestlib();
}

static struct nw_handle_pool *handle_pool = NULL;

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

void
__handle_command_hip(struct command_base *__cmd, int _chan_no)
{
    int ava_is_in,
     ava_is_out;
    set_ava_chan_no(_chan_no);
    switch (__cmd->command_id) {
    case RET_HIP_HIP_DEVICE_SYNCHRONIZE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_device_synchronize_ret *__ret = (struct hip_hip_device_synchronize_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_device_synchronize_ret));
        struct hip_hip_device_synchronize_call_record *__local =
            (struct hip_hip_device_synchronize_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MALLOC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_malloc_ret *__ret = (struct hip_hip_malloc_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_malloc_ret));
        struct hip_hip_malloc_call_record *__local =
            (struct hip_hip_malloc_call_record *)ava_remove_call(__ret->__call_id);

        {

            void **dptr;
            dptr = __local->dptr;

            size_t size;
            size = __local->size;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void ** dptr */
            if (__local->dptr != NULL && __ret->dptr != NULL) {
                memcpy(__local->dptr, ((__ret->dptr) != (NULL)) ? (((void **)command_channel_get_buffer(__chan, __cmd,
                                __ret->dptr))) : (__ret->dptr), (1) * sizeof(void *));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_FREE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_free_ret *__ret = (struct hip_hip_free_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_free_ret));
        struct hip_hip_free_call_record *__local = (struct hip_hip_free_call_record *)ava_remove_call(__ret->__call_id);

        {

            void *ptr;
            ptr = __local->ptr;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMCPY_HTO_D:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_hto_d_ret *__ret = (struct hip_hip_memcpy_hto_d_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_hto_d_ret));
        struct hip_hip_memcpy_hto_d_call_record *__local =
            (struct hip_hip_memcpy_hto_d_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t dst;
            dst = __local->dst;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            void *src;
            src = __local->src;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMCPY_DTO_H:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_dto_h_ret *__ret = (struct hip_hip_memcpy_dto_h_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_dto_h_ret));
        struct hip_hip_memcpy_dto_h_call_record *__local =
            (struct hip_hip_memcpy_dto_h_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t src;
            src = __local->src;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            void *dst;
            dst = __local->dst;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void * dst */
            if (__local->dst != NULL && __ret->dst != NULL) {
                memcpy(__local->dst, ((__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                __ret->dst))) : (__ret->dst), (sizeBytes) * sizeof(void));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMCPY_DTO_D:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_dto_d_ret *__ret = (struct hip_hip_memcpy_dto_d_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_dto_d_ret));
        struct hip_hip_memcpy_dto_d_call_record *__local =
            (struct hip_hip_memcpy_dto_d_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t dst;
            dst = __local->dst;

            hipDeviceptr_t src;
            src = __local->src;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_MEMCPY:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_memcpy_ret *__ret = (struct hip_nw_hip_memcpy_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_memcpy_ret));
        struct hip_nw_hip_memcpy_call_record *__local =
            (struct hip_nw_hip_memcpy_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipMemcpyKind kind;
            kind = __local->kind;

            void *src;
            src = __local->src;

            void *dst;
            dst = __local->dst;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && __local->dst != NULL && __ret->dst != NULL) {
                    memcpy(__local->dst, (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst),
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                        && __local->dst != NULL && __ret->dst != NULL) {
                        void *__tmp_dst_0;
                        __tmp_dst_0 = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst);
                        const size_t __dst_size_0 = ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                        for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                            const size_t ava_index = __dst_index_0;

                            char *__dst_a_0;
                            __dst_a_0 = (__local->dst) + __dst_index_0;

                            char *__dst_b_0;
                            __dst_b_0 = (__tmp_dst_0) + __dst_index_0;

                            if (kind == hipMemcpyDeviceToHost) {
                                *__dst_a_0 = *__dst_b_0;
                            }
                        }
                    }
                } else {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                        __local->dst = __ret->dst;
                    }
                }
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_MEMCPY_PEER_ASYNC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_memcpy_peer_async_ret *__ret = (struct hip_nw_hip_memcpy_peer_async_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_memcpy_peer_async_ret));
        struct hip_nw_hip_memcpy_peer_async_call_record *__local =
            (struct hip_nw_hip_memcpy_peer_async_call_record *)ava_remove_call(__ret->__call_id);

        {

            void *dst;
            dst = __local->dst;

            int dstDeviceId;
            dstDeviceId = __local->dstDeviceId;

            void *src;
            src = __local->src;

            int srcDevice;
            srcDevice = __local->srcDevice;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        break;
    }
    case RET_HIP_HIP_MEMCPY_HTO_D_ASYNC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_hto_d_async_ret *__ret = (struct hip_hip_memcpy_hto_d_async_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_hto_d_async_ret));
        struct hip_hip_memcpy_hto_d_async_call_record *__local =
            (struct hip_hip_memcpy_hto_d_async_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t dst;
            dst = __local->dst;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipStream_t stream;
            stream = __local->stream;

            void *src;
            src = __local->src;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMCPY_DTO_H_ASYNC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_dto_h_async_ret *__ret = (struct hip_hip_memcpy_dto_h_async_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_dto_h_async_ret));
        struct hip_hip_memcpy_dto_h_async_call_record *__local =
            (struct hip_hip_memcpy_dto_h_async_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t src;
            src = __local->src;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipStream_t stream;
            stream = __local->stream;

            void *dst;
            dst = __local->dst;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void * dst */
            if (__local->dst != NULL && __ret->dst != NULL) {
                memcpy(__local->dst, ((__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                __ret->dst))) : (__ret->dst), (sizeBytes) * sizeof(void));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMCPY_DTO_D_ASYNC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memcpy_dto_d_async_ret *__ret = (struct hip_hip_memcpy_dto_d_async_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memcpy_dto_d_async_ret));
        struct hip_hip_memcpy_dto_d_async_call_record *__local =
            (struct hip_hip_memcpy_dto_d_async_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDeviceptr_t dst;
            dst = __local->dst;

            hipDeviceptr_t src;
            src = __local->src;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_MEMCPY_SYNC:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_memcpy_sync_ret *__ret = (struct hip_nw_hip_memcpy_sync_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_memcpy_sync_ret));
        struct hip_nw_hip_memcpy_sync_call_record *__local =
            (struct hip_nw_hip_memcpy_sync_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipMemcpyKind kind;
            kind = __local->kind;

            void *dst;
            dst = __local->dst;

            void *src;
            src = __local->src;

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && __local->dst != NULL && __ret->dst != NULL) {
                    memcpy(__local->dst, (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst),
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                        && __local->dst != NULL && __ret->dst != NULL) {
                        void *__tmp_dst_0;
                        __tmp_dst_0 = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst);
                        const size_t __dst_size_0 = ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                        for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                            const size_t ava_index = __dst_index_0;

                            char *__dst_a_0;
                            __dst_a_0 = (__local->dst) + __dst_index_0;

                            char *__dst_b_0;
                            __dst_b_0 = (__tmp_dst_0) + __dst_index_0;

                            if (kind == hipMemcpyDeviceToHost) {
                                *__dst_a_0 = *__dst_b_0;
                            }
                        }
                    }
                } else {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                        __local->dst = __ret->dst;
                    }
                }
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_GET_DEVICE_COUNT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_get_device_count_ret *__ret = (struct hip_hip_get_device_count_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_get_device_count_ret));
        struct hip_hip_get_device_count_call_record *__local =
            (struct hip_hip_get_device_count_call_record *)ava_remove_call(__ret->__call_id);

        {

            int *count;
            count = __local->count;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: int * count */
            if (__local->count != NULL && __ret->count != NULL) {
                memcpy(__local->count, ((__ret->count) != (NULL)) ? (((int *)command_channel_get_buffer(__chan, __cmd,
                                __ret->count))) : (__ret->count), (1) * sizeof(int));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_SET_DEVICE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_set_device_ret *__ret = (struct hip_nw_hip_set_device_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_set_device_ret));
        struct hip_nw_hip_set_device_call_record *__local =
            (struct hip_nw_hip_set_device_call_record *)ava_remove_call(__ret->__call_id);

        {

            int deviceId;
            deviceId = __local->deviceId;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEM_GET_INFO:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_mem_get_info_ret *__ret = (struct hip_hip_mem_get_info_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_mem_get_info_ret));
        struct hip_hip_mem_get_info_call_record *__local =
            (struct hip_hip_mem_get_info_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t *__free;
            __free = __local->__free;

            size_t *total;
            total = __local->total;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: size_t * __free */
            if (__local->__free != NULL && __ret->__free != NULL) {
                memcpy(__local->__free, ((__ret->__free) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan,
                                __cmd, __ret->__free))) : (__ret->__free), (1) * sizeof(size_t));
            }

            /* Output: size_t * total */
            if (__local->total != NULL && __ret->total != NULL) {
                memcpy(__local->total, ((__ret->total) != (NULL)) ? (((size_t *) command_channel_get_buffer(__chan,
                                __cmd, __ret->total))) : (__ret->total), (1) * sizeof(size_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_STREAM_CREATE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_stream_create_ret *__ret = (struct hip_nw_hip_stream_create_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_stream_create_ret));
        struct hip_nw_hip_stream_create_call_record *__local =
            (struct hip_nw_hip_stream_create_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipStream_t *stream;
            stream = __local->stream;

            hsa_agent_t *agent;
            agent = __local->agent;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipStream_t * stream */
            if (__local->stream != NULL && __ret->stream != NULL) {
                memcpy(__local->stream,
                    ((__ret->stream) != (NULL)) ? (((hipStream_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->stream))) : (__ret->stream), (1) * sizeof(hipStream_t));
            }

            /* Output: hsa_agent_t * agent */
            if (__local->agent != NULL && __ret->agent != NULL) {
                hsa_agent_t *__tmp_agent_0;
                __tmp_agent_0 =
                    ((__ret->agent) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->agent))) : (__ret->agent);
                const size_t __agent_size_0 = (1);
                const size_t __agent_index_0 = 0;
                const size_t ava_index = 0;

                hsa_agent_t *__agent_a_0;
                __agent_a_0 = (__local->agent) + __agent_index_0;

                hsa_agent_t *__agent_b_0;
                __agent_b_0 = (__tmp_agent_0) + __agent_index_0;

                hsa_agent_t *ava_self;
                ava_self = &*__agent_a_0;
                uint64_t *__agent_a_1_handle;
                __agent_a_1_handle = &(*__agent_a_0).handle;
                uint64_t *__agent_b_1_handle;
                __agent_b_1_handle = &(*__agent_b_0).handle;
                *__agent_a_1_handle = *__agent_b_1_handle;
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_GET_DEVICE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_get_device_ret *__ret = (struct hip_nw_hip_get_device_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_get_device_ret));
        struct hip_nw_hip_get_device_call_record *__local =
            (struct hip_nw_hip_get_device_call_record *)ava_remove_call(__ret->__call_id);

        {

            int *deviceId;
            deviceId = __local->deviceId;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: int * deviceId */
            if (__local->deviceId != NULL && __ret->deviceId != NULL) {
                memcpy(__local->deviceId, ((__ret->deviceId) != (NULL)) ? (((int *)command_channel_get_buffer(__chan,
                                __cmd, __ret->deviceId))) : (__ret->deviceId), (1) * sizeof(int));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_INIT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_init_ret *__ret = (struct hip_hip_init_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_init_ret));
        struct hip_hip_init_call_record *__local = (struct hip_hip_init_call_record *)ava_remove_call(__ret->__call_id);

        {

            unsigned int flags;
            flags = __local->flags;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_CTX_GET_CURRENT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_ctx_get_current_ret *__ret = (struct hip_hip_ctx_get_current_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_ctx_get_current_ret));
        struct hip_hip_ctx_get_current_call_record *__local =
            (struct hip_hip_ctx_get_current_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipCtx_t *ctx;
            ctx = __local->ctx;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipCtx_t * ctx */
            if (__local->ctx != NULL && __ret->ctx != NULL) {
                memcpy(__local->ctx, ((__ret->ctx) != (NULL)) ? (((hipCtx_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->ctx))) : (__ret->ctx), (1) * sizeof(hipCtx_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_STREAM_SYNCHRONIZE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_stream_synchronize_ret *__ret = (struct hip_nw_hip_stream_synchronize_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_stream_synchronize_ret));
        struct hip_nw_hip_stream_synchronize_call_record *__local =
            (struct hip_nw_hip_stream_synchronize_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hip_get_device_properties_ret *__ret =
            (struct hip___do_c_hip_get_device_properties_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hip_get_device_properties_ret));
        struct hip___do_c_hip_get_device_properties_call_record *__local =
            (struct hip___do_c_hip_get_device_properties_call_record *)ava_remove_call(__ret->__call_id);

        {

            char *prop;
            prop = __local->prop;

            int deviceId;
            deviceId = __local->deviceId;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: char * prop */
            if (__local->prop != NULL && __ret->prop != NULL) {
                memcpy(__local->prop, ((__ret->prop) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                                __ret->prop))) : (__ret->prop), (sizeof(hipDeviceProp_t)) * sizeof(char));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hip_hcc_module_launch_kernel_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_kernel_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hip_hcc_module_launch_kernel_ret));
        struct hip___do_c_hip_hcc_module_launch_kernel_call_record *__local =
            (struct hip___do_c_hip_hcc_module_launch_kernel_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_kernel_dispatch_packet_t *aql;
            aql = __local->aql;

            hipStream_t stream;
            stream = __local->stream;

            void **kernelParams;
            kernelParams = __local->kernelParams;

            size_t extra_size;
            extra_size = __local->extra_size;

            hipEvent_t start;
            start = __local->start;

            char *extra;
            extra = __local->extra;

            hipEvent_t stop;
            stop = __local->stop;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_ret));
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record *__local =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record *)ava_remove_call(__ret->__call_id);

        {

            int numKernels;
            numKernels = __local->numKernels;

            hipEvent_t *stop;
            stop = __local->stop;

            hipStream_t stream;
            stream = __local->stream;

            hsa_kernel_dispatch_packet_t *aql;
            aql = __local->aql;

            hipEvent_t *start;
            start = __local->start;

            size_t *extra_size;
            extra_size = __local->extra_size;

            size_t total_extra_size;
            total_extra_size = __local->total_extra_size;

            char *all_extra;
            all_extra = __local->all_extra;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL_AND_MEMCPY:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret *__ret =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_ret));
        struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call_record *__local =
            (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call_record *)ava_remove_call(__ret->
            __call_id);

        {

            int numKernels;
            numKernels = __local->numKernels;

            hsa_kernel_dispatch_packet_t *aql;
            aql = __local->aql;

            size_t *extra_size;
            extra_size = __local->extra_size;

            hipEvent_t *stop;
            stop = __local->stop;

            hipStream_t stream;
            stream = __local->stream;

            hipEvent_t *start;
            start = __local->start;

            size_t total_extra_size;
            total_extra_size = __local->total_extra_size;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            char *all_extra;
            all_extra = __local->all_extra;

            hipMemcpyKind kind;
            kind = __local->kind;

            void *dst;
            dst = __local->dst;

            void *src;
            src = __local->src;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: void * dst */
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                    && __local->dst != NULL && __ret->dst != NULL) {
                    memcpy(__local->dst, (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst),
                        ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) * sizeof(void));
                }
            } else {
                if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                        && __local->dst != NULL && __ret->dst != NULL) {
                        void *__tmp_dst_0;
                        __tmp_dst_0 = (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)
                            && (__ret->dst) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                    __ret->dst))) : (__ret->dst);
                        const size_t __dst_size_0 = ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0));
                        for (size_t __dst_index_0 = 0; __dst_index_0 < __dst_size_0; __dst_index_0++) {
                            const size_t ava_index = __dst_index_0;

                            char *__dst_a_0;
                            __dst_a_0 = (__local->dst) + __dst_index_0;

                            char *__dst_b_0;
                            __dst_b_0 = (__tmp_dst_0) + __dst_index_0;

                            if (kind == hipMemcpyDeviceToHost) {
                                *__dst_a_0 = *__dst_b_0;
                            }
                        }
                    }
                } else {
                    if (kind == hipMemcpyDeviceToHost
                        && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_OPAQUE)) {
                        __local->dst = __ret->dst;
                    }
                }
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hsa_system_major_extension_supported_ret *__ret =
            (struct hip_nw_hsa_system_major_extension_supported_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hsa_system_major_extension_supported_ret));
        struct hip_nw_hsa_system_major_extension_supported_call_record *__local =
            (struct hip_nw_hsa_system_major_extension_supported_call_record *)ava_remove_call(__ret->__call_id);

        {

            uint16_t extension;
            extension = __local->extension;

            uint16_t version_major;
            version_major = __local->version_major;

            uint16_t *version_minor;
            version_minor = __local->version_minor;

            _Bool *result;
            result = __local->result;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: uint16_t * version_minor */
            if (__local->version_minor != NULL && __ret->version_minor != NULL) {
                memcpy(__local->version_minor,
                    ((__ret->version_minor) != (NULL)) ? (((uint16_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->version_minor))) : (__ret->version_minor), (1) * sizeof(uint16_t));
            }

            /* Output: _Bool * result */
            if (__local->result != NULL && __ret->result != NULL) {
                memcpy(__local->result, ((__ret->result) != (NULL)) ? (((_Bool *) command_channel_get_buffer(__chan,
                                __cmd, __ret->result))) : (__ret->result), (1) * sizeof(_Bool));
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HSA_EXECUTABLE_CREATE_ALT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hsa_executable_create_alt_ret *__ret = (struct hip_nw_hsa_executable_create_alt_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hsa_executable_create_alt_ret));
        struct hip_nw_hsa_executable_create_alt_call_record *__local =
            (struct hip_nw_hsa_executable_create_alt_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_profile_t profile;
            profile = __local->profile;

            char *options;
            options = __local->options;

            hsa_default_float_rounding_mode_t default_float_rounding_mode;
            default_float_rounding_mode = __local->default_float_rounding_mode;

            hsa_executable_t *executable;
            executable = __local->executable;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: hsa_executable_t * executable */
            if (__local->executable != NULL && __ret->executable != NULL) {
                hsa_executable_t *__tmp_executable_0;
                __tmp_executable_0 =
                    ((__ret->executable) != (NULL)) ? (((hsa_executable_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->executable))) : (__ret->executable);
                const size_t __executable_size_0 = (1);
                const size_t __executable_index_0 = 0;
                const size_t ava_index = 0;

                hsa_executable_t *__executable_a_0;
                __executable_a_0 = (__local->executable) + __executable_index_0;

                hsa_executable_t *__executable_b_0;
                __executable_b_0 = (__tmp_executable_0) + __executable_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__executable_a_0;
                uint64_t *__executable_a_1_handle;
                __executable_a_1_handle = &(*__executable_a_0).handle;
                uint64_t *__executable_b_1_handle;
                __executable_b_1_handle = &(*__executable_b_0).handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HSA_ISA_FROM_NAME:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hsa_isa_from_name_ret *__ret = (struct hip_nw_hsa_isa_from_name_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hsa_isa_from_name_ret));
        struct hip_nw_hsa_isa_from_name_call_record *__local =
            (struct hip_nw_hsa_isa_from_name_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_isa_t *isa;
            isa = __local->isa;

            char *name;
            name = __local->name;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: hsa_isa_t * isa */
            if (__local->isa != NULL && __ret->isa != NULL) {
                hsa_isa_t *__tmp_isa_0;
                __tmp_isa_0 =
                    ((__ret->isa) != (NULL)) ? (((hsa_isa_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->isa))) : (__ret->isa);
                const size_t __isa_size_0 = (1);
                const size_t __isa_index_0 = 0;
                const size_t ava_index = 0;

                hsa_isa_t *__isa_a_0;
                __isa_a_0 = (__local->isa) + __isa_index_0;

                hsa_isa_t *__isa_b_0;
                __isa_b_0 = (__tmp_isa_0) + __isa_index_0;

                hsa_isa_t *ava_self;
                ava_self = &*__isa_a_0;
                uint64_t *__isa_a_1_handle;
                __isa_a_1_handle = &(*__isa_a_0).handle;
                uint64_t *__isa_b_1_handle;
                __isa_b_1_handle = &(*__isa_b_0).handle;
                *__isa_a_1_handle = *__isa_b_1_handle;
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_PEEK_AT_LAST_ERROR:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_peek_at_last_error_ret *__ret = (struct hip_hip_peek_at_last_error_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_peek_at_last_error_ret));
        struct hip_hip_peek_at_last_error_call_record *__local =
            (struct hip_hip_peek_at_last_error_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_device_get_attribute_ret *__ret = (struct hip_nw_hip_device_get_attribute_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_device_get_attribute_ret));
        struct hip_nw_hip_device_get_attribute_call_record *__local =
            (struct hip_nw_hip_device_get_attribute_call_record *)ava_remove_call(__ret->__call_id);

        {

            int *pi;
            pi = __local->pi;

            hipDeviceAttribute_t attr;
            attr = __local->attr;

            int deviceId;
            deviceId = __local->deviceId;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: int * pi */
            if (__local->pi != NULL && __ret->pi != NULL) {
                memcpy(__local->pi, ((__ret->pi) != (NULL)) ? (((int *)command_channel_get_buffer(__chan, __cmd,
                                __ret->pi))) : (__ret->pi), (1) * sizeof(int));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MODULE_LOAD_DATA:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_module_load_data_ret *__ret = (struct hip_hip_module_load_data_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_module_load_data_ret));
        struct hip_hip_module_load_data_call_record *__local =
            (struct hip_hip_module_load_data_call_record *)ava_remove_call(__ret->__call_id);

        {

            void *image;
            image = __local->image;

            hipModule_t *module;
            module = __local->module;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipModule_t * module */
            if (__local->module != NULL && __ret->module != NULL) {
                memcpy(__local->module,
                    ((__ret->module) != (NULL)) ? (((hipModule_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->module))) : (__ret->module), (1) * sizeof(hipModule_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hsa_executable_symbol_get_info_ret *__ret =
            (struct hip___do_c_hsa_executable_symbol_get_info_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hsa_executable_symbol_get_info_ret));
        struct hip___do_c_hsa_executable_symbol_get_info_call_record *__local =
            (struct hip___do_c_hsa_executable_symbol_get_info_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_executable_symbol_t executable_symbol;
            executable_symbol = __local->executable_symbol;

            hsa_executable_symbol_info_t attribute;
            attribute = __local->attribute;

            size_t max_value;
            max_value = __local->max_value;

            char *value;
            value = __local->value;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: char * value */
            if (__local->value != NULL && __ret->value != NULL) {
                memcpy(__local->value, ((__ret->value) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                                __ret->value))) : (__ret->value), (max_value) * sizeof(char));
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_CTX_SET_CURRENT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_ctx_set_current_ret *__ret = (struct hip_nw_hip_ctx_set_current_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_ctx_set_current_ret));
        struct hip_nw_hip_ctx_set_current_call_record *__local =
            (struct hip_nw_hip_ctx_set_current_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipCtx_t ctx;
            ctx = __local->ctx;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_EVENT_CREATE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_event_create_ret *__ret = (struct hip_hip_event_create_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_event_create_ret));
        struct hip_hip_event_create_call_record *__local =
            (struct hip_hip_event_create_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipEvent_t *event;
            event = __local->event;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipEvent_t * event */
            if (__local->event != NULL && __ret->event != NULL) {
                memcpy(__local->event, ((__ret->event) != (NULL)) ? (((hipEvent_t *) command_channel_get_buffer(__chan,
                                __cmd, __ret->event))) : (__ret->event), (1) * sizeof(hipEvent_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_EVENT_RECORD:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_event_record_ret *__ret = (struct hip_hip_event_record_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_event_record_ret));
        struct hip_hip_event_record_call_record *__local =
            (struct hip_hip_event_record_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipEvent_t event;
            event = __local->event;

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_EVENT_SYNCHRONIZE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_event_synchronize_ret *__ret = (struct hip_hip_event_synchronize_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_event_synchronize_ret));
        struct hip_hip_event_synchronize_call_record *__local =
            (struct hip_hip_event_synchronize_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipEvent_t event;
            event = __local->event;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_EVENT_DESTROY:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_event_destroy_ret *__ret = (struct hip_hip_event_destroy_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_event_destroy_ret));
        struct hip_hip_event_destroy_call_record *__local =
            (struct hip_hip_event_destroy_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipEvent_t event;
            event = __local->event;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_EVENT_ELAPSED_TIME:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_event_elapsed_time_ret *__ret = (struct hip_hip_event_elapsed_time_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_event_elapsed_time_ret));
        struct hip_hip_event_elapsed_time_call_record *__local =
            (struct hip_hip_event_elapsed_time_call_record *)ava_remove_call(__ret->__call_id);

        {

            float *ms;
            ms = __local->ms;

            hipEvent_t start;
            start = __local->start;

            hipEvent_t stop;
            stop = __local->stop;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: float * ms */
            if (__local->ms != NULL && __ret->ms != NULL) {
                memcpy(__local->ms, ((__ret->ms) != (NULL)) ? (((float *)command_channel_get_buffer(__chan, __cmd,
                                __ret->ms))) : (__ret->ms), (1) * sizeof(float));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MODULE_LOAD:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_module_load_ret *__ret = (struct hip_hip_module_load_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_module_load_ret));
        struct hip_hip_module_load_call_record *__local =
            (struct hip_hip_module_load_call_record *)ava_remove_call(__ret->__call_id);

        {

            char *fname;
            fname = __local->fname;

            hipModule_t *module;
            module = __local->module;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipModule_t * module */
            if (__local->module != NULL && __ret->module != NULL) {
                memcpy(__local->module,
                    ((__ret->module) != (NULL)) ? (((hipModule_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->module))) : (__ret->module), (1) * sizeof(hipModule_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MODULE_UNLOAD:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_module_unload_ret *__ret = (struct hip_hip_module_unload_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_module_unload_ret));
        struct hip_hip_module_unload_call_record *__local =
            (struct hip_hip_module_unload_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipModule_t module;
            module = __local->module;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_STREAM_DESTROY:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_stream_destroy_ret *__ret = (struct hip_nw_hip_stream_destroy_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_stream_destroy_ret));
        struct hip_nw_hip_stream_destroy_call_record *__local =
            (struct hip_nw_hip_stream_destroy_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipStream_t stream;
            stream = __local->stream;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MODULE_GET_FUNCTION:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_module_get_function_ret *__ret = (struct hip_hip_module_get_function_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_module_get_function_ret));
        struct hip_hip_module_get_function_call_record *__local =
            (struct hip_hip_module_get_function_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipFunction_t *function;
            function = __local->function;

            char *kname;
            kname = __local->kname;

            hipModule_t module;
            module = __local->module;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipFunction_t * function */
            if (__local->function != NULL && __ret->function != NULL) {
                memcpy(__local->function,
                    ((__ret->function) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->function))) : (__ret->function), (1) * sizeof(hipFunction_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_GET_LAST_ERROR:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_get_last_error_ret *__ret = (struct hip_hip_get_last_error_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_get_last_error_ret));
        struct hip_hip_get_last_error_call_record *__local =
            (struct hip_hip_get_last_error_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_MEMSET:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_memset_ret *__ret = (struct hip_hip_memset_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_memset_ret));
        struct hip_hip_memset_call_record *__local =
            (struct hip_hip_memset_call_record *)ava_remove_call(__ret->__call_id);

        {

            void *dst;
            dst = __local->dst;

            int value;
            value = __local->value;

            size_t sizeBytes;
            sizeBytes = __local->sizeBytes;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_HIP_STREAM_WAIT_EVENT:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_hip_stream_wait_event_ret *__ret = (struct hip_hip_stream_wait_event_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_hip_stream_wait_event_ret));
        struct hip_hip_stream_wait_event_call_record *__local =
            (struct hip_hip_stream_wait_event_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipStream_t stream;
            stream = __local->stream;

            hipEvent_t event;
            event = __local->event;

            unsigned int flags;
            flags = __local->flags;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_HSA_AGENT_GET_INFO:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_hsa_agent_get_info_ret *__ret = (struct hip___do_c_hsa_agent_get_info_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_hsa_agent_get_info_ret));
        struct hip___do_c_hsa_agent_get_info_call_record *__local =
            (struct hip___do_c_hsa_agent_get_info_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_agent_t agent;
            agent = __local->agent;

            hsa_agent_info_t attribute;
            attribute = __local->attribute;

            size_t max_value;
            max_value = __local->max_value;

            void *value;
            value = __local->value;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: void * value */
            if (__local->value != NULL && __ret->value != NULL) {
                memcpy(__local->value, ((__ret->value) != (NULL)) ? (((void *)command_channel_get_buffer(__chan, __cmd,
                                __ret->value))) : (__ret->value), (max_value) * sizeof(void));
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_LOAD_EXECUTABLE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_load_executable_ret *__ret = (struct hip___do_c_load_executable_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_load_executable_ret));
        struct hip___do_c_load_executable_call_record *__local =
            (struct hip___do_c_load_executable_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t file_len;
            file_len = __local->file_len;

            char *file_buf;
            file_buf = __local->file_buf;

            hsa_executable_t *executable;
            executable = __local->executable;

            hsa_agent_t *agent;
            agent = __local->agent;

            int ret;
            ret = __ret->ret;

            /* Output: hsa_executable_t * executable */
            if (__local->executable != NULL && __ret->executable != NULL) {
                hsa_executable_t *__tmp_executable_0;
                __tmp_executable_0 =
                    ((__ret->executable) != (NULL)) ? (((hsa_executable_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->executable))) : (__ret->executable);
                const size_t __executable_size_0 = (1);
                const size_t __executable_index_0 = 0;
                const size_t ava_index = 0;

                hsa_executable_t *__executable_a_0;
                __executable_a_0 = (__local->executable) + __executable_index_0;

                hsa_executable_t *__executable_b_0;
                __executable_b_0 = (__tmp_executable_0) + __executable_index_0;

                hsa_executable_t *ava_self;
                ava_self = &*__executable_a_0;
                uint64_t *__executable_a_1_handle;
                __executable_a_1_handle = &(*__executable_a_0).handle;
                uint64_t *__executable_b_1_handle;
                __executable_b_1_handle = &(*__executable_b_0).handle;
                *__executable_a_1_handle = *__executable_b_1_handle;
            }

            /* Output: int ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_GET_AGENTS:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_get_agents_ret *__ret = (struct hip___do_c_get_agents_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_get_agents_ret));
        struct hip___do_c_get_agents_call_record *__local =
            (struct hip___do_c_get_agents_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t max_agents;
            max_agents = __local->max_agents;

            hsa_agent_t *agents;
            agents = __local->agents;

            size_t ret;
            ret = __ret->ret;

            /* Output: hsa_agent_t * agents */
            if (__local->agents != NULL && __ret->agents != NULL) {
                hsa_agent_t *__tmp_agents_0;
                __tmp_agents_0 =
                    ((__ret->agents) != (NULL)) ? (((hsa_agent_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->agents))) : (__ret->agents);
                const size_t __agents_size_0 = (max_agents);
                for (size_t __agents_index_0 = 0; __agents_index_0 < __agents_size_0; __agents_index_0++) {
                    const size_t ava_index = __agents_index_0;

                    hsa_agent_t *__agents_a_0;
                    __agents_a_0 = (__local->agents) + __agents_index_0;

                    hsa_agent_t *__agents_b_0;
                    __agents_b_0 = (__tmp_agents_0) + __agents_index_0;

                    hsa_agent_t *ava_self;
                    ava_self = &*__agents_a_0;
                    uint64_t *__agents_a_1_handle;
                    __agents_a_1_handle = &(*__agents_a_0).handle;
                    uint64_t *__agents_b_1_handle;
                    __agents_b_1_handle = &(*__agents_b_0).handle;
                    *__agents_a_1_handle = *__agents_b_1_handle;
            }}

            /* Output: size_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_GET_ISAS:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_get_isas_ret *__ret = (struct hip___do_c_get_isas_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_get_isas_ret));
        struct hip___do_c_get_isas_call_record *__local =
            (struct hip___do_c_get_isas_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_agent_t agents;
            agents = __local->agents;

            size_t max_isas;
            max_isas = __local->max_isas;

            hsa_isa_t *isas;
            isas = __local->isas;

            size_t ret;
            ret = __ret->ret;

            /* Output: hsa_isa_t * isas */
            if (__local->isas != NULL && __ret->isas != NULL) {
                hsa_isa_t *__tmp_isas_0;
                __tmp_isas_0 =
                    ((__ret->isas) != (NULL)) ? (((hsa_isa_t *) command_channel_get_buffer(__chan, __cmd,
                            __ret->isas))) : (__ret->isas);
                const size_t __isas_size_0 = (max_isas);
                for (size_t __isas_index_0 = 0; __isas_index_0 < __isas_size_0; __isas_index_0++) {
                    const size_t ava_index = __isas_index_0;

                    hsa_isa_t *__isas_a_0;
                    __isas_a_0 = (__local->isas) + __isas_index_0;

                    hsa_isa_t *__isas_b_0;
                    __isas_b_0 = (__tmp_isas_0) + __isas_index_0;

                    hsa_isa_t *ava_self;
                    ava_self = &*__isas_a_0;
                    uint64_t *__isas_a_1_handle;
                    __isas_a_1_handle = &(*__isas_a_0).handle;
                    uint64_t *__isas_b_1_handle;
                    __isas_b_1_handle = &(*__isas_b_0).handle;
                    *__isas_a_1_handle = *__isas_b_1_handle;
            }}

            /* Output: size_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_GET_KERENEL_SYMBOLS:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_get_kerenel_symbols_ret *__ret = (struct hip___do_c_get_kerenel_symbols_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_get_kerenel_symbols_ret));
        struct hip___do_c_get_kerenel_symbols_call_record *__local =
            (struct hip___do_c_get_kerenel_symbols_call_record *)ava_remove_call(__ret->__call_id);

        {

            hsa_executable_t *exec;
            exec = __local->exec;

            hsa_agent_t *agent;
            agent = __local->agent;

            size_t max_symbols;
            max_symbols = __local->max_symbols;

            hsa_executable_symbol_t *symbols;
            symbols = __local->symbols;

            size_t ret;
            ret = __ret->ret;

            /* Output: hsa_executable_symbol_t * symbols */
            if (__local->symbols != NULL && __ret->symbols != NULL) {
                hsa_executable_symbol_t *__tmp_symbols_0;
                __tmp_symbols_0 =
                    ((__ret->symbols) != (NULL)) ? (((hsa_executable_symbol_t *) command_channel_get_buffer(__chan,
                            __cmd, __ret->symbols))) : (__ret->symbols);
                const size_t __symbols_size_0 = (max_symbols);
                for (size_t __symbols_index_0 = 0; __symbols_index_0 < __symbols_size_0; __symbols_index_0++) {
                    const size_t ava_index = __symbols_index_0;

                    hsa_executable_symbol_t *__symbols_a_0;
                    __symbols_a_0 = (__local->symbols) + __symbols_index_0;

                    hsa_executable_symbol_t *__symbols_b_0;
                    __symbols_b_0 = (__tmp_symbols_0) + __symbols_index_0;

                    hsa_executable_symbol_t *ava_self;
                    ava_self = &*__symbols_a_0;
                    uint64_t *__symbols_a_1_handle;
                    __symbols_a_1_handle = &(*__symbols_a_0).handle;
                    uint64_t *__symbols_b_1_handle;
                    __symbols_b_1_handle = &(*__symbols_b_0).handle;
                    *__symbols_a_1_handle = *__symbols_b_1_handle;
            }}

            /* Output: size_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_QUERY_HOST_ADDRESS:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_query_host_address_ret *__ret = (struct hip___do_c_query_host_address_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_query_host_address_ret));
        struct hip___do_c_query_host_address_call_record *__local =
            (struct hip___do_c_query_host_address_call_record *)ava_remove_call(__ret->__call_id);

        {

            uint64_t kernel_object_;
            kernel_object_ = __local->kernel_object_;

            char *kernel_header_;
            kernel_header_ = __local->kernel_header_;

            hsa_status_t ret;
            ret = __ret->ret;

            /* Output: char * kernel_header_ */
            if (__local->kernel_header_ != NULL && __ret->kernel_header_ != NULL) {
                memcpy(__local->kernel_header_,
                    ((__ret->kernel_header_) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                                __ret->kernel_header_))) : (__ret->kernel_header_),
                    (sizeof(amd_kernel_code_t)) * sizeof(char));
            }

            /* Output: hsa_status_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_GET_KERNEL_DESCRIPTOR:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_get_kernel_descriptor_ret *__ret = (struct hip___do_c_get_kernel_descriptor_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_get_kernel_descriptor_ret));
        struct hip___do_c_get_kernel_descriptor_call_record *__local =
            (struct hip___do_c_get_kernel_descriptor_call_record *)ava_remove_call(__ret->__call_id);

        {

            char *name;
            name = __local->name;

            hsa_executable_symbol_t *symbol;
            symbol = __local->symbol;

            hipFunction_t *f;
            f = __local->f;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipFunction_t * f */
            if (__local->f != NULL && __ret->f != NULL) {
                memcpy(__local->f, ((__ret->f) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan,
                                __cmd, __ret->f))) : (__ret->f), (1) * sizeof(hipFunction_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_HIP_CTX_GET_DEVICE:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_hip_ctx_get_device_ret *__ret = (struct hip_nw_hip_ctx_get_device_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_hip_ctx_get_device_ret));
        struct hip_nw_hip_ctx_get_device_call_record *__local =
            (struct hip_nw_hip_ctx_get_device_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipDevice_t *device;
            device = __local->device;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: hipDevice_t * device */
            if (__local->device != NULL && __ret->device != NULL) {
                memcpy(__local->device,
                    ((__ret->device) != (NULL)) ? (((hipDevice_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->device))) : (__ret->device), (1) * sizeof(hipDevice_t));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP_NW_LOOKUP_KERN_INFO:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip_nw_lookup_kern_info_ret *__ret = (struct hip_nw_lookup_kern_info_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip_nw_lookup_kern_info_ret));
        struct hip_nw_lookup_kern_info_call_record *__local =
            (struct hip_nw_lookup_kern_info_call_record *)ava_remove_call(__ret->__call_id);

        {

            hipFunction_t f;
            f = __local->f;

            struct nw_kern_info *info;
            info = __local->info;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: struct nw_kern_info * info */
            if (__local->info != NULL && __ret->info != NULL) {
                struct nw_kern_info *__tmp_info_0;
                __tmp_info_0 =
                    ((__ret->info) != (NULL)) ? (((struct nw_kern_info *)command_channel_get_buffer(__chan, __cmd,
                            __ret->info))) : (__ret->info);
                const size_t __info_size_0 = (1);
                const size_t __info_index_0 = 0;
                const size_t ava_index = 0;

                struct nw_kern_info *__info_a_0;
                __info_a_0 = (__local->info) + __info_index_0;

                struct nw_kern_info *__info_b_0;
                __info_b_0 = (__tmp_info_0) + __info_index_0;

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

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }
    case RET_HIP___DO_C_MASS_SYMBOL_INFO:{
        ava_is_in = 0;
        ava_is_out = 1;
        struct hip___do_c_mass_symbol_info_ret *__ret = (struct hip___do_c_mass_symbol_info_ret *)__cmd;
        assert(__ret->base.api_id == HIP_API);
        assert(__ret->base.command_size == sizeof(struct hip___do_c_mass_symbol_info_ret));
        struct hip___do_c_mass_symbol_info_call_record *__local =
            (struct hip___do_c_mass_symbol_info_call_record *)ava_remove_call(__ret->__call_id);

        {

            size_t n;
            n = __local->n;

            unsigned int *offsets;
            offsets = __local->offsets;

            hsa_executable_symbol_t *syms;
            syms = __local->syms;

            size_t pool_size;
            pool_size = __local->pool_size;

            uint8_t *agents;
            agents = __local->agents;

            hsa_symbol_kind_t *types;
            types = __local->types;

            hipFunction_t *descriptors;
            descriptors = __local->descriptors;

            char *pool;
            pool = __local->pool;

            hipError_t ret;
            ret = __ret->ret;

            /* Output: unsigned int * offsets */
            if (__local->offsets != NULL && __ret->offsets != NULL) {
                memcpy(__local->offsets,
                    ((__ret->offsets) != (NULL)) ? (((unsigned int *)command_channel_get_buffer(__chan, __cmd,
                                __ret->offsets))) : (__ret->offsets), (n) * sizeof(unsigned int));
            }

            /* Output: uint8_t * agents */
            if (__local->agents != NULL && __ret->agents != NULL) {
                memcpy(__local->agents, ((__ret->agents) != (NULL)) ? (((uint8_t *) command_channel_get_buffer(__chan,
                                __cmd, __ret->agents))) : (__ret->agents), (n * sizeof(agents)) * sizeof(uint8_t));
            }

            /* Output: hsa_symbol_kind_t * types */
            if (__local->types != NULL && __ret->types != NULL) {
                memcpy(__local->types,
                    ((__ret->types) != (NULL)) ? (((hsa_symbol_kind_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->types))) : (__ret->types), (n) * sizeof(hsa_symbol_kind_t));
            }

            /* Output: hipFunction_t * descriptors */
            if (__local->descriptors != NULL && __ret->descriptors != NULL) {
                memcpy(__local->descriptors,
                    ((__ret->descriptors) != (NULL)) ? (((hipFunction_t *) command_channel_get_buffer(__chan, __cmd,
                                __ret->descriptors))) : (__ret->descriptors), (n) * sizeof(hipFunction_t));
            }

            /* Output: char * pool */
            if (__local->pool != NULL && __ret->pool != NULL) {
                memcpy(__local->pool, ((__ret->pool) != (NULL)) ? (((char *)command_channel_get_buffer(__chan, __cmd,
                                __ret->pool))) : (__ret->pool), (pool_size) * sizeof(char));
            }

            /* Output: hipError_t ret */
            __local->ret = __ret->ret;

        }

        if (__local->__handler_deallocate) {
            free(__local);
        }
        __local->__call_complete = 1;
        command_channel_free_command(__chan, (struct command_base *)__ret);
        break;
    }

    default:
        abort_with_reason("Received unsupported command");
    }                                            // switch
}

////// API function stub implementations

EXPORTED hipError_t
hipDeviceSynchronize()
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipDeviceSynchronize = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_device_synchronize_call *__cmd =
        (struct hip_hip_device_synchronize_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_device_synchronize_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_DEVICE_SYNCHRONIZE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

    }

    struct hip_hip_device_synchronize_call_record *__call_record =
        (struct hip_hip_device_synchronize_call_record *)calloc(1,
        sizeof(struct hip_hip_device_synchronize_call_record));

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipDeviceSynchronize);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMalloc(void **dptr, size_t size)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMalloc = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_malloc_call *__cmd =
        (struct hip_hip_malloc_call *)command_channel_new_command(__chan, sizeof(struct hip_hip_malloc_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MALLOC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void ** dptr */
        if ((dptr) != (NULL)) {
            __cmd->dptr = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->dptr = NULL;
        }
        /* Input: size_t size */
        __cmd->size = size;
    }

    struct hip_hip_malloc_call_record *__call_record =
        (struct hip_hip_malloc_call_record *)calloc(1, sizeof(struct hip_hip_malloc_call_record));

    __call_record->dptr = dptr;

    __call_record->size = size;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMalloc);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipFree(void *ptr)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipFree = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_free_call *__cmd =
        (struct hip_hip_free_call *)command_channel_new_command(__chan, sizeof(struct hip_hip_free_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_FREE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * ptr */
        __cmd->ptr = ptr;
    }

    struct hip_hip_free_call_record *__call_record =
        (struct hip_hip_free_call_record *)calloc(1, sizeof(struct hip_hip_free_call_record));

    __call_record->ptr = ptr;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipFree); /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyHtoD = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: void * src */
        if ((src) != (NULL) && (sizeBytes) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (sizeBytes) * sizeof(void));
        }
    }
    struct hip_hip_memcpy_hto_d_call *__cmd =
        (struct hip_hip_memcpy_hto_d_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_hto_d_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_HTO_D;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipDeviceptr_t dst */
        __cmd->dst = dst;
        /* Input: void * src */
        if ((src) != (NULL) && (sizeBytes) > (0)) {
            __cmd->src =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, src,
                (sizeBytes) * sizeof(void));
        } else {
            __cmd->src = NULL;
        }
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
    }

    struct hip_hip_memcpy_hto_d_call_record *__call_record =
        (struct hip_hip_memcpy_hto_d_call_record *)calloc(1, sizeof(struct hip_hip_memcpy_hto_d_call_record));

    __call_record->dst = dst;

    __call_record->sizeBytes = sizeBytes;

    __call_record->src = src;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyHtoD);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyDtoH = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_memcpy_dto_h_call *__cmd =
        (struct hip_hip_memcpy_dto_h_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_dto_h_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_DTO_H;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        if ((dst) != (NULL) && (sizeBytes) > (0)) {
            __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->dst = NULL;
        }
        /* Input: hipDeviceptr_t src */
        __cmd->src = src;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
    }

    struct hip_hip_memcpy_dto_h_call_record *__call_record =
        (struct hip_hip_memcpy_dto_h_call_record *)calloc(1, sizeof(struct hip_hip_memcpy_dto_h_call_record));

    __call_record->src = src;

    __call_record->sizeBytes = sizeBytes;

    __call_record->dst = dst;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoH);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyDtoD = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_memcpy_dto_d_call *__cmd =
        (struct hip_hip_memcpy_dto_d_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_dto_d_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_DTO_D;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipDeviceptr_t dst */
        __cmd->dst = dst;
        /* Input: hipDeviceptr_t src */
        __cmd->src = src;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
    }

    struct hip_hip_memcpy_dto_d_call_record *__call_record =
        (struct hip_hip_memcpy_dto_d_call_record *)calloc(1, sizeof(struct hip_hip_memcpy_dto_d_call_record));

    __call_record->dst = dst;

    __call_record->src = src;

    __call_record->sizeBytes = sizeBytes;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoD);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipMemcpy = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __total_buffer_size +=
                    command_channel_buffer_size(__chan,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                }
            }
    }}
    struct hip_nw_hip_memcpy_call *__cmd =
        (struct hip_nw_hip_memcpy_call *)command_channel_new_command(__chan, sizeof(struct hip_nw_hip_memcpy_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_MEMCPY;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
                } else {
                    __cmd->dst = NULL;
                }
            } else {
                __cmd->dst = dst;
            }
        }
        /* Input: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyHostToDevice
                && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __cmd->src =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, src,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            } else {
                __cmd->src = NULL;
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyHostToDevice
                    && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    void *__tmp_src_0;
                    __tmp_src_0 =
                        (void *)calloc(1, ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpy, __tmp_src_0);
                    const size_t __src_size_0 = ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                        const size_t ava_index = __src_index_0;

                        char *__src_a_0;
                        __src_a_0 = (src) + __src_index_0;

                        char *__src_b_0;
                        __src_b_0 = (__tmp_src_0) + __src_index_0;

                        *__src_b_0 = *__src_a_0;
                    }
                    __cmd->src =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_src_0,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                } else {
                    __cmd->src = NULL;
                }
            } else {
                __cmd->src = src;
            }
        }
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipMemcpyKind kind */
        __cmd->kind = kind;
    }

    struct hip_nw_hip_memcpy_call_record *__call_record =
        (struct hip_nw_hip_memcpy_call_record *)calloc(1, sizeof(struct hip_nw_hip_memcpy_call_record));

    __call_record->sizeBytes = sizeBytes;

    __call_record->kind = kind;

    __call_record->src = src;

    __call_record->dst = dst;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpy);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src, int srcDevice, size_t sizeBytes, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipMemcpyPeerAsync = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_memcpy_peer_async_call *__cmd =
        (struct hip_nw_hip_memcpy_peer_async_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_memcpy_peer_async_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_MEMCPY_PEER_ASYNC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        __cmd->dst = dst;
        /* Input: int dstDeviceId */
        __cmd->dstDeviceId = dstDeviceId;
        /* Input: const void * src */
        __cmd->src = src;
        /* Input: int srcDevice */
        __cmd->srcDevice = srcDevice;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_nw_hip_memcpy_peer_async_call_record *__call_record =
        (struct hip_nw_hip_memcpy_peer_async_call_record *)calloc(1,
        sizeof(struct hip_nw_hip_memcpy_peer_async_call_record));

    __call_record->dst = dst;

    __call_record->dstDeviceId = dstDeviceId;

    __call_record->src = src;

    __call_record->srcDevice = srcDevice;

    __call_record->sizeBytes = sizeBytes;

    __call_record->stream = stream;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpyPeerAsync);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyHtoDAsync = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: void * src */
        if ((src) != (NULL) && (sizeBytes) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (sizeBytes) * sizeof(void));
        }
    }
    struct hip_hip_memcpy_hto_d_async_call *__cmd =
        (struct hip_hip_memcpy_hto_d_async_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_hto_d_async_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_HTO_D_ASYNC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipDeviceptr_t dst */
        __cmd->dst = dst;
        /* Input: void * src */
        if ((src) != (NULL) && (sizeBytes) > (0)) {
            __cmd->src =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, src,
                (sizeBytes) * sizeof(void));
        } else {
            __cmd->src = NULL;
        }
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_hip_memcpy_hto_d_async_call_record *__call_record =
        (struct hip_hip_memcpy_hto_d_async_call_record *)calloc(1,
        sizeof(struct hip_hip_memcpy_hto_d_async_call_record));

    __call_record->dst = dst;

    __call_record->sizeBytes = sizeBytes;

    __call_record->stream = stream;

    __call_record->src = src;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyHtoDAsync);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyDtoHAsync = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_memcpy_dto_h_async_call *__cmd =
        (struct hip_hip_memcpy_dto_h_async_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_dto_h_async_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_DTO_H_ASYNC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        if ((dst) != (NULL) && (sizeBytes) > (0)) {
            __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->dst = NULL;
        }
        /* Input: hipDeviceptr_t src */
        __cmd->src = src;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_hip_memcpy_dto_h_async_call_record *__call_record =
        (struct hip_hip_memcpy_dto_h_async_call_record *)calloc(1,
        sizeof(struct hip_hip_memcpy_dto_h_async_call_record));

    __call_record->src = src;

    __call_record->sizeBytes = sizeBytes;

    __call_record->stream = stream;

    __call_record->dst = dst;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoHAsync);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemcpyDtoDAsync = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_memcpy_dto_d_async_call *__cmd =
        (struct hip_hip_memcpy_dto_d_async_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_memcpy_dto_d_async_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMCPY_DTO_D_ASYNC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipDeviceptr_t dst */
        __cmd->dst = dst;
        /* Input: hipDeviceptr_t src */
        __cmd->src = src;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_hip_memcpy_dto_d_async_call_record *__call_record =
        (struct hip_hip_memcpy_dto_d_async_call_record *)calloc(1,
        sizeof(struct hip_hip_memcpy_dto_d_async_call_record));

    __call_record->dst = dst;

    __call_record->src = src;

    __call_record->sizeBytes = sizeBytes;

    __call_record->stream = stream;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemcpyDtoDAsync);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipMemcpySync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipMemcpySync = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __total_buffer_size +=
                    command_channel_buffer_size(__chan,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                }
            }
    }}
    struct hip_nw_hip_memcpy_sync_call *__cmd =
        (struct hip_nw_hip_memcpy_sync_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_memcpy_sync_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_MEMCPY_SYNC;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
                } else {
                    __cmd->dst = NULL;
                }
            } else {
                __cmd->dst = dst;
            }
        }
        /* Input: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyHostToDevice
                && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __cmd->src =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, src,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            } else {
                __cmd->src = NULL;
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyHostToDevice
                    && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    void *__tmp_src_0;
                    __tmp_src_0 =
                        (void *)calloc(1, ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list_nw_hipMemcpySync, __tmp_src_0);
                    const size_t __src_size_0 = ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                        const size_t ava_index = __src_index_0;

                        char *__src_a_0;
                        __src_a_0 = (src) + __src_index_0;

                        char *__src_b_0;
                        __src_b_0 = (__tmp_src_0) + __src_index_0;

                        *__src_b_0 = *__src_a_0;
                    }
                    __cmd->src =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_src_0,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                } else {
                    __cmd->src = NULL;
                }
            } else {
                __cmd->src = src;
            }
        }
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipMemcpyKind kind */
        __cmd->kind = kind;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_nw_hip_memcpy_sync_call_record *__call_record =
        (struct hip_nw_hip_memcpy_sync_call_record *)calloc(1, sizeof(struct hip_nw_hip_memcpy_sync_call_record));

    __call_record->sizeBytes = sizeBytes;

    __call_record->kind = kind;

    __call_record->src = src;

    __call_record->stream = stream;

    __call_record->dst = dst;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipMemcpySync);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipGetDeviceCount(int *count)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipGetDeviceCount = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_get_device_count_call *__cmd =
        (struct hip_hip_get_device_count_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_get_device_count_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_GET_DEVICE_COUNT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int * count */
        if ((count) != (NULL)) {
            __cmd->count = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->count = NULL;
        }
    }

    struct hip_hip_get_device_count_call_record *__call_record =
        (struct hip_hip_get_device_count_call_record *)calloc(1, sizeof(struct hip_hip_get_device_count_call_record));

    __call_record->count = count;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipGetDeviceCount);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipSetDevice(int deviceId)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipSetDevice = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_set_device_call *__cmd =
        (struct hip_nw_hip_set_device_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_set_device_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_SET_DEVICE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int deviceId */
        __cmd->deviceId = deviceId;
    }

    struct hip_nw_hip_set_device_call_record *__call_record =
        (struct hip_nw_hip_set_device_call_record *)calloc(1, sizeof(struct hip_nw_hip_set_device_call_record));

    __call_record->deviceId = deviceId;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipSetDevice);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemGetInfo(size_t * __free, size_t * total)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemGetInfo = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_mem_get_info_call *__cmd =
        (struct hip_hip_mem_get_info_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_mem_get_info_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEM_GET_INFO;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: size_t * __free */
        if ((__free) != (NULL)) {
            __cmd->__free = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->__free = NULL;
        }
        /* Input: size_t * total */
        if ((total) != (NULL)) {
            __cmd->total = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->total = NULL;
        }
    }

    struct hip_hip_mem_get_info_call_record *__call_record =
        (struct hip_hip_mem_get_info_call_record *)calloc(1, sizeof(struct hip_hip_mem_get_info_call_record));

    __call_record->__free = __free;

    __call_record->total = total;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemGetInfo);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipStreamCreate(hipStream_t * stream, hsa_agent_t * agent)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipStreamCreate = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_stream_create_call *__cmd =
        (struct hip_nw_hip_stream_create_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_stream_create_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_STREAM_CREATE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipStream_t * stream */
        if ((stream) != (NULL)) {
            __cmd->stream = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->stream = NULL;
        }
        /* Input: hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            __cmd->agent = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->agent = NULL;
        }
    }

    struct hip_nw_hip_stream_create_call_record *__call_record =
        (struct hip_nw_hip_stream_create_call_record *)calloc(1, sizeof(struct hip_nw_hip_stream_create_call_record));

    __call_record->stream = stream;

    __call_record->agent = agent;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipStreamCreate);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipGetDevice(int *deviceId)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipGetDevice = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_get_device_call *__cmd =
        (struct hip_nw_hip_get_device_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_get_device_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_GET_DEVICE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int * deviceId */
        if ((deviceId) != (NULL)) {
            __cmd->deviceId = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->deviceId = NULL;
        }
    }

    struct hip_nw_hip_get_device_call_record *__call_record =
        (struct hip_nw_hip_get_device_call_record *)calloc(1, sizeof(struct hip_nw_hip_get_device_call_record));

    __call_record->deviceId = deviceId;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipGetDevice);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipInit(unsigned int flags)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipInit = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_init_call *__cmd =
        (struct hip_hip_init_call *)command_channel_new_command(__chan, sizeof(struct hip_hip_init_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_INIT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: unsigned int flags */
        __cmd->flags = flags;
    }

    struct hip_hip_init_call_record *__call_record =
        (struct hip_hip_init_call_record *)calloc(1, sizeof(struct hip_hip_init_call_record));

    __call_record->flags = flags;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipInit); /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipCtxGetCurrent(hipCtx_t * ctx)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipCtxGetCurrent = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_ctx_get_current_call *__cmd =
        (struct hip_hip_ctx_get_current_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_ctx_get_current_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_CTX_GET_CURRENT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipCtx_t * ctx */
        if ((ctx) != (NULL)) {
            __cmd->ctx = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->ctx = NULL;
        }
    }

    struct hip_hip_ctx_get_current_call_record *__call_record =
        (struct hip_hip_ctx_get_current_call_record *)calloc(1, sizeof(struct hip_hip_ctx_get_current_call_record));

    __call_record->ctx = ctx;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipCtxGetCurrent);       /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipStreamSynchronize(hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipStreamSynchronize = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_stream_synchronize_call *__cmd =
        (struct hip_nw_hip_stream_synchronize_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_stream_synchronize_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_STREAM_SYNCHRONIZE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_nw_hip_stream_synchronize_call_record *__call_record =
        (struct hip_nw_hip_stream_synchronize_call_record *)calloc(1,
        sizeof(struct hip_nw_hip_stream_synchronize_call_record));

    __call_record->stream = stream;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipStreamSynchronize);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_hipGetDeviceProperties(char *prop, int deviceId)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hipGetDeviceProperties = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_hip_get_device_properties_call *__cmd =
        (struct hip___do_c_hip_get_device_properties_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hip_get_device_properties_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HIP_GET_DEVICE_PROPERTIES;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: char * prop */
        if ((prop) != (NULL) && (sizeof(hipDeviceProp_t)) > (0)) {
            __cmd->prop = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->prop = NULL;
        }
        /* Input: int deviceId */
        __cmd->deviceId = deviceId;
    }

    struct hip___do_c_hip_get_device_properties_call_record *__call_record =
        (struct hip___do_c_hip_get_device_properties_call_record *)calloc(1,
        sizeof(struct hip___do_c_hip_get_device_properties_call_record));

    __call_record->prop = prop;

    __call_record->deviceId = deviceId;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hipGetDeviceProperties);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_hipHccModuleLaunchKernel(hsa_kernel_dispatch_packet_t * aql, hipStream_t stream, void **kernelParams,
    char *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchKernel = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL)) {
            const size_t __aql_size_0 = (1);
            const size_t __aql_index_0 = 0;
            const size_t ava_index = 0;

            hsa_kernel_dispatch_packet_t *__aql_a_0;
            __aql_a_0 = (aql) + __aql_index_0;

            hsa_kernel_dispatch_packet_t *ava_self;
            ava_self = &*__aql_a_0;
            void **__aql_a_1_kernarg_address;
            __aql_a_1_kernarg_address = &(*__aql_a_0).kernarg_address;
            if ((*__aql_a_1_kernarg_address) != (NULL)) {
                __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(void));
            }
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_kernel_dispatch_packet_t));
        }

        /* Size: void ** kernelParams */
        if ((kernelParams) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(void *));
        }

        /* Size: char * extra */
        if ((extra) != (NULL) && (extra_size) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (extra_size) * sizeof(char));
        }
    }
    struct hip___do_c_hip_hcc_module_launch_kernel_call *__cmd =
        (struct hip___do_c_hip_hcc_module_launch_kernel_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hip_hcc_module_launch_kernel_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_KERNEL;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL)) {
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 = (hsa_kernel_dispatch_packet_t *) calloc(1, (1) * sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchKernel, __tmp_aql_0);
            const size_t __aql_size_0 = (1);
            const size_t __aql_index_0 = 0;
            const size_t ava_index = 0;

            hsa_kernel_dispatch_packet_t *__aql_a_0;
            __aql_a_0 = (aql) + __aql_index_0;

            hsa_kernel_dispatch_packet_t *__aql_b_0;
            __aql_b_0 = (__tmp_aql_0) + __aql_index_0;
            memcpy(__aql_b_0, __aql_a_0, sizeof(hsa_kernel_dispatch_packet_t));
            __cmd->aql =
                (hsa_kernel_dispatch_packet_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_aql_0, (1) * sizeof(hsa_kernel_dispatch_packet_t));
        } else {
            __cmd->aql = NULL;
        }
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
        /* Input: void ** kernelParams */
        if ((kernelParams) != (NULL)) {
            __cmd->kernelParams =
                (void **)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, kernelParams,
                (1) * sizeof(void *));
        } else {
            __cmd->kernelParams = NULL;
        }
        /* Input: char * extra */
        if ((extra) != (NULL) && (extra_size) > (0)) {
            __cmd->extra =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, extra,
                (extra_size) * sizeof(char));
        } else {
            __cmd->extra = NULL;
        }
        /* Input: size_t extra_size */
        __cmd->extra_size = extra_size;
        /* Input: hipEvent_t start */
        __cmd->start = start;
        /* Input: hipEvent_t stop */
        __cmd->stop = stop;
    }

    struct hip___do_c_hip_hcc_module_launch_kernel_call_record *__call_record =
        (struct hip___do_c_hip_hcc_module_launch_kernel_call_record *)calloc(1,
        sizeof(struct hip___do_c_hip_hcc_module_launch_kernel_call_record));

    __call_record->aql = aql;

    __call_record->stream = stream;

    __call_record->kernelParams = kernelParams;

    __call_record->extra_size = extra_size;

    __call_record->start = start;

    __call_record->extra = extra;

    __call_record->stop = stop;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchKernel);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_hipHccModuleLaunchMultiKernel(int numKernels, hsa_kernel_dispatch_packet_t * aql, hipStream_t stream,
    char *all_extra, size_t total_extra_size, size_t * extra_size, hipEvent_t * start, hipEvent_t * stop)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: hipEvent_t * stop */
        if ((stop) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(hipEvent_t));
        }

        /* Size: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL) && (numKernels) > (0)) {
            const size_t __aql_size_0 = (numKernels);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *ava_self;
                ava_self = &*__aql_a_0;
                void **__aql_a_1_kernarg_address;
                __aql_a_1_kernarg_address = &(*__aql_a_0).kernarg_address;
                if ((*__aql_a_1_kernarg_address) != (NULL)) {
                    __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(void));
                }
            } __total_buffer_size +=
                command_channel_buffer_size(__chan, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
        }

        /* Size: hipEvent_t * start */
        if ((start) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(hipEvent_t));
        }

        /* Size: size_t * extra_size */
        if ((extra_size) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(size_t));
        }

        /* Size: char * all_extra */
        if ((all_extra) != (NULL) && (total_extra_size) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (total_extra_size) * sizeof(char));
        }
    }
    struct hip___do_c_hip_hcc_module_launch_multi_kernel_call *__cmd =
        (struct hip___do_c_hip_hcc_module_launch_multi_kernel_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int numKernels */
        __cmd->numKernels = numKernels;
        /* Input: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL) && (numKernels) > (0)) {
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 =
                (hsa_kernel_dispatch_packet_t *) calloc(1, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel, __tmp_aql_0);
            const size_t __aql_size_0 = (numKernels);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_b_0;
                __aql_b_0 = (__tmp_aql_0) + __aql_index_0;
                memcpy(__aql_b_0, __aql_a_0, sizeof(hsa_kernel_dispatch_packet_t));
            }
            __cmd->aql =
                (hsa_kernel_dispatch_packet_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_aql_0, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
        } else {
            __cmd->aql = NULL;
        }
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
        /* Input: char * all_extra */
        if ((all_extra) != (NULL) && (total_extra_size) > (0)) {
            __cmd->all_extra =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, all_extra,
                (total_extra_size) * sizeof(char));
        } else {
            __cmd->all_extra = NULL;
        }
        /* Input: size_t total_extra_size */
        __cmd->total_extra_size = total_extra_size;
        /* Input: size_t * extra_size */
        if ((extra_size) != (NULL) && (numKernels) > (0)) {
            __cmd->extra_size =
                (size_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, extra_size,
                (numKernels) * sizeof(size_t));
        } else {
            __cmd->extra_size = NULL;
        }
        /* Input: hipEvent_t * start */
        if ((start) != (NULL) && (numKernels) > (0)) {
            __cmd->start =
                (hipEvent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, start,
                (numKernels) * sizeof(hipEvent_t));
        } else {
            __cmd->start = NULL;
        }
        /* Input: hipEvent_t * stop */
        if ((stop) != (NULL) && (numKernels) > (0)) {
            __cmd->stop =
                (hipEvent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, stop,
                (numKernels) * sizeof(hipEvent_t));
        } else {
            __cmd->stop = NULL;
        }
    }

    struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record *__call_record =
        (struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record *)calloc(1,
        sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_call_record));

    __call_record->numKernels = numKernels;

    __call_record->stop = stop;

    __call_record->stream = stream;

    __call_record->aql = aql;

    __call_record->start = start;

    __call_record->extra_size = extra_size;

    __call_record->total_extra_size = total_extra_size;

    __call_record->all_extra = all_extra;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_hipHccModuleLaunchMultiKernel_and_memcpy(int numKernels, hsa_kernel_dispatch_packet_t * aql, hipStream_t stream,
    char *all_extra, size_t total_extra_size, size_t * extra_size, hipEvent_t * start, hipEvent_t * stop, void *dst,
    const void *src, size_t sizeBytes, hipMemcpyKind kind)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL) && (numKernels) > (0)) {
            const size_t __aql_size_0 = (numKernels);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *ava_self;
                ava_self = &*__aql_a_0;
                void **__aql_a_1_kernarg_address;
                __aql_a_1_kernarg_address = &(*__aql_a_0).kernarg_address;
                if ((*__aql_a_1_kernarg_address) != (NULL)) {
                    __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(void));
                }
            } __total_buffer_size +=
                command_channel_buffer_size(__chan, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
        }

        /* Size: size_t * extra_size */
        if ((extra_size) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(size_t));
        }

        /* Size: hipEvent_t * stop */
        if ((stop) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(hipEvent_t));
        }

        /* Size: hipEvent_t * start */
        if ((start) != (NULL) && (numKernels) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (numKernels) * sizeof(hipEvent_t));
        }

        /* Size: char * all_extra */
        if ((all_extra) != (NULL) && (total_extra_size) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (total_extra_size) * sizeof(char));
        }

        /* Size: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __total_buffer_size +=
                    command_channel_buffer_size(__chan,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    __total_buffer_size +=
                        command_channel_buffer_size(__chan,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                }
            }
    }}
    struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call *__cmd =
        (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HIP_HCC_MODULE_LAUNCH_MULTI_KERNEL_AND_MEMCPY;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int numKernels */
        __cmd->numKernels = numKernels;
        /* Input: hsa_kernel_dispatch_packet_t * aql */
        if ((aql) != (NULL) && (numKernels) > (0)) {
            hsa_kernel_dispatch_packet_t *__tmp_aql_0;
            __tmp_aql_0 =
                (hsa_kernel_dispatch_packet_t *) calloc(1, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
            g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, __tmp_aql_0);
            const size_t __aql_size_0 = (numKernels);
            for (size_t __aql_index_0 = 0; __aql_index_0 < __aql_size_0; __aql_index_0++) {
                const size_t ava_index = __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_a_0;
                __aql_a_0 = (aql) + __aql_index_0;

                hsa_kernel_dispatch_packet_t *__aql_b_0;
                __aql_b_0 = (__tmp_aql_0) + __aql_index_0;
                memcpy(__aql_b_0, __aql_a_0, sizeof(hsa_kernel_dispatch_packet_t));
            }
            __cmd->aql =
                (hsa_kernel_dispatch_packet_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_aql_0, (numKernels) * sizeof(hsa_kernel_dispatch_packet_t));
        } else {
            __cmd->aql = NULL;
        }
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
        /* Input: char * all_extra */
        if ((all_extra) != (NULL) && (total_extra_size) > (0)) {
            __cmd->all_extra =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, all_extra,
                (total_extra_size) * sizeof(char));
        } else {
            __cmd->all_extra = NULL;
        }
        /* Input: size_t total_extra_size */
        __cmd->total_extra_size = total_extra_size;
        /* Input: size_t * extra_size */
        if ((extra_size) != (NULL) && (numKernels) > (0)) {
            __cmd->extra_size =
                (size_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, extra_size,
                (numKernels) * sizeof(size_t));
        } else {
            __cmd->extra_size = NULL;
        }
        /* Input: hipEvent_t * start */
        if ((start) != (NULL) && (numKernels) > (0)) {
            __cmd->start =
                (hipEvent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, start,
                (numKernels) * sizeof(hipEvent_t));
        } else {
            __cmd->start = NULL;
        }
        /* Input: hipEvent_t * stop */
        if ((stop) != (NULL) && (numKernels) > (0)) {
            __cmd->stop =
                (hipEvent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, stop,
                (numKernels) * sizeof(hipEvent_t));
        } else {
            __cmd->stop = NULL;
        }
        /* Input: void * dst */
        if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyDeviceToHost
                && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dst = NULL;
            }
        } else {
            if (((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyDeviceToHost
                    && ((kind == hipMemcpyDeviceToHost) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (dst) != (NULL)
                    && ((kind == hipMemcpyDeviceToHost) ? (sizeBytes) : (0)) > (0)) {
                    __cmd->dst = HAS_OUT_BUFFER_SENTINEL;
                } else {
                    __cmd->dst = NULL;
                }
            } else {
                __cmd->dst = dst;
            }
        }
        /* Input: const void * src */
        if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
            if (kind == hipMemcpyHostToDevice
                && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                __cmd->src =
                    (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, src,
                    ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
            } else {
                __cmd->src = NULL;
            }
        } else {
            if (((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER)) {
                if (kind == hipMemcpyHostToDevice
                    && ((kind == hipMemcpyHostToDevice) ? (NW_BUFFER) : (NW_OPAQUE)) == (NW_BUFFER) && (src) != (NULL)
                    && ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) > (0)) {
                    void *__tmp_src_0;
                    __tmp_src_0 =
                        (void *)calloc(1, ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                    g_ptr_array_add(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy, __tmp_src_0);
                    const size_t __src_size_0 = ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0));
                    for (size_t __src_index_0 = 0; __src_index_0 < __src_size_0; __src_index_0++) {
                        const size_t ava_index = __src_index_0;

                        char *__src_a_0;
                        __src_a_0 = (src) + __src_index_0;

                        char *__src_b_0;
                        __src_b_0 = (__tmp_src_0) + __src_index_0;

                        *__src_b_0 = *__src_a_0;
                    }
                    __cmd->src =
                        (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_src_0,
                        ((kind == hipMemcpyHostToDevice) ? (sizeBytes) : (0)) * sizeof(const void));
                } else {
                    __cmd->src = NULL;
                }
            } else {
                __cmd->src = src;
            }
        }
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
        /* Input: hipMemcpyKind kind */
        __cmd->kind = kind;
    }

    struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call_record *__call_record =
        (struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call_record *)calloc(1,
        sizeof(struct hip___do_c_hip_hcc_module_launch_multi_kernel_and_memcpy_call_record));

    __call_record->numKernels = numKernels;

    __call_record->aql = aql;

    __call_record->extra_size = extra_size;

    __call_record->stop = stop;

    __call_record->stream = stream;

    __call_record->start = start;

    __call_record->total_extra_size = total_extra_size;

    __call_record->sizeBytes = sizeBytes;

    __call_record->all_extra = all_extra;

    __call_record->kind = kind;

    __call_record->dst = dst;

    __call_record->src = src;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hipHccModuleLaunchMultiKernel_and_memcpy);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
nw_hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major, uint16_t * version_minor,
    _Bool * result)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hsa_system_major_extension_supported = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: uint16_t * version_minor */
        if ((version_minor) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(uint16_t));
        }
    }
    struct hip_nw_hsa_system_major_extension_supported_call *__cmd =
        (struct hip_nw_hsa_system_major_extension_supported_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hsa_system_major_extension_supported_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HSA_SYSTEM_MAJOR_EXTENSION_SUPPORTED;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: uint16_t extension */
        __cmd->extension = extension;
        /* Input: uint16_t version_major */
        __cmd->version_major = version_major;
        /* Input: uint16_t * version_minor */
        if ((version_minor) != (NULL)) {
            __cmd->version_minor =
                (uint16_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, version_minor,
                (1) * sizeof(uint16_t));
        } else {
            if ((version_minor) != (NULL)) {
                __cmd->version_minor = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->version_minor = NULL;
            }
        }
        /* Input: _Bool * result */
        if ((result) != (NULL)) {
            __cmd->result = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->result = NULL;
        }
    }

    struct hip_nw_hsa_system_major_extension_supported_call_record *__call_record =
        (struct hip_nw_hsa_system_major_extension_supported_call_record *)calloc(1,
        sizeof(struct hip_nw_hsa_system_major_extension_supported_call_record));

    __call_record->extension = extension;

    __call_record->version_major = version_major;

    __call_record->version_minor = version_minor;

    __call_record->result = result;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hsa_system_major_extension_supported);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
nw_hsa_executable_create_alt(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options, hsa_executable_t * executable)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hsa_executable_create_alt = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * options */
        if ((options) != (NULL) && (strlen(options) + 1) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (strlen(options) + 1) * sizeof(const char));
        }
    }
    struct hip_nw_hsa_executable_create_alt_call *__cmd =
        (struct hip_nw_hsa_executable_create_alt_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hsa_executable_create_alt_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HSA_EXECUTABLE_CREATE_ALT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_profile_t profile */
        __cmd->profile = profile;
        /* Input: hsa_default_float_rounding_mode_t default_float_rounding_mode */
        __cmd->default_float_rounding_mode = default_float_rounding_mode;
        /* Input: const char * options */
        if ((options) != (NULL) && (strlen(options) + 1) > (0)) {
            __cmd->options =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, options,
                (strlen(options) + 1) * sizeof(const char));
        } else {
            __cmd->options = NULL;
        }
        /* Input: hsa_executable_t * executable */
        if ((executable) != (NULL)) {
            __cmd->executable = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->executable = NULL;
        }
    }

    struct hip_nw_hsa_executable_create_alt_call_record *__call_record =
        (struct hip_nw_hsa_executable_create_alt_call_record *)calloc(1,
        sizeof(struct hip_nw_hsa_executable_create_alt_call_record));

    __call_record->profile = profile;

    __call_record->options = options;

    __call_record->default_float_rounding_mode = default_float_rounding_mode;

    __call_record->executable = executable;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hsa_executable_create_alt);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
nw_hsa_isa_from_name(const char *name, hsa_isa_t * isa)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hsa_isa_from_name = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (strlen(name) + 1) * sizeof(const char));
        }
    }
    struct hip_nw_hsa_isa_from_name_call *__cmd =
        (struct hip_nw_hsa_isa_from_name_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hsa_isa_from_name_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HSA_ISA_FROM_NAME;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __cmd->name =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, name,
                (strlen(name) + 1) * sizeof(const char));
        } else {
            __cmd->name = NULL;
        }
        /* Input: hsa_isa_t * isa */
        if ((isa) != (NULL)) {
            __cmd->isa = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->isa = NULL;
        }
    }

    struct hip_nw_hsa_isa_from_name_call_record *__call_record =
        (struct hip_nw_hsa_isa_from_name_call_record *)calloc(1, sizeof(struct hip_nw_hsa_isa_from_name_call_record));

    __call_record->isa = isa;

    __call_record->name = name;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hsa_isa_from_name);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipPeekAtLastError()
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipPeekAtLastError = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_peek_at_last_error_call *__cmd =
        (struct hip_hip_peek_at_last_error_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_peek_at_last_error_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_PEEK_AT_LAST_ERROR;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

    }

    struct hip_hip_peek_at_last_error_call_record *__call_record =
        (struct hip_hip_peek_at_last_error_call_record *)calloc(1,
        sizeof(struct hip_hip_peek_at_last_error_call_record));

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipPeekAtLastError);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipDeviceGetAttribute = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_device_get_attribute_call *__cmd =
        (struct hip_nw_hip_device_get_attribute_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_device_get_attribute_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_DEVICE_GET_ATTRIBUTE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: int * pi */
        if ((pi) != (NULL)) {
            __cmd->pi = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->pi = NULL;
        }
        /* Input: hipDeviceAttribute_t attr */
        __cmd->attr = attr;
        /* Input: int deviceId */
        __cmd->deviceId = deviceId;
    }

    struct hip_nw_hip_device_get_attribute_call_record *__call_record =
        (struct hip_nw_hip_device_get_attribute_call_record *)calloc(1,
        sizeof(struct hip_nw_hip_device_get_attribute_call_record));

    __call_record->pi = pi;

    __call_record->attr = attr;

    __call_record->deviceId = deviceId;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipDeviceGetAttribute);       /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipModuleLoadData(hipModule_t * module, const void *image)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipModuleLoadData = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const void * image */
        if ((image) != (NULL) && (calc_image_size(image)) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (calc_image_size(image)) * sizeof(const void));
        }
    }
    struct hip_hip_module_load_data_call *__cmd =
        (struct hip_hip_module_load_data_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_module_load_data_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MODULE_LOAD_DATA;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipModule_t * module */
        if ((module) != (NULL)) {
            __cmd->module = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->module = NULL;
        }
        /* Input: const void * image */
        if ((image) != (NULL) && (calc_image_size(image)) > (0)) {
            __cmd->image =
                (void *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, image,
                (calc_image_size(image)) * sizeof(const void));
        } else {
            __cmd->image = NULL;
        }
    }

    struct hip_hip_module_load_data_call_record *__call_record =
        (struct hip_hip_module_load_data_call_record *)calloc(1, sizeof(struct hip_hip_module_load_data_call_record));

    __call_record->image = image;

    __call_record->module = module;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipModuleLoadData);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
__do_c_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute,
    char *value, size_t max_value)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hsa_executable_symbol_get_info = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_hsa_executable_symbol_get_info_call *__cmd =
        (struct hip___do_c_hsa_executable_symbol_get_info_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hsa_executable_symbol_get_info_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HSA_EXECUTABLE_SYMBOL_GET_INFO;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_executable_symbol_t executable_symbol */

        hsa_executable_symbol_t *ava_self;
        ava_self = &executable_symbol;
        uint64_t *__executable_symbol_a_0_handle;
        __executable_symbol_a_0_handle = &(executable_symbol).handle;
        uint64_t *__executable_symbol_b_0_handle;
        __executable_symbol_b_0_handle = &(__cmd->executable_symbol).handle;
        *__executable_symbol_b_0_handle = *__executable_symbol_a_0_handle;
        /* Input: hsa_executable_symbol_info_t attribute */
        __cmd->attribute = attribute;
        /* Input: char * value */
        if ((value) != (NULL) && (max_value) > (0)) {
            __cmd->value = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->value = NULL;
        }
        /* Input: size_t max_value */
        __cmd->max_value = max_value;
    }

    struct hip___do_c_hsa_executable_symbol_get_info_call_record *__call_record =
        (struct hip___do_c_hsa_executable_symbol_get_info_call_record *)calloc(1,
        sizeof(struct hip___do_c_hsa_executable_symbol_get_info_call_record));

    __call_record->executable_symbol = executable_symbol;

    __call_record->attribute = attribute;

    __call_record->max_value = max_value;

    __call_record->value = value;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hsa_executable_symbol_get_info);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipCtxSetCurrent(hipCtx_t ctx)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipCtxSetCurrent = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_ctx_set_current_call *__cmd =
        (struct hip_nw_hip_ctx_set_current_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_ctx_set_current_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_CTX_SET_CURRENT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipCtx_t ctx */
        __cmd->ctx = ctx;
    }

    struct hip_nw_hip_ctx_set_current_call_record *__call_record =
        (struct hip_nw_hip_ctx_set_current_call_record *)calloc(1,
        sizeof(struct hip_nw_hip_ctx_set_current_call_record));

    __call_record->ctx = ctx;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipCtxSetCurrent);    /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipEventCreate(hipEvent_t * event)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipEventCreate = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_event_create_call *__cmd =
        (struct hip_hip_event_create_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_event_create_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_EVENT_CREATE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipEvent_t * event */
        if ((event) != (NULL)) {
            __cmd->event = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->event = NULL;
        }
    }

    struct hip_hip_event_create_call_record *__call_record =
        (struct hip_hip_event_create_call_record *)calloc(1, sizeof(struct hip_hip_event_create_call_record));

    __call_record->event = event;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipEventCreate); /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipEventRecord = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_event_record_call *__cmd =
        (struct hip_hip_event_record_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_event_record_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_EVENT_RECORD;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipEvent_t event */
        __cmd->event = event;
        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_hip_event_record_call_record *__call_record =
        (struct hip_hip_event_record_call_record *)calloc(1, sizeof(struct hip_hip_event_record_call_record));

    __call_record->event = event;

    __call_record->stream = stream;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipEventRecord); /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipEventSynchronize(hipEvent_t event)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipEventSynchronize = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_event_synchronize_call *__cmd =
        (struct hip_hip_event_synchronize_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_event_synchronize_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_EVENT_SYNCHRONIZE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipEvent_t event */
        __cmd->event = event;
    }

    struct hip_hip_event_synchronize_call_record *__call_record =
        (struct hip_hip_event_synchronize_call_record *)calloc(1, sizeof(struct hip_hip_event_synchronize_call_record));

    __call_record->event = event;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipEventSynchronize);    /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipEventDestroy(hipEvent_t event)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipEventDestroy = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_event_destroy_call *__cmd =
        (struct hip_hip_event_destroy_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_event_destroy_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_EVENT_DESTROY;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipEvent_t event */
        __cmd->event = event;
    }

    struct hip_hip_event_destroy_call_record *__call_record =
        (struct hip_hip_event_destroy_call_record *)calloc(1, sizeof(struct hip_hip_event_destroy_call_record));

    __call_record->event = event;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipEventDestroy);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipEventElapsedTime = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_event_elapsed_time_call *__cmd =
        (struct hip_hip_event_elapsed_time_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_event_elapsed_time_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_EVENT_ELAPSED_TIME;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: float * ms */
        if ((ms) != (NULL)) {
            __cmd->ms = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->ms = NULL;
        }
        /* Input: hipEvent_t start */
        __cmd->start = start;
        /* Input: hipEvent_t stop */
        __cmd->stop = stop;
    }

    struct hip_hip_event_elapsed_time_call_record *__call_record =
        (struct hip_hip_event_elapsed_time_call_record *)calloc(1,
        sizeof(struct hip_hip_event_elapsed_time_call_record));

    __call_record->ms = ms;

    __call_record->start = start;

    __call_record->stop = stop;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipEventElapsedTime);    /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipModuleLoad(hipModule_t * module, const char *fname)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipModuleLoad = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * fname */
        if ((fname) != (NULL) && (strlen(fname) + 1) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (strlen(fname) + 1) * sizeof(const char));
        }
    }
    struct hip_hip_module_load_call *__cmd =
        (struct hip_hip_module_load_call *)command_channel_new_command(__chan, sizeof(struct hip_hip_module_load_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MODULE_LOAD;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipModule_t * module */
        if ((module) != (NULL)) {
            __cmd->module = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->module = NULL;
        }
        /* Input: const char * fname */
        if ((fname) != (NULL) && (strlen(fname) + 1) > (0)) {
            __cmd->fname =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, fname,
                (strlen(fname) + 1) * sizeof(const char));
        } else {
            __cmd->fname = NULL;
        }
    }

    struct hip_hip_module_load_call_record *__call_record =
        (struct hip_hip_module_load_call_record *)calloc(1, sizeof(struct hip_hip_module_load_call_record));

    __call_record->fname = fname;

    __call_record->module = module;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipModuleLoad);  /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipModuleUnload(hipModule_t module)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipModuleUnload = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_module_unload_call *__cmd =
        (struct hip_hip_module_unload_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_module_unload_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MODULE_UNLOAD;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipModule_t module */
        __cmd->module = module;
    }

    struct hip_hip_module_unload_call_record *__call_record =
        (struct hip_hip_module_unload_call_record *)calloc(1, sizeof(struct hip_hip_module_unload_call_record));

    __call_record->module = module;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipModuleUnload);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipStreamDestroy(hipStream_t stream)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipStreamDestroy = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_stream_destroy_call *__cmd =
        (struct hip_nw_hip_stream_destroy_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_stream_destroy_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_STREAM_DESTROY;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipStream_t stream */
        __cmd->stream = stream;
    }

    struct hip_nw_hip_stream_destroy_call_record *__call_record =
        (struct hip_nw_hip_stream_destroy_call_record *)calloc(1, sizeof(struct hip_nw_hip_stream_destroy_call_record));

    __call_record->stream = stream;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipStreamDestroy);    /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char *kname)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipModuleGetFunction = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * kname */
        if ((kname) != (NULL) && (strlen(kname) + 1) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (strlen(kname) + 1) * sizeof(const char));
        }
    }
    struct hip_hip_module_get_function_call *__cmd =
        (struct hip_hip_module_get_function_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_module_get_function_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MODULE_GET_FUNCTION;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipFunction_t * function */
        if ((function) != (NULL)) {
            __cmd->function = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->function = NULL;
        }
        /* Input: hipModule_t module */
        __cmd->module = module;
        /* Input: const char * kname */
        if ((kname) != (NULL) && (strlen(kname) + 1) > (0)) {
            __cmd->kname =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, kname,
                (strlen(kname) + 1) * sizeof(const char));
        } else {
            __cmd->kname = NULL;
        }
    }

    struct hip_hip_module_get_function_call_record *__call_record =
        (struct hip_hip_module_get_function_call_record *)calloc(1,
        sizeof(struct hip_hip_module_get_function_call_record));

    __call_record->function = function;

    __call_record->kname = kname;

    __call_record->module = module;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipModuleGetFunction);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipGetLastError()
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipGetLastError = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_get_last_error_call *__cmd =
        (struct hip_hip_get_last_error_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_get_last_error_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_GET_LAST_ERROR;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

    }

    struct hip_hip_get_last_error_call_record *__call_record =
        (struct hip_hip_get_last_error_call_record *)calloc(1, sizeof(struct hip_hip_get_last_error_call_record));

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipGetLastError);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipMemset(void *dst, int value, size_t sizeBytes)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipMemset = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_memset_call *__cmd =
        (struct hip_hip_memset_call *)command_channel_new_command(__chan, sizeof(struct hip_hip_memset_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_MEMSET;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: void * dst */
        __cmd->dst = dst;
        /* Input: int value */
        __cmd->value = value;
        /* Input: size_t sizeBytes */
        __cmd->sizeBytes = sizeBytes;
    }

    struct hip_hip_memset_call_record *__call_record =
        (struct hip_hip_memset_call_record *)calloc(1, sizeof(struct hip_hip_memset_call_record));

    __call_record->dst = dst;

    __call_record->value = value;

    __call_record->sizeBytes = sizeBytes;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipMemset);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_hipStreamWaitEvent = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_hip_stream_wait_event_call *__cmd =
        (struct hip_hip_stream_wait_event_call *)command_channel_new_command(__chan,
        sizeof(struct hip_hip_stream_wait_event_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_HIP_STREAM_WAIT_EVENT;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipStream_t stream */
        __cmd->stream = stream;
        /* Input: hipEvent_t event */
        __cmd->event = event;
        /* Input: unsigned int flags */
        __cmd->flags = flags;
    }

    struct hip_hip_stream_wait_event_call_record *__call_record =
        (struct hip_hip_stream_wait_event_call_record *)calloc(1, sizeof(struct hip_hip_stream_wait_event_call_record));

    __call_record->stream = stream;

    __call_record->event = event;

    __call_record->flags = flags;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_hipStreamWaitEvent);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
__do_c_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void *value, size_t max_value)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_hsa_agent_get_info = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_hsa_agent_get_info_call *__cmd =
        (struct hip___do_c_hsa_agent_get_info_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_hsa_agent_get_info_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_HSA_AGENT_GET_INFO;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_agent_t agent */

        hsa_agent_t *ava_self;
        ava_self = &agent;
        uint64_t *__agent_a_0_handle;
        __agent_a_0_handle = &(agent).handle;
        uint64_t *__agent_b_0_handle;
        __agent_b_0_handle = &(__cmd->agent).handle;
        *__agent_b_0_handle = *__agent_a_0_handle;
        /* Input: hsa_agent_info_t attribute */
        __cmd->attribute = attribute;
        /* Input: void * value */
        if ((value) != (NULL) && (max_value) > (0)) {
            __cmd->value = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->value = NULL;
        }
        /* Input: size_t max_value */
        __cmd->max_value = max_value;
    }

    struct hip___do_c_hsa_agent_get_info_call_record *__call_record =
        (struct hip___do_c_hsa_agent_get_info_call_record *)calloc(1,
        sizeof(struct hip___do_c_hsa_agent_get_info_call_record));

    __call_record->agent = agent;

    __call_record->attribute = attribute;

    __call_record->max_value = max_value;

    __call_record->value = value;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_hsa_agent_get_info);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED int
__do_c_load_executable(const char *file_buf, size_t file_len, hsa_executable_t * executable, hsa_agent_t * agent)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_load_executable = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * file_buf */
        if ((file_buf) != (NULL) && (file_len) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (file_len) * sizeof(const char));
        }

        /* Size: hsa_executable_t * executable */
        if ((executable) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_executable_t));
        }

        /* Size: hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(hsa_agent_t));
        }
    }
    struct hip___do_c_load_executable_call *__cmd =
        (struct hip___do_c_load_executable_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_load_executable_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_LOAD_EXECUTABLE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: const char * file_buf */
        if ((file_buf) != (NULL) && (file_len) > (0)) {
            __cmd->file_buf =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, file_buf,
                (file_len) * sizeof(const char));
        } else {
            __cmd->file_buf = NULL;
        }
        /* Input: size_t file_len */
        __cmd->file_len = file_len;
        /* Input: hsa_executable_t * executable */
        if ((executable) != (NULL)) {
            hsa_executable_t *__tmp_executable_0;
            __tmp_executable_0 = (hsa_executable_t *) calloc(1, (1) * sizeof(hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list___do_c_load_executable, __tmp_executable_0);
            const size_t __executable_size_0 = (1);
            const size_t __executable_index_0 = 0;
            const size_t ava_index = 0;

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
            *__executable_b_1_handle = *__executable_a_1_handle;
            __cmd->executable =
                (hsa_executable_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_executable_0, (1) * sizeof(hsa_executable_t));
        } else {
            if ((executable) != (NULL)) {
                __cmd->executable = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->executable = NULL;
            }
        }
        /* Input: hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            hsa_agent_t *__tmp_agent_0;
            __tmp_agent_0 = (hsa_agent_t *) calloc(1, (1) * sizeof(hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_load_executable, __tmp_agent_0);
            const size_t __agent_size_0 = (1);
            const size_t __agent_index_0 = 0;
            const size_t ava_index = 0;

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
            *__agent_b_1_handle = *__agent_a_1_handle;
            __cmd->agent =
                (hsa_agent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_agent_0,
                (1) * sizeof(hsa_agent_t));
        } else {
            __cmd->agent = NULL;
        }
    }

    struct hip___do_c_load_executable_call_record *__call_record =
        (struct hip___do_c_load_executable_call_record *)calloc(1,
        sizeof(struct hip___do_c_load_executable_call_record));

    __call_record->file_len = file_len;

    __call_record->file_buf = file_buf;

    __call_record->executable = executable;

    __call_record->agent = agent;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_load_executable); /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    int ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED size_t
__do_c_get_agents(hsa_agent_t * agents, size_t max_agents)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_get_agents = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_get_agents_call *__cmd =
        (struct hip___do_c_get_agents_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_get_agents_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_GET_AGENTS;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_agent_t * agents */
        if ((agents) != (NULL) && (max_agents) > (0)) {
            __cmd->agents = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->agents = NULL;
        }
        /* Input: size_t max_agents */
        __cmd->max_agents = max_agents;
    }

    struct hip___do_c_get_agents_call_record *__call_record =
        (struct hip___do_c_get_agents_call_record *)calloc(1, sizeof(struct hip___do_c_get_agents_call_record));

    __call_record->max_agents = max_agents;

    __call_record->agents = agents;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_get_agents);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    size_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED size_t
__do_c_get_isas(hsa_agent_t agents, hsa_isa_t * isas, size_t max_isas)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_get_isas = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_get_isas_call *__cmd =
        (struct hip___do_c_get_isas_call *)command_channel_new_command(__chan, sizeof(struct hip___do_c_get_isas_call),
        __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_GET_ISAS;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hsa_agent_t agents */

        hsa_agent_t *ava_self;
        ava_self = &agents;
        uint64_t *__agents_a_0_handle;
        __agents_a_0_handle = &(agents).handle;
        uint64_t *__agents_b_0_handle;
        __agents_b_0_handle = &(__cmd->agents).handle;
        *__agents_b_0_handle = *__agents_a_0_handle;
        /* Input: hsa_isa_t * isas */
        if ((isas) != (NULL) && (max_isas) > (0)) {
            __cmd->isas = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->isas = NULL;
        }
        /* Input: size_t max_isas */
        __cmd->max_isas = max_isas;
    }

    struct hip___do_c_get_isas_call_record *__call_record =
        (struct hip___do_c_get_isas_call_record *)calloc(1, sizeof(struct hip___do_c_get_isas_call_record));

    __call_record->agents = agents;

    __call_record->max_isas = max_isas;

    __call_record->isas = isas;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_get_isas);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    size_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED size_t
__do_c_get_kerenel_symbols(const hsa_executable_t * exec, const hsa_agent_t * agent, hsa_executable_symbol_t * symbols,
    size_t max_symbols)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_get_kerenel_symbols = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const hsa_executable_t * exec */
        if ((exec) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(const hsa_executable_t));
        }

        /* Size: const hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(const hsa_agent_t));
        }
    }
    struct hip___do_c_get_kerenel_symbols_call *__cmd =
        (struct hip___do_c_get_kerenel_symbols_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_get_kerenel_symbols_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_GET_KERENEL_SYMBOLS;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: const hsa_executable_t * exec */
        if ((exec) != (NULL)) {
            hsa_executable_t *__tmp_exec_0;
            __tmp_exec_0 = (hsa_executable_t *) calloc(1, (1) * sizeof(const hsa_executable_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, __tmp_exec_0);
            const size_t __exec_size_0 = (1);
            const size_t __exec_index_0 = 0;
            const size_t ava_index = 0;

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
            *__exec_b_1_handle = *__exec_a_1_handle;
            __cmd->exec =
                (hsa_executable_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_exec_0,
                (1) * sizeof(const hsa_executable_t));
        } else {
            __cmd->exec = NULL;
        }
        /* Input: const hsa_agent_t * agent */
        if ((agent) != (NULL)) {
            hsa_agent_t *__tmp_agent_0;
            __tmp_agent_0 = (hsa_agent_t *) calloc(1, (1) * sizeof(const hsa_agent_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kerenel_symbols, __tmp_agent_0);
            const size_t __agent_size_0 = (1);
            const size_t __agent_index_0 = 0;
            const size_t ava_index = 0;

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
            *__agent_b_1_handle = *__agent_a_1_handle;
            __cmd->agent =
                (hsa_agent_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd, __tmp_agent_0,
                (1) * sizeof(const hsa_agent_t));
        } else {
            __cmd->agent = NULL;
        }
        /* Input: hsa_executable_symbol_t * symbols */
        if ((symbols) != (NULL) && (max_symbols) > (0)) {
            __cmd->symbols = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->symbols = NULL;
        }
        /* Input: size_t max_symbols */
        __cmd->max_symbols = max_symbols;
    }

    struct hip___do_c_get_kerenel_symbols_call_record *__call_record =
        (struct hip___do_c_get_kerenel_symbols_call_record *)calloc(1,
        sizeof(struct hip___do_c_get_kerenel_symbols_call_record));

    __call_record->exec = exec;

    __call_record->agent = agent;

    __call_record->max_symbols = max_symbols;

    __call_record->symbols = symbols;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_get_kerenel_symbols);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    size_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hsa_status_t
__do_c_query_host_address(uint64_t kernel_object_, char *kernel_header_)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_query_host_address = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip___do_c_query_host_address_call *__cmd =
        (struct hip___do_c_query_host_address_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_query_host_address_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_QUERY_HOST_ADDRESS;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: uint64_t kernel_object_ */
        __cmd->kernel_object_ = kernel_object_;
        /* Input: char * kernel_header_ */
        if ((kernel_header_) != (NULL) && (sizeof(amd_kernel_code_t)) > (0)) {
            __cmd->kernel_header_ = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->kernel_header_ = NULL;
        }
    }

    struct hip___do_c_query_host_address_call_record *__call_record =
        (struct hip___do_c_query_host_address_call_record *)calloc(1,
        sizeof(struct hip___do_c_query_host_address_call_record));

    __call_record->kernel_object_ = kernel_object_;

    __call_record->kernel_header_ = kernel_header_;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_query_host_address);      /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hsa_status_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_get_kernel_descriptor(const hsa_executable_symbol_t * symbol, const char *name, hipFunction_t * f)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_get_kernel_descriptor = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (strlen(name) + 1) * sizeof(const char));
        }

        /* Size: const hsa_executable_symbol_t * symbol */
        if ((symbol) != (NULL)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (1) * sizeof(const hsa_executable_symbol_t));
        }
    }
    struct hip___do_c_get_kernel_descriptor_call *__cmd =
        (struct hip___do_c_get_kernel_descriptor_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_get_kernel_descriptor_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_GET_KERNEL_DESCRIPTOR;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: const hsa_executable_symbol_t * symbol */
        if ((symbol) != (NULL)) {
            hsa_executable_symbol_t *__tmp_symbol_0;
            __tmp_symbol_0 = (hsa_executable_symbol_t *) calloc(1, (1) * sizeof(const hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_get_kernel_descriptor, __tmp_symbol_0);
            const size_t __symbol_size_0 = (1);
            const size_t __symbol_index_0 = 0;
            const size_t ava_index = 0;

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
            *__symbol_b_1_handle = *__symbol_a_1_handle;
            __cmd->symbol =
                (hsa_executable_symbol_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_symbol_0, (1) * sizeof(const hsa_executable_symbol_t));
        } else {
            __cmd->symbol = NULL;
        }
        /* Input: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __cmd->name =
                (char *)command_channel_attach_buffer(__chan, (struct command_base *)__cmd, name,
                (strlen(name) + 1) * sizeof(const char));
        } else {
            __cmd->name = NULL;
        }
        /* Input: hipFunction_t * f */
        if ((f) != (NULL)) {
            __cmd->f = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->f = NULL;
        }
    }

    struct hip___do_c_get_kernel_descriptor_call_record *__call_record =
        (struct hip___do_c_get_kernel_descriptor_call_record *)calloc(1,
        sizeof(struct hip___do_c_get_kernel_descriptor_call_record));

    __call_record->name = name;

    __call_record->symbol = symbol;

    __call_record->f = f;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_get_kernel_descriptor);   /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_hipCtxGetDevice(hipDevice_t * device)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_hipCtxGetDevice = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_hip_ctx_get_device_call *__cmd =
        (struct hip_nw_hip_ctx_get_device_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_hip_ctx_get_device_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_HIP_CTX_GET_DEVICE;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipDevice_t * device */
        if ((device) != (NULL)) {
            __cmd->device = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->device = NULL;
        }
    }

    struct hip_nw_hip_ctx_get_device_call_record *__call_record =
        (struct hip_nw_hip_ctx_get_device_call_record *)calloc(1, sizeof(struct hip_nw_hip_ctx_get_device_call_record));

    __call_record->device = device;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_hipCtxGetDevice);     /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
nw_lookup_kern_info(hipFunction_t f, struct nw_kern_info * info)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list_nw_lookup_kern_info = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
    }
    struct hip_nw_lookup_kern_info_call *__cmd =
        (struct hip_nw_lookup_kern_info_call *)command_channel_new_command(__chan,
        sizeof(struct hip_nw_lookup_kern_info_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP_NW_LOOKUP_KERN_INFO;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: hipFunction_t f */
        __cmd->f = f;
        /* Input: struct nw_kern_info * info */
        if ((info) != (NULL)) {
            __cmd->info = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->info = NULL;
        }
    }

    struct hip_nw_lookup_kern_info_call_record *__call_record =
        (struct hip_nw_lookup_kern_info_call_record *)calloc(1, sizeof(struct hip_nw_lookup_kern_info_call_record));

    __call_record->f = f;

    __call_record->info = info;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list_nw_lookup_kern_info);    /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
__do_c_mass_symbol_info(size_t n, const hsa_executable_symbol_t * syms, hsa_symbol_kind_t * types,
    hipFunction_t * descriptors, uint8_t * agents, unsigned int *offsets, char *pool, size_t pool_size)
{
    const int ava_is_in = 1,
        ava_is_out = 0;
    pthread_once(&guestlib_init, init_hip_guestlib);
    GPtrArray *__ava_alloc_list___do_c_mass_symbol_info = g_ptr_array_new_full(0, free);

    size_t __total_buffer_size = 0; {
        /* Size: const hsa_executable_symbol_t * syms */
        if ((syms) != (NULL) && (n) > (0)) {
            __total_buffer_size += command_channel_buffer_size(__chan, (n) * sizeof(const hsa_executable_symbol_t));
        }
    }
    struct hip___do_c_mass_symbol_info_call *__cmd =
        (struct hip___do_c_mass_symbol_info_call *)command_channel_new_command(__chan,
        sizeof(struct hip___do_c_mass_symbol_info_call), __total_buffer_size);
    __cmd->base.api_id = HIP_API;
    __cmd->base.command_id = CALL_HIP___DO_C_MASS_SYMBOL_INFO;

    intptr_t __call_id = ava_get_call_id();
    __cmd->__call_id = __call_id;

    {

        /* Input: size_t n */
        __cmd->n = n;
        /* Input: const hsa_executable_symbol_t * syms */
        if ((syms) != (NULL) && (n) > (0)) {
            hsa_executable_symbol_t *__tmp_syms_0;
            __tmp_syms_0 = (hsa_executable_symbol_t *) calloc(1, (n) * sizeof(const hsa_executable_symbol_t));
            g_ptr_array_add(__ava_alloc_list___do_c_mass_symbol_info, __tmp_syms_0);
            const size_t __syms_size_0 = (n);
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
                *__syms_b_1_handle = *__syms_a_1_handle;
            }
            __cmd->syms =
                (hsa_executable_symbol_t *) command_channel_attach_buffer(__chan, (struct command_base *)__cmd,
                __tmp_syms_0, (n) * sizeof(const hsa_executable_symbol_t));
        } else {
            __cmd->syms = NULL;
        }
        /* Input: hsa_symbol_kind_t * types */
        if ((types) != (NULL) && (n) > (0)) {
            __cmd->types = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->types = NULL;
        }
        /* Input: hipFunction_t * descriptors */
        if ((descriptors) != (NULL) && (n) > (0)) {
            __cmd->descriptors = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->descriptors = NULL;
        }
        /* Input: uint8_t * agents */
        if ((agents) != (NULL) && (n * sizeof(agents)) > (0)) {
            __cmd->agents = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->agents = NULL;
        }
        /* Input: unsigned int * offsets */
        if ((offsets) != (NULL) && (n) > (0)) {
            __cmd->offsets = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->offsets = NULL;
        }
        /* Input: char * pool */
        if ((pool) != (NULL) && (pool_size) > (0)) {
            __cmd->pool = HAS_OUT_BUFFER_SENTINEL;
        } else {
            __cmd->pool = NULL;
        }
        /* Input: size_t pool_size */
        __cmd->pool_size = pool_size;
    }

    struct hip___do_c_mass_symbol_info_call_record *__call_record =
        (struct hip___do_c_mass_symbol_info_call_record *)calloc(1,
        sizeof(struct hip___do_c_mass_symbol_info_call_record));

    __call_record->n = n;

    __call_record->offsets = offsets;

    __call_record->syms = syms;

    __call_record->pool_size = pool_size;

    __call_record->agents = agents;

    __call_record->types = types;

    __call_record->descriptors = descriptors;

    __call_record->pool = pool;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    ava_add_call(__call_id, __call_record);

    command_channel_send_command(__chan, (struct command_base *)__cmd);

    g_ptr_array_unref(__ava_alloc_list___do_c_mass_symbol_info);        /* Deallocate all memory in the alloc list */

    handle_commands_until(HIP_API, __call_record->__call_complete);
    hipError_t ret;
    ret = __call_record->ret;
    free(__call_record);
    command_channel_free_command(__chan, (struct command_base *)__cmd);
    return ret;
}

EXPORTED hipError_t
hipDeviceReset()
{
    abort_with_reason("Unsupported API function: hipDeviceReset");
}

EXPORTED hipError_t
hipSetDevice(int deviceId)
{
    abort_with_reason("Unsupported API function: hipSetDevice");
}

EXPORTED hipError_t
hipGetDevice(int *deviceId)
{
    abort_with_reason("Unsupported API function: hipGetDevice");
}

EXPORTED hipError_t
hipGetDeviceProperties(hipDeviceProp_t * prop, int deviceId)
{
    abort_with_reason("Unsupported API function: hipGetDeviceProperties");
}

EXPORTED hipError_t
hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig)
{
    abort_with_reason("Unsupported API function: hipDeviceSetCacheConfig");
}

EXPORTED hipError_t
hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig)
{
    abort_with_reason("Unsupported API function: hipDeviceGetCacheConfig");
}

EXPORTED hipError_t
hipDeviceGetLimit(size_t * pValue, enum hipLimit_t limit)
{
    abort_with_reason("Unsupported API function: hipDeviceGetLimit");
}

EXPORTED hipError_t
hipFuncSetCacheConfig(const void *func, hipFuncCache_t config)
{
    abort_with_reason("Unsupported API function: hipFuncSetCacheConfig");
}

EXPORTED hipError_t
hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig)
{
    abort_with_reason("Unsupported API function: hipDeviceGetSharedMemConfig");
}

EXPORTED hipError_t
hipDeviceSetSharedMemConfig(hipSharedMemConfig config)
{
    abort_with_reason("Unsupported API function: hipDeviceSetSharedMemConfig");
}

EXPORTED hipError_t
hipSetDeviceFlags(unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipSetDeviceFlags");
}

EXPORTED hipError_t
hipChooseDevice(int *device, const hipDeviceProp_t * prop)
{
    abort_with_reason("Unsupported API function: hipChooseDevice");
}

EXPORTED const char *
hipGetErrorName(hipError_t hip_error)
{
    abort_with_reason("Unsupported API function: hipGetErrorName");
}

EXPORTED const char *
hipGetErrorString(hipError_t hipError)
{
    abort_with_reason("Unsupported API function: hipGetErrorString");
}

EXPORTED hipError_t
hipStreamCreate(hipStream_t * stream)
{
    abort_with_reason("Unsupported API function: hipStreamCreate");
}

EXPORTED hipError_t
hipStreamCreateWithFlags(hipStream_t * stream, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipStreamCreateWithFlags");
}

EXPORTED hipError_t
hipStreamDestroy(hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipStreamDestroy");
}

EXPORTED hipError_t
hipStreamQuery(hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipStreamQuery");
}

EXPORTED hipError_t
hipStreamSynchronize(hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipStreamSynchronize");
}

EXPORTED hipError_t
hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
    abort_with_reason("Unsupported API function: hipStreamGetFlags");
}

EXPORTED hipError_t
hipStreamAddCallback(hipStream_t stream, void (*callback) (hipStream_t, hipError_t, void *), void *userData,
    unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipStreamAddCallback");
}

EXPORTED hipError_t
hipEventCreateWithFlags(hipEvent_t * event, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipEventCreateWithFlags");
}

EXPORTED hipError_t
hipEventQuery(hipEvent_t event)
{
    abort_with_reason("Unsupported API function: hipEventQuery");
}

EXPORTED hipError_t
hipPointerGetAttributes(hipPointerAttribute_t * attributes, const void *ptr)
{
    abort_with_reason("Unsupported API function: hipPointerGetAttributes");
}

EXPORTED hipError_t
hipMallocHost(void **ptr, size_t size)
{
    abort_with_reason("Unsupported API function: hipMallocHost");
}

EXPORTED hipError_t
hipHostMalloc(void **ptr, size_t size, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipHostMalloc");
}

EXPORTED hipError_t
hipHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipHostAlloc");
}

EXPORTED hipError_t
hipHostGetDevicePointer(void **devPtr, void *hstPtr, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipHostGetDevicePointer");
}

EXPORTED hipError_t
hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr)
{
    abort_with_reason("Unsupported API function: hipHostGetFlags");
}

EXPORTED hipError_t
hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipHostRegister");
}

EXPORTED hipError_t
hipHostUnregister(void *hostPtr)
{
    abort_with_reason("Unsupported API function: hipHostUnregister");
}

EXPORTED hipError_t
hipMallocPitch(void **ptr, size_t * pitch, size_t width, size_t height)
{
    abort_with_reason("Unsupported API function: hipMallocPitch");
}

EXPORTED hipError_t
hipFreeHost(void *ptr)
{
    abort_with_reason("Unsupported API function: hipFreeHost");
}

EXPORTED hipError_t
hipHostFree(void *ptr)
{
    abort_with_reason("Unsupported API function: hipHostFree");
}

EXPORTED hipError_t
hipMemcpyToSymbol(const void *symbolName, const void *src, size_t sizeBytes, size_t offset, hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpyToSymbol");
}

EXPORTED hipError_t
hipMemcpyToSymbolAsync(const void *symbolName, const void *src, size_t sizeBytes, size_t offset, hipMemcpyKind kind,
    hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemcpyToSymbolAsync");
}

EXPORTED hipError_t
hipMemcpyFromSymbol(void *dst, const void *symbolName, size_t sizeBytes, size_t offset, hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpyFromSymbol");
}

EXPORTED hipError_t
hipMemcpyFromSymbolAsync(void *dst, const void *symbolName, size_t sizeBytes, size_t offset, hipMemcpyKind kind,
    hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemcpyFromSymbolAsync");
}

EXPORTED hipError_t
hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes)
{
    abort_with_reason("Unsupported API function: hipMemsetD8");
}

EXPORTED hipError_t
hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemsetAsync");
}

EXPORTED hipError_t
hipMemset2D(void *dst, size_t pitch, int value, size_t width, size_t height)
{
    abort_with_reason("Unsupported API function: hipMemset2D");
}

EXPORTED hipError_t
hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemset2DAsync");
}

EXPORTED hipError_t
hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent)
{
    abort_with_reason("Unsupported API function: hipMemset3D");
}

EXPORTED hipError_t
hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemset3DAsync");
}

EXPORTED hipError_t
hipMemPtrGetInfo(void *ptr, size_t * size)
{
    abort_with_reason("Unsupported API function: hipMemPtrGetInfo");
}

EXPORTED hipError_t
hipMallocArray(hipArray ** array, const hipChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipMallocArray");
}

EXPORTED hipError_t
hipArrayCreate(hipArray ** pHandle, const HIP_ARRAY_DESCRIPTOR * pAllocateArray)
{
    abort_with_reason("Unsupported API function: hipArrayCreate");
}

EXPORTED hipError_t
hipArray3DCreate(hipArray ** array, const HIP_ARRAY_DESCRIPTOR * pAllocateArray)
{
    abort_with_reason("Unsupported API function: hipArray3DCreate");
}

EXPORTED hipError_t
hipMalloc3D(hipPitchedPtr * pitchedDevPtr, hipExtent extent)
{
    abort_with_reason("Unsupported API function: hipMalloc3D");
}

EXPORTED hipError_t
hipFreeArray(hipArray * array)
{
    abort_with_reason("Unsupported API function: hipFreeArray");
}

EXPORTED hipError_t
hipMalloc3DArray(hipArray ** array, const struct hipChannelFormatDesc *desc, struct hipExtent extent,
    unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipMalloc3DArray");
}

EXPORTED hipError_t
hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpy2D");
}

EXPORTED hipError_t
hipMemcpyParam2D(const hip_Memcpy2D * pCopy)
{
    abort_with_reason("Unsupported API function: hipMemcpyParam2D");
}

EXPORTED hipError_t
hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
    hipMemcpyKind kind, hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipMemcpy2DAsync");
}

EXPORTED hipError_t
hipMemcpy2DToArray(hipArray * dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width,
    size_t height, hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpy2DToArray");
}

EXPORTED hipError_t
hipMemcpyToArray(hipArray * dst, size_t wOffset, size_t hOffset, const void *src, size_t count, hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpyToArray");
}

EXPORTED hipError_t
hipMemcpyFromArray(void *dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count,
    hipMemcpyKind kind)
{
    abort_with_reason("Unsupported API function: hipMemcpyFromArray");
}

EXPORTED hipError_t
hipMemcpyAtoH(void *dst, hipArray * srcArray, size_t srcOffset, size_t count)
{
    abort_with_reason("Unsupported API function: hipMemcpyAtoH");
}

EXPORTED hipError_t
hipMemcpyHtoA(hipArray * dstArray, size_t dstOffset, const void *srcHost, size_t count)
{
    abort_with_reason("Unsupported API function: hipMemcpyHtoA");
}

EXPORTED hipError_t
hipMemcpy3D(const struct hipMemcpy3DParms *p)
{
    abort_with_reason("Unsupported API function: hipMemcpy3D");
}

EXPORTED hipError_t
hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId, int peerDeviceId)
{
    abort_with_reason("Unsupported API function: hipDeviceCanAccessPeer");
}

EXPORTED hipError_t
hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipDeviceEnablePeerAccess");
}

EXPORTED hipError_t
hipDeviceDisablePeerAccess(int peerDeviceId)
{
    abort_with_reason("Unsupported API function: hipDeviceDisablePeerAccess");
}

EXPORTED hipError_t
hipMemGetAddressRange(hipDeviceptr_t * pbase, size_t * psize, hipDeviceptr_t dptr)
{
    abort_with_reason("Unsupported API function: hipMemGetAddressRange");
}

EXPORTED hipError_t
hipMemcpyPeer(void *dst, int dstDeviceId, const void *src, int srcDeviceId, size_t sizeBytes)
{
    abort_with_reason("Unsupported API function: hipMemcpyPeer");
}

EXPORTED hipError_t
hipCtxCreate(hipCtx_t * ctx, unsigned int flags, hipDevice_t device)
{
    abort_with_reason("Unsupported API function: hipCtxCreate");
}

EXPORTED hipError_t
hipCtxDestroy(hipCtx_t ctx)
{
    abort_with_reason("Unsupported API function: hipCtxDestroy");
}

EXPORTED hipError_t
hipCtxPopCurrent(hipCtx_t * ctx)
{
    abort_with_reason("Unsupported API function: hipCtxPopCurrent");
}

EXPORTED hipError_t
hipCtxPushCurrent(hipCtx_t ctx)
{
    abort_with_reason("Unsupported API function: hipCtxPushCurrent");
}

EXPORTED hipError_t
hipCtxSetCurrent(hipCtx_t ctx)
{
    abort_with_reason("Unsupported API function: hipCtxSetCurrent");
}

EXPORTED hipError_t
hipCtxGetApiVersion(hipCtx_t ctx, int *apiVersion)
{
    abort_with_reason("Unsupported API function: hipCtxGetApiVersion");
}

EXPORTED hipError_t
hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig)
{
    abort_with_reason("Unsupported API function: hipCtxGetCacheConfig");
}

EXPORTED hipError_t
hipCtxSetCacheConfig(hipFuncCache_t cacheConfig)
{
    abort_with_reason("Unsupported API function: hipCtxSetCacheConfig");
}

EXPORTED hipError_t
hipCtxSetSharedMemConfig(hipSharedMemConfig config)
{
    abort_with_reason("Unsupported API function: hipCtxSetSharedMemConfig");
}

EXPORTED hipError_t
hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig)
{
    abort_with_reason("Unsupported API function: hipCtxGetSharedMemConfig");
}

EXPORTED hipError_t
hipCtxSynchronize()
{
    abort_with_reason("Unsupported API function: hipCtxSynchronize");
}

EXPORTED hipError_t
hipCtxGetFlags(unsigned int *flags)
{
    abort_with_reason("Unsupported API function: hipCtxGetFlags");
}

EXPORTED hipError_t
hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipCtxEnablePeerAccess");
}

EXPORTED hipError_t
hipCtxDisablePeerAccess(hipCtx_t peerCtx)
{
    abort_with_reason("Unsupported API function: hipCtxDisablePeerAccess");
}

EXPORTED hipError_t
hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int *flags, int *active)
{
    abort_with_reason("Unsupported API function: hipDevicePrimaryCtxGetState");
}

EXPORTED hipError_t
hipDevicePrimaryCtxRelease(hipDevice_t dev)
{
    abort_with_reason("Unsupported API function: hipDevicePrimaryCtxRelease");
}

EXPORTED hipError_t
hipDevicePrimaryCtxRetain(hipCtx_t * pctx, hipDevice_t dev)
{
    abort_with_reason("Unsupported API function: hipDevicePrimaryCtxRetain");
}

EXPORTED hipError_t
hipDevicePrimaryCtxReset(hipDevice_t dev)
{
    abort_with_reason("Unsupported API function: hipDevicePrimaryCtxReset");
}

EXPORTED hipError_t
hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipDevicePrimaryCtxSetFlags");
}

EXPORTED hipError_t
hipDeviceGet(hipDevice_t * device, int ordinal)
{
    abort_with_reason("Unsupported API function: hipDeviceGet");
}

EXPORTED hipError_t
hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device)
{
    abort_with_reason("Unsupported API function: hipDeviceComputeCapability");
}

EXPORTED hipError_t
hipDeviceGetName(char *name, int len, hipDevice_t device)
{
    abort_with_reason("Unsupported API function: hipDeviceGetName");
}

EXPORTED hipError_t
hipDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    abort_with_reason("Unsupported API function: hipDeviceGetPCIBusId");
}

EXPORTED hipError_t
hipDeviceGetByPCIBusId(int *device, const char *pciBusId)
{
    abort_with_reason("Unsupported API function: hipDeviceGetByPCIBusId");
}

EXPORTED hipError_t
hipDeviceTotalMem(size_t * bytes, hipDevice_t device)
{
    abort_with_reason("Unsupported API function: hipDeviceTotalMem");
}

EXPORTED hipError_t
hipDriverGetVersion(int *driverVersion)
{
    abort_with_reason("Unsupported API function: hipDriverGetVersion");
}

EXPORTED hipError_t
hipRuntimeGetVersion(int *runtimeVersion)
{
    abort_with_reason("Unsupported API function: hipRuntimeGetVersion");
}

EXPORTED hipError_t
hipFuncGetAttributes(hipFuncAttributes * attr, const void *func)
{
    abort_with_reason("Unsupported API function: hipFuncGetAttributes");
}

EXPORTED hipError_t
hipModuleGetGlobal(hipDeviceptr_t * dptr, size_t * bytes, hipModule_t hmod, const char *name)
{
    abort_with_reason("Unsupported API function: hipModuleGetGlobal");
}

EXPORTED hipError_t
hipModuleGetTexRef(textureReference ** texRef, hipModule_t hmod, const char *name)
{
    abort_with_reason("Unsupported API function: hipModuleGetTexRef");
}

EXPORTED hipError_t
hipModuleLoadDataEx(hipModule_t * module, const void *image, unsigned int numOptions, hipJitOption * options,
    void **optionValues)
{
    abort_with_reason("Unsupported API function: hipModuleLoadDataEx");
}

EXPORTED hipError_t
hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    hipStream_t stream, void **kernelParams, void **extra)
{
    abort_with_reason("Unsupported API function: hipModuleLaunchKernel");
}

EXPORTED hipError_t
hipProfilerStart()
{
    abort_with_reason("Unsupported API function: hipProfilerStart");
}

EXPORTED hipError_t
hipProfilerStop()
{
    abort_with_reason("Unsupported API function: hipProfilerStop");
}

EXPORTED hipError_t
hipIpcGetMemHandle(hipIpcMemHandle_t * handle, void *devPtr)
{
    abort_with_reason("Unsupported API function: hipIpcGetMemHandle");
}

EXPORTED hipError_t
hipIpcOpenMemHandle(void **devPtr, hipIpcMemHandle_t handle, unsigned int flags)
{
    abort_with_reason("Unsupported API function: hipIpcOpenMemHandle");
}

EXPORTED hipError_t
hipIpcCloseMemHandle(void *devPtr)
{
    abort_with_reason("Unsupported API function: hipIpcCloseMemHandle");
}

EXPORTED hipError_t
hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream)
{
    abort_with_reason("Unsupported API function: hipConfigureCall");
}

EXPORTED hipError_t
hipSetupArgument(const void *arg, size_t size, size_t offset)
{
    abort_with_reason("Unsupported API function: hipSetupArgument");
}

EXPORTED hipError_t
hipLaunchByPtr(const void *func)
{
    abort_with_reason("Unsupported API function: hipLaunchByPtr");
}

EXPORTED hsa_status_t
nw_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute,
    void *value)
{
    abort_with_reason("Unsupported API function: nw_hsa_executable_symbol_get_info");
}
