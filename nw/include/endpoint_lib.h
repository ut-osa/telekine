#ifndef __VGPU_ENDPOINT_LIB_H__
#define __VGPU_ENDPOINT_LIB_H__

#include <stdio.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include <glib.h>
#include <glib/ghash.h>
#include <gmodule.h>

#include "common/murmur3.h"
#include "common/cmd_channel.h"
#include "common/cmd_handler.h"

//// Utilities

#define __ava_check_type(type, expr) ({ type __tmp = (expr); __tmp; })

struct nw_handle_pool;

struct nw_handle_pool* nw_handle_pool_new();
void* nw_handle_pool_insert(struct nw_handle_pool *pool,
                            const void* handle);
void* nw_handle_pool_lookup_or_insert(struct nw_handle_pool *pool,
                                      const void* handle);
void* nw_handle_pool_deref(struct nw_handle_pool *pool,
                           const void* id);
void* nw_handle_pool_deref_and_remove(struct nw_handle_pool *pool,
                                      const void* id);

gboolean nw_hash_table_remove_flipped(gconstpointer key, GHashTable *hash_table);

#define abort_with_reason(reason) assert(reason && 0)

/* Sentinel to tell worker there is a buffer to return data into. */
#define HAS_OUT_BUFFER_SENTINEL ((void*)1)

#define __STRINGIFY_DETAIL(x) #x
#define __STRINGIFY(x) __STRINGIFY_DETAIL(x)

/// Extract the explicit state of the object `o` and write it to `state_stream_handle` using `ava_write_state`.
/// `extract` must return `TRUE` on success and `FALSE` on failure, and should leave `errno` unchanged if there is an
/// error.
typedef int (*ava_extract_function)(void*, void*);

/// Replace (reconstruct) the explicit state of the object `o` based on the state stored in `state_stream_handle` and
/// read using `ava_read_state`. `replace` must return `TRUE` on success and `FALSE` on failure, and should leave
/// `errno` unchanged if there is an error.
typedef int (*ava_replace_function)(void*, void*);


//// Library functions/macros expected by metadata expressions

#include <string.h>

#define max(a,b)   ({                  \
            __typeof__ (a) _a = (a);   \
            __typeof__ (b) _b = (b);   \
            _a > _b ? _a : _b;         \
        })

#define min(a,b)   ({                  \
            __typeof__ (a) _a = (a);   \
            __typeof__ (b) _b = (b);   \
            _a < _b ? _a : _b;         \
        })

enum ava_sync_mode_t {
    NW_ASYNC = 0,
    NW_SYNC,
    NW_FLUSH
};

enum ava_transfer_t {
    NW_NONE = 0,
    NW_HANDLE,
    NW_OPAQUE,
    NW_BUFFER,
    NW_CALLBACK,
    NW_FILE
};

enum ava_lifetime_t {
    AVA_NONE = 0,
    AVA_COUPLED,
    AVA_STATIC,
    AVA_MANUAL
};


struct call_id_and_handle_t {
    int call_id;
    const void *handle;
};

guint nw_hash_mix64variant13(gconstpointer ptr);

static inline guint nw_hash_struct(gconstpointer ptr, size_t size) {
    guint ret;
    MurmurHash3_x86_32(ptr, size, 0xfbcdabc7 + size, &ret);
    return ret;
}

static guint nw_hash_call_id_and_handle(gconstpointer ptr) {
    return nw_hash_struct(ptr, sizeof(struct call_id_and_handle_t));
}

static gint nw_equal_call_id_and_handle(gconstpointer ptr1, gconstpointer ptr2) {
    return memcmp(ptr1, ptr2, sizeof(struct call_id_and_handle_t)) == 0;
}

static guint nw_hash_pointer(gconstpointer ptr) {
    return nw_hash_mix64variant13(ptr);
}

/// Create a new metadata map.
GHashTable * metadata_map_new();

static gpointer nw_hash_table_steal_value(GHashTable *hash_table,
                                          gconstpointer key)
{
    gpointer value = g_hash_table_lookup(hash_table, key);
    gboolean b = g_hash_table_steal(hash_table, key);
    /* In GLIB 2.58 we could use the following:
       g_hash_table_steal_extended(hash_table, key, NULL, &value); */
    if (!b) {
        return NULL;
    } else {
        return value;
    }
}

#endif
