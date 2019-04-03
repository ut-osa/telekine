#include "common/endpoint_lib.h"

#include <glib.h>
#include <pthread.h>
#include <stdint.h>

struct nw_handle_pool {
    GHashTable *to_handle;
    GHashTable *to_id;
    pthread_mutex_t lock;
};

static atomic_uintptr_t id_counter = 1024;

static void* next_id() {
    uintptr_t n;
    while((n = atomic_fetch_add(&id_counter, 1)) < 1024)
        ;
    return (void*)n;
}

struct nw_handle_pool* nw_handle_pool_new() {
    struct nw_handle_pool* ret = (struct nw_handle_pool*)malloc(sizeof(struct nw_handle_pool));
    ret->to_handle = metadata_map_new();
    ret->to_id = metadata_map_new();
    pthread_mutex_init(&ret->lock, NULL);
    return ret;
}

static void* internal_handle_pool_insert(struct nw_handle_pool *pool,
                                  const void* handle) {
    void* id = next_id();
    gboolean b = g_hash_table_insert(pool->to_id, handle, id);
    assert(b && "handle already exists in nw_handle_pool_insert");
    b = g_hash_table_insert(pool->to_handle, id, handle);
    assert(b && "id already exists in nw_handle_pool_insert");
    return id;
}

void* nw_handle_pool_insert(struct nw_handle_pool *pool,
                            const void* handle) {
    if (handle == NULL || pool == NULL)
        return (void*)handle;
    pthread_mutex_lock(&pool->lock);
    void* id = internal_handle_pool_insert(pool, handle);
    pthread_mutex_unlock(&pool->lock);
    return id;
}

void* nw_handle_pool_lookup_or_insert(struct nw_handle_pool *pool,
                                      const void* handle) {
    if (handle == NULL || pool == NULL)
        return (void*)handle;
    pthread_mutex_lock(&pool->lock);
    void* id = g_hash_table_lookup(pool->to_id, handle);
    if (id == NULL)
        id = internal_handle_pool_insert(pool, handle);
    pthread_mutex_unlock(&pool->lock);
    return id;
}

void* nw_handle_pool_deref(struct nw_handle_pool *pool,
                           const void* id) {
    if (id == NULL || pool == NULL)
        return (void*)id;
    pthread_mutex_lock(&pool->lock);
    void* handle = g_hash_table_lookup(pool->to_handle, id);
    assert(handle != NULL);
    pthread_mutex_unlock(&pool->lock);
    return handle;
}

void* nw_handle_pool_deref_and_remove(struct nw_handle_pool *pool,
                                      const void* id) {
    if (id == NULL || pool == NULL)
        return (void*)id;
    pthread_mutex_lock(&pool->lock);
    void* handle = nw_hash_table_steal_value(pool->to_handle, id);
    assert(handle != NULL);
    g_hash_table_remove(pool->to_id, handle);
    pthread_mutex_unlock(&pool->lock);
    return handle;
}

gboolean nw_hash_table_remove_flipped(gconstpointer key, GHashTable *hash_table) {
    return g_hash_table_remove(hash_table, key);
}


/// 64-bit to 64-bit hash from http://dx.doi.org/10.1145/2714064.2660195.
/// Modified to produce a 32-bit result.
/// This hash was chosen based on the paper above and https://nullprogram.com/blog/2018/07/31/.
guint nw_hash_mix64variant13(gconstpointer ptr) {
    uintptr_t x = (uintptr_t)ptr;
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return (guint)x;
}

GHashTable * metadata_map_new() {
    return g_hash_table_new(nw_hash_mix64variant13, g_direct_equal);
}
