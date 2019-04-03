#include <stdio.h>

#include "common/devconf.h"
#include "maptable.h"

/**
 * Encrypt objects
 *
 * Address mapping is used to hide the real hardware information, e.g. the
 * GPU memory addresses or object handles.
 * In this demo, we implemented Feistel cipher to encrypt/decrypt 64-bit
 * addresses. Feistel cipher is widely used in cryptography.
 *
 *   1. Create a hash of the lower 32 bits and XOR it into the upper 32 bits
 *   2. Swap the lower and upper 32 bits
 *   3. Repeat a few times.
 *
 * Encrypt/decrypt functions can be replaced with any reversible hash
 * functions, or a self-implemented hash map.
 */
uintptr_t feistel_cipher(uintptr_t text)
{
    uintptr_t hash;
    int i;

    for (i = 0; i < HASH_TIMES; ++i) {
        hash = ((text << 32) & 0xFFFFFFFF00000000) ^ text;
        text = ((hash << 32) & 0xFFFFFFFF00000000) | ((hash >> 32) & 0xFFFFFFFF);
    }

    return text;
}

/**
 * Self-implemented hash tables
 *
 * Each VM has a separate hash table with address starting from 0x1.
 */
struct vm_state vm_st[MAX_VM_NUM];
void addr_map_init(size_t vm_id)
{
    vm_st[vm_id].map_hash = g_hash_table_new(g_direct_hash, g_direct_equal);
    vm_st[vm_id].map_addr = 0x8;
}

uintptr_t addr_map(size_t vm_id, uintptr_t old)
{
    GHashTable *hash = vm_st[vm_id].map_hash;
    uintptr_t new = vm_st[vm_id].map_addr;

    vm_st[vm_id].map_addr += 0x8;
    g_hash_table_insert(hash, GINT_TO_POINTER(new), GINT_TO_POINTER(old));

    return new;
}

uintptr_t addr_demap(size_t vm_id, uintptr_t new)
{
    GHashTable *hash = vm_st[vm_id].map_hash;

    return (uintptr_t)g_hash_table_lookup(hash, GINT_TO_POINTER(new));
}

void addr_unmap(size_t vm_id, uintptr_t new)
{
    GHashTable *hash = vm_st[vm_id].map_hash;
    gboolean found = g_hash_table_remove(hash, GINT_TO_POINTER(new));

    if (!found)
        fprintf(stderr, "failed to find hash for 0x%lx\n", new);
}
