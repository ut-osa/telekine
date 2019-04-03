#ifndef __VGPU_GUEST_MEM_H__
#define __VGPU_GUEST_MEM_H__

#ifdef __KERNEL__

#include <linux/list.h>

#else

#include <pthread.h>
#include <stdint.h>
#include <semaphore.h>
#include "import/list.h"

#endif

/* parameter data block */
struct param_block
{
    struct list_head list;

    void *base;
    size_t size;
    uintptr_t offset;     /* offset to vgpu_dev.dstore->base_addr */

    // management
    uintptr_t cur_offset; /* start from 0 */
};

struct block_seeker
{
    uintptr_t global_offset; /* local_offset + param_block.offset */
    uintptr_t local_offset;  /* start from 0 */
    uintptr_t cur_offset;    /* start from local_offset + 0x4 */
};

struct param_block_info {
    uintptr_t param_local_offset;
    uintptr_t param_block_size;
};

#endif
