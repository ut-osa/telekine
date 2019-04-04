#ifndef __EXECUTOR_WORKER_H__
#define __EXECUTOR_WORKER_H__

#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

typedef struct MemoryRegion {
    void *addr;
    size_t size;
} MemoryRegion;

struct command_channel* command_channel_shm_worker_new(int vm_id, int rt_type, int listen_port,
        uintptr_t param_block_local_offset, size_t param_block_size);
struct command_channel* command_channel_min_worker_new(int vm_id, int rt_type, int listen_port,
        uintptr_t param_block_local_offset, size_t param_block_size, int fd);
struct command_channel* command_channel_socket_worker_new(int vm_id, int rt_type, int listen_port,
        uintptr_t param_block_local_offset, size_t param_block_size);

void nw_report_storage_resource_allocation(const char* const name, ssize_t amount);
void nw_report_throughput_resource_consumption(const char* const name, ssize_t amount);

#endif
