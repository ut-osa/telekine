#ifndef __VGPU_CMD_CHANNEL_IMPL_H__
#define __VGPU_CMD_CHANNEL_IMPL_H__

#include "cmd_channel.h"

struct command_channel_vtable {
    size_t (*command_channel_buffer_size)(struct command_channel* chan, size_t size);
    struct command_base* (*command_channel_new_command)(struct command_channel* chan, size_t command_struct_size, size_t data_region_size);
    void* (*command_channel_attach_buffer)(struct command_channel* chan, struct command_base* cmd, void* buffer, size_t size);
    void (*command_channel_send_command)(struct command_channel* chan, struct command_base* cmd);
    struct command_base* (*command_channel_receive_command)(struct command_channel* chan);
    void* (*command_channel_get_buffer)(struct command_channel* chan, struct command_base* cmd, void* buffer_id);
    void* (*command_channel_get_data_region)(struct command_channel* c, struct command_base* cmd);
    void (*command_channel_free_command)(struct command_channel* chan, struct command_base* cmd);
    void (*command_channel_free)(struct command_channel* chan);
    void (*command_channel_print_command)(struct command_channel* chan, struct command_base* cmd);
    void (*command_channel_report_storage_resource_allocation)(struct command_channel*c, const char* const name, ssize_t amount);
    void (*command_channel_report_throughput_resource_consumption)(struct command_channel* c, const char* const name, ssize_t amount);
};

/**
 * The "base" structure for command channels, this must be the first field of every command channel.
 */
struct command_channel_base {
    struct command_channel_vtable* vtable;
};

static inline void command_channel_preinitialize(struct command_channel* chan, struct command_channel_vtable* vtable) {
    ((struct command_channel_base*)chan)->vtable = vtable;
}

#endif // ndef __VGPU_CMD_CHANNEL_IMPL_H__
