#include "common/cmd_channel_impl.h"
#include "common/linkage.h"

static __thread int cur_ava_channel;

EXPORTED void set_ava_chan_no(int ava_channel)
{
   cur_ava_channel = ava_channel;
}

EXPORTED int get_ava_chan_no(void)
{
   return cur_ava_channel;
}

void command_channel_free(struct command_channel* chan)  {
  ((struct command_channel_base*)chan)->vtable->command_channel_free(chan);
}

size_t command_channel_buffer_size(struct command_channel* chan, size_t size) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_buffer_size(chan, size);
}

struct command_base* command_channel_new_command(struct command_channel* chan, size_t command_struct_size, size_t data_region_size) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_new_command(chan, command_struct_size, data_region_size);
}

void* command_channel_attach_buffer(struct command_channel* chan, struct command_base* cmd, const void* buffer, size_t size) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_attach_buffer(chan, cmd, buffer, size);
}

void command_channel_send_command(struct command_channel* chan, struct command_base* cmd) {
  ((struct command_channel_base*)chan)->vtable->command_channel_send_command(chan, cmd);
}

struct command_base* command_channel_receive_command(struct command_channel* chan) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_receive_command(chan);
}

void* command_channel_get_buffer(struct command_channel* chan, struct command_base* cmd, const void* buffer_id) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_get_buffer(chan, cmd, buffer_id);
}

void* command_channel_get_data_region(struct command_channel* chan, struct command_base* cmd) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_get_data_region(chan, cmd);
}

void command_channel_free_command(struct command_channel* chan, struct command_base* cmd) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_free_command(chan, cmd);
}

void command_channel_print_command(struct command_channel* chan, struct command_base* cmd) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_print_command(chan, cmd);
}

void command_channel_report_storage_resource_allocation(struct command_channel*chan, const char* const name, ssize_t amount) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_report_storage_resource_allocation(chan, name, amount);
}

void command_channel_report_throughput_resource_consumption(struct command_channel* chan, const char* const name, ssize_t amount) {
  return ((struct command_channel_base*)chan)->vtable->command_channel_report_throughput_resource_consumption(chan, name, amount);
}
