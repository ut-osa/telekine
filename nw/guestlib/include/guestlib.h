#ifndef __VGPU_GUESTLIB_H__
#define __VGPU_GUESTLIB_H__

#include <stdint.h>

void nw_init_guestlib(intptr_t api_id);
void nw_destroy_guestlib(void);

struct command_channel* command_channel_shm_new();
struct command_channel* command_channel_min_new();
struct command_channel* command_channel_socket_new();

#endif
