#ifndef __VGPU_CMD_HANDLER_H__
#define __VGPU_CMD_HANDLER_H__

#include <pthread.h>

#include "common/cmd_channel.h"

/**
 * Register a function to handle commands with the specified API id.
 *
 * This call will abort the program if init_command_handler has
 * already been called.
 */
void register_command_handler(int api_id, void (*handle_command)(struct command_base* cmd, int chan_no));

/**
 * Block until a command with the specified API is executed.
 */
void handle_commands_until_api(int api_id, int chan_no);

void _internal_handle_commands_until_api(int api_id, int chan_no);
extern pthread_mutex_t nw_handler_lock[3];

/**
 * Block until a command with the specified API is executed and the
 * predicate is true.
 */
#define handle_commands_until(api_id, predicate)               \
    pthread_mutex_lock(&nw_handler_lock[chan_no]);                      \
    while(!(predicate)) _internal_handle_commands_until_api(api_id, chan_no);\
    pthread_mutex_unlock(&nw_handler_lock[chan_no]);

/**
 * Initialize and start the command handler thread.
 *
 * This call is always very slow.
 */
void init_command_handler(struct command_channel* (*channel_create)(), int chan_no);

/**
 * Terminate the handler and close the channel and release other
 * resources.
 */
void destroy_command_handler();

/**
 * Block until the command handler thread exits. This may never
 * happen.
 */
void wait_for_command_handler();

/**
 * The global channel used by this process (either the guestlib or the
 * worker).
 */
extern struct command_channel* nw_global_command_channel[3];

#define MAX_API_ID 256

///// Commands

/**
 * We use an internal API to communicate between components. This is
 * its ID.
 */
#define COMMAND_HANDLER_API 0

/**
 * Commands in the internal API.
 */
enum command_handler_command_id {
    COMMAND_HANDLER_INITIALIZE_API,
};

struct command_handler_initialize_api_command {
    struct command_base base;
    intptr_t new_api_id;
};

#endif // ndef __VGPU_CMD_HANDLER_H__
