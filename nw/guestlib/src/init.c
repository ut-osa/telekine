#include "guestlib.h"
#include "common/linkage.h"
#include "common/cmd_handler.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"
#include <string.h>
#include <unistd.h>

EXPORTED_WEAKLY void nw_init_guestlib(intptr_t api_id)
{
    /* Create connection to worker and start command handler thread */
    if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "LOCAL")) {
        init_command_handler(command_channel_min_new, 0);
        init_command_handler(command_channel_min_new, 1);
        init_command_handler(command_channel_min_new, 2);
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "SHM")) {
        init_command_handler(command_channel_shm_new, 0);
        init_command_handler(command_channel_shm_new, 1);
        init_command_handler(command_channel_shm_new, 1);
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
        init_command_handler(command_channel_socket_new, 0);
        init_command_handler(command_channel_socket_new, 1);
        init_command_handler(command_channel_socket_new, 2);
    }
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK]\n");
        return;
    }

    /* Send initialize API command to the worker */
    struct command_handler_initialize_api_command* api_init_command0 =
        (struct command_handler_initialize_api_command*)command_channel_new_command(
            nw_global_command_channel[0], sizeof(struct command_handler_initialize_api_command), 0);
    api_init_command0->base.api_id = COMMAND_HANDLER_API;
    api_init_command0->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
    api_init_command0->new_api_id = api_id;
    command_channel_send_command(nw_global_command_channel[0], (struct command_base*)api_init_command0);

    struct command_handler_initialize_api_command* api_init_command1 =
        (struct command_handler_initialize_api_command*)command_channel_new_command(
            nw_global_command_channel[1], sizeof(struct command_handler_initialize_api_command), 0);
    api_init_command1->base.api_id = COMMAND_HANDLER_API;
    api_init_command1->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
    api_init_command1->new_api_id = api_id;
    command_channel_send_command(nw_global_command_channel[1], (struct command_base*)api_init_command1);

    struct command_handler_initialize_api_command* api_init_command2 =
        (struct command_handler_initialize_api_command*)command_channel_new_command(
            nw_global_command_channel[2], sizeof(struct command_handler_initialize_api_command), 0);
    api_init_command2->base.api_id = COMMAND_HANDLER_API;
    api_init_command2->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
    api_init_command2->new_api_id = api_id;
    command_channel_send_command(nw_global_command_channel[2], (struct command_base*)api_init_command2);
}

EXPORTED_WEAKLY void nw_destroy_guestlib(void)
{
    // TODO: This is called by the guestlib so destructor for each API. This is safe, but will make the handler shutdown when the FIRST API unloads when having it shutdown with the last would be better.
    destroy_command_handler(0);
    destroy_command_handler(1);
    destroy_command_handler(3);
}
