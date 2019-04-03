#include "common/linkage.h"
#include "common/debug.h"
#include "common/cmd_handler.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>

// Internal flag set by the first call to init_command_handler
EXPORTED_WEAKLY volatile int init_command_handler_executed;

EXPORTED_WEAKLY struct command_channel* nw_global_command_channel;
EXPORTED_WEAKLY pthread_mutex_t nw_handler_lock = PTHREAD_MUTEX_INITIALIZER;

EXPORTED_WEAKLY void (*nw_api_handlers[MAX_API_ID])(struct command_base* cmd) = {0, };
EXPORTED_WEAKLY pthread_cond_t nw_api_conds[MAX_API_ID] = {PTHREAD_COND_INITIALIZER, };
EXPORTED_WEAKLY pthread_t nw_handler_thread;

static void _handle_commands_until_api_loop(struct command_channel* chan, int until_api_id);

EXPORTED_WEAKLY void register_command_handler(int api_id, void (*handle_command)(struct command_base* cmd)) {
    assert(api_id > 0);
    assert(api_id < MAX_API_ID);
    DEBUG_PRINT("Registering API command handler for API id %d: function at 0x%lx\n", api_id, (uintptr_t)handle_command);
    assert(nw_api_handlers[api_id] == NULL && "Only one handler can be registered for each API id");
    nw_api_handlers[api_id] = handle_command;
}

EXPORTED_WEAKLY void _internal_handle_commands_until_api(int api_id) {
    if (pthread_equal(pthread_self(), nw_handler_thread)) {
        // If we are executing on the handler thread already then recursively call the handler loop.
        // Otherwise we would deadlock.
        pthread_mutex_unlock(&nw_handler_lock);
        _handle_commands_until_api_loop(nw_global_command_channel, api_id);
        pthread_mutex_lock(&nw_handler_lock);
    } else {
        pthread_cond_wait(&nw_api_conds[api_id], &nw_handler_lock);
    }
}

EXPORTED_WEAKLY void handle_commands_until_api(int api_id) {
    // TODO:PERFORMANCE: This implementation is probably way slower than it needs to be.
    pthread_mutex_lock(&nw_handler_lock);
    _internal_handle_commands_until_api(api_id);
    pthread_mutex_unlock(&nw_handler_lock);
}


static void _handle_commands_until_api_loop(struct command_channel* chan, int until_api_id) {
    while(1) {
        struct command_base* cmd = command_channel_receive_command(chan);
        const intptr_t api_id = cmd->api_id;
        pthread_mutex_lock(&nw_handler_lock);
        // TODO:PERFORMANCE: The locking prevents two commands from running at the same time. This is a big problem. Could it deadlock?
        // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.

        // TODO: handle internal APIs (api_id = 0).
        if (api_id > 0) {
            nw_api_handlers[api_id](cmd);
            pthread_cond_broadcast(&nw_api_conds[api_id]);
        }
        pthread_mutex_unlock(&nw_handler_lock);
        if (until_api_id != -1 && api_id == until_api_id) {
            return;
        }
    };
}

static void* handle_commands(void* userdata) {
    struct command_channel* chan = (struct command_channel*)userdata;

    // set cancellation state
    if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL)) {
        perror("pthread_setcancelstate failed\n");
        exit(0);
    }

    // PTHREAD_CANCEL_DEFERRED means that it will wait the pthread_join
    if (pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL)) {
        perror("pthread_setcanceltype failed\n");
        exit(0);
    }

    _handle_commands_until_api_loop(chan, -1);
    return NULL;
}
// TODO: This will not correctly handle running callbacks in the initially calling thread.

EXPORTED_WEAKLY void init_command_handler(struct command_channel* (*channel_create)()) {
    pthread_mutex_lock(&nw_handler_lock);
    if (!init_command_handler_executed) {
        nw_global_command_channel = channel_create();
        pthread_create(&nw_handler_thread, NULL,
                       handle_commands, (void*)nw_global_command_channel);
        atomic_thread_fence(memory_order_release);
        init_command_handler_executed = 1;
    }
    pthread_mutex_unlock(&nw_handler_lock);
}

EXPORTED_WEAKLY void destroy_command_handler() {
    pthread_mutex_lock(&nw_handler_lock);
    if (init_command_handler_executed) {
        pthread_cancel(nw_handler_thread);
        pthread_join(nw_handler_thread, NULL);
        command_channel_free(nw_global_command_channel);
        atomic_thread_fence(memory_order_release);
        init_command_handler_executed = 0;
    }
    pthread_mutex_unlock(&nw_handler_lock);
}

EXPORTED_WEAKLY void wait_for_command_handler() {
    pthread_join(nw_handler_thread, NULL);
}
