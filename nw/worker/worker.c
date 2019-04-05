#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>

#include "worker.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"
#include "common/cmd_handler.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/socket.h"

struct command_channel *chan;

void sig_handler(int signo)
{
    command_channel_free(chan);
    exit(0);
}

void sigsegv_handler(int signo, siginfo_t *info, void *data)
{
    DEBUG_PRINT("segfault in thread pid = %lx\n", (long)pthread_self());
    exit(0);
}

void nw_report_storage_resource_allocation(const char* const name, ssize_t amount)
{
    command_channel_report_storage_resource_allocation(chan, name, amount);
}

void nw_report_throughput_resource_consumption(const char* const name, ssize_t amount)
{
    command_channel_report_throughput_resource_consumption(chan, name, amount);
}

static struct command_channel* channel_create()
{
    return chan;
}

int main(int argc, char *argv[])
{
    int err;

    if (argc != 6) {
        printf("Usage: %s <vm_id> <api_id> <listen_port> <pb_offset> <pb_size>\n", argv[0]);
        return 0;
    }

#if 0
    /* setup signal handler */
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    /* setup SIGSEGV handler */
    struct sigaction sigsegv_act;
    sigemptyset(&sigsegv_act.sa_mask);
    sigsegv_act.sa_sigaction = &sigsegv_handler;
    sigsegv_act.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sigsegv_act, NULL);
#endif

    if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "LOCAL")) {
        chan = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "SHM")) {
        chan = command_channel_shm_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
        chan = command_channel_socket_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK]\n");
        return 0;
    }

    init_command_handler(channel_create);
    DEBUG_PRINT("[worker#%s] start polling tasks\n", argv[3]);
    wait_for_command_handler();
    command_channel_free(chan);

    return 0;
}
