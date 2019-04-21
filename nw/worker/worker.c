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

struct command_channel *chan0;
struct command_channel *chan1;
struct command_channel *chan2;

void sig_handler(int signo)
{
    command_channel_free(chan0);
    command_channel_free(chan1);
    command_channel_free(chan2);
    exit(0);
}

void sigsegv_handler(int signo, siginfo_t *info, void *data)
{
    DEBUG_PRINT("segfault in thread pid = %lx\n", (long)pthread_self());
    exit(0);
}

void nw_report_storage_resource_allocation(const char* const name, ssize_t amount)
{
    command_channel_report_storage_resource_allocation(chan0, name, amount);
    command_channel_report_storage_resource_allocation(chan1, name, amount);
    command_channel_report_storage_resource_allocation(chan2, name, amount);
}

void nw_report_throughput_resource_consumption(const char* const name, ssize_t amount)
{
    command_channel_report_throughput_resource_consumption(chan0, name, amount);
    command_channel_report_throughput_resource_consumption(chan1, name, amount);
    command_channel_report_throughput_resource_consumption(chan2, name, amount);
}

static struct command_channel* channel_create(int chan_no)
{
   if (chan_no == 0)
    return chan0;
   else if (chan_no == 1)
    return chan1;
   else
    return chan2;
}

void notify_manager()
{
        int manager_port = 4000;
        const char* manager_port_str = getenv("AVA_MANAGER_PORT");
        if (manager_port_str != NULL) manager_port = atoi(manager_port_str);
        char ava_fifo[32];
        sprintf(ava_fifo, "/tmp/ava_fifo_%d", manager_port);
        uint64_t rdy = 1;
        int fifo_fd = open(ava_fifo, O_WRONLY|O_CLOEXEC);
        if (fifo_fd < 0) {
           perror(ava_fifo);
           abort();
        }
        if (write(fifo_fd, &rdy, sizeof(rdy)) != sizeof(rdy)) {
           perror("write fifo");
           abort();
        }
        close(fifo_fd);
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
        int listen_fds[3];
        int port = atoi(argv[3]);
        set_up_ports(port, listen_fds);

        notify_manager();
        chan0 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), listen_fds[0], port, atoi(argv[5]));
        chan1 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), listen_fds[1], port + CHANNEL_OFFSET, atoi(argv[5]));
        chan2 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), listen_fds[2], port + CHANNEL_OFFSET * 2, atoi(argv[5]));
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "SHM")) {
        chan0 = command_channel_shm_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        chan1 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) + CHANNEL_OFFSET, atoi(argv[4]), atoi(argv[5]));
        chan2 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) + CHANNEL_OFFSET * 2, atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
        chan0 = command_channel_socket_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        chan1 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) + CHANNEL_OFFSET, atoi(argv[4]), atoi(argv[5]));
        chan2 = command_channel_min_worker_new(
                atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) + CHANNEL_OFFSET * 2, atoi(argv[4]), atoi(argv[5]));
    }
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK]\n");
        return 0;
    }

    init_command_handler(channel_create, 0);
    init_command_handler(channel_create, 1);
    init_command_handler(channel_create, 2);
    DEBUG_PRINT("[worker#%s] start polling tasks\n", argv[3]);
    wait_for_command_handler();
    command_channel_free(chan0);
    command_channel_free(chan1);
    command_channel_free(chan2);

    return 0;
}
