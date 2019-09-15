#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>

#include <sys/mman.h>

#include "worker.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"
#include "common/cmd_handler.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/socket.h"

struct command_channel *chans[N_AVA_CHANNELS];

void sig_handler(int signo)
{
    int i;
    for (i = 0; i < N_AVA_CHANNELS; i++)
       command_channel_free(chans[i]);
    abort();
}

void sigsegv_handler(int signo, siginfo_t *info, void *data)
{
    DEBUG_PRINT("segfault in thread pid = %lx\n", (long)pthread_self());
    abort();
}

void nw_report_storage_resource_allocation(const char* const name, ssize_t amount)
{
    int i;
    for (i = 0; i < N_AVA_CHANNELS; i++)
       command_channel_report_storage_resource_allocation(chans[i], name, amount);
}

void nw_report_throughput_resource_consumption(const char* const name, ssize_t amount)
{
    int i;
    for (i = 0; i < N_AVA_CHANNELS; i++)
       command_channel_report_throughput_resource_consumption(chans[i], name, amount);
}

static struct command_channel* channel_create(int chan_no)
{
   return chans[chan_no];
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

    if (argc != 3) {
        printf("Usage: %s <remote> <port>\n", argv[0]);
        return 0;
    }

    const char* remote = argv[1];
    int port = atoi(argv[2]);

    int i;
    for (i = 0; i < N_AVA_CHANNELS; i++) {
        chans[i] = command_channel_min_worker_new_reverse_socket(
                remote, port + (i * CHANNEL_OFFSET));
    }

    
    for (i = 0; i < N_AVA_CHANNELS; i++)
       init_command_handler(channel_create, i);
    DEBUG_PRINT("[worker#%s] start polling tasks\n", argv[3]);
    wait_for_command_handler();
    for (i = 0; i < N_AVA_CHANNELS; i++)
       command_channel_free(chans[i]);

    return 0;
}
