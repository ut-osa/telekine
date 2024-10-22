#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/mman.h>

#include "common/cmd_channel_impl.h"
#include "common/guest_mem.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/socket.h"

int listen_fd;
int worker_id;

void sig_handler(int signo)
{
    if (listen_fd > 0)
        close(listen_fd);
    abort();
}

int main(int argc, char *argv[])
{
    /* setup signal handler */
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    /* initialize vsock */
    struct sockaddr_vm sa_listen;
    int client_fd;
    pid_t child;
    struct command_base msg, response;
    struct param_block_info *pb_info;
    uintptr_t *worker_port;

    worker_id = 1;
    listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, WORKER_MANAGER_PORT);
    listen_vm_socket(listen_fd, &sa_listen);

    /* polling new applications */
    do {
        client_fd = accept_vm_socket(listen_fd);

        /* get guestlib info */
        recv_socket(client_fd, &msg, sizeof(struct command_base));
        if (msg.command_type != MSG_NEW_APPLICATION) {
            printf("[manager] wrong message type\n");
            close(client_fd);
            continue;
        }

        /* return worker port to guestlib */
        response.api_id = INTERNAL_API;
        worker_port = (uintptr_t *)response.reserved_area;
        *worker_port = worker_id + WORKER_PORT_BASE;
        send_socket(client_fd, &response, sizeof(struct command_base));
        close(client_fd);

        /* spawn a worker */
        child = fork();
        if (child == 0) {
            close(listen_fd);
            break;
        }

        worker_id++;
    } while (1);

    /* spawn worker */
    char str_vm_id[10];
    char str_rt_type[10];
    char str_port[10];
    char str_pb_offset[10];
    char str_pb_size[10];
    pb_info = (struct param_block_info *)msg.reserved_area;
    sprintf(str_vm_id, "%d", msg.vm_id);
    sprintf(str_rt_type, "%d", msg.api_id);
    sprintf(str_port, "%d", worker_id + WORKER_PORT_BASE);
    sprintf(str_pb_offset, "%lu", pb_info->param_local_offset);
    sprintf(str_pb_size, "%lu", pb_info->param_block_size);
    char *argv_list[] = {"worker.out",
                         str_vm_id, str_rt_type, str_port,
                         str_pb_offset, str_pb_size, NULL};
    printf("[manager] worker vm_id=%d, rt_type=%d, port=%s, pb_offset=%lx, pb_size=%lx\n",
           msg.vm_id, msg.api_id, str_port,
           pb_info->param_local_offset,
           pb_info->param_block_size);
    if (execv("./worker", argv_list)) {
        perror("execv worker");
    }

    return 0;
}
