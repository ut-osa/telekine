#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/stat.h>

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
    exit(0);
}

#define MAX_PATH 4096
int cannonize(char cannon[MAX_PATH], const char *bin)
{
    char next_cannon[MAX_PATH];
    struct stat stat_info = {0};

    if (stat(bin, &stat_info) == 0) {
       mode_t m = stat_info.st_mode;
       if (S_ISLNK(m) && (readlink(bin, cannon, MAX_PATH) == 0)) {
          int ret = cannonize(next_cannon, cannon);
          if (ret == 0)
             strncpy(cannon, next_cannon, MAX_PATH);
          return ret;
       } else if (S_ISREG(m)) {
          if (access(bin, X_OK) == 0) {
             strncpy(cannon, bin, MAX_PATH);
             return 0;
          }
       } else {
          errno = EINVAL;
       }
    }
    return 1;
}


int main(int argc, char *argv[])
{
    /* setup signal handler */
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    /* initialize TCP socket */
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    int client_fd;
    int i;
    int use_log_files = 0;
    pid_t child;
    struct command_base msg, response;
    struct param_block_info *pb_info;
    uintptr_t *worker_port;
    const char *worker_bin;
    char worker_bin_buf[4096];

    worker_id = 1;

    worker_bin = "./worker";
    for (i = 1; i < argc; i++) {
       if (strcmp(argv[i], "-l") == 0)
          use_log_files = 1;
       else
          worker_bin = argv[i];
    }

    if (cannonize(worker_bin_buf, worker_bin)) {
       perror(worker_bin);
       return 1;
    }

    worker_bin = worker_bin_buf;

    if ((listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket");
    }
    // Forcefully attaching socket to the port 4000
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( 4000 );

    if (bind(listen_fd, (struct sockaddr *)&address, sizeof(address))<0) {
        perror("bind failed");
    }
    if (listen(listen_fd, 10) < 0) {
        perror("listen");
    }
    int ready_pipe[2];

    /* polling new applications */
    do {
        client_fd = accept(listen_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);

        /* get guestlib info */
        recv_socket(client_fd, &msg, sizeof(struct command_base));
        if (msg.command_type != MSG_NEW_APPLICATION) {
            printf("[manager] wrong message type\n");
            close(client_fd);
            continue;
        }

        /* spawn a worker and wait for it to be ready */
        uint64_t val;
        pipe(ready_pipe);
        child = fork();
        if (child == 0) {
            close(ready_pipe[0]);
            close(listen_fd);
            break;
        }
        /* wait for the worker to listen... */
        close(ready_pipe[1]);
        read(ready_pipe[0], &val, sizeof(val));
        close(ready_pipe[0]);

        /* return worker port to guestlib */
        response.api_id = INTERNAL_API;
        worker_port = (uintptr_t *)response.reserved_area;
        *worker_port = worker_id + WORKER_PORT_BASE;
        send_socket(client_fd, &response, sizeof(struct command_base));
        close(client_fd);

        worker_id++;
    } while (1);

    if (use_log_files) {
       char stdout_fname[64], stderr_fname[64];
       int new_stdout, new_stderr;
       snprintf(stdout_fname, 64, "ava_%d_stdout.log", worker_id);
       snprintf(stderr_fname, 64, "ava_%d_stderr.log", worker_id);
       printf("worker output in %s/%s\n", stdout_fname, stderr_fname);

       new_stdout = open(stdout_fname, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR);
       if (new_stdout < 0) {
          perror(stdout_fname);
          return 1;
       }
       new_stderr = open(stderr_fname, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR);
       if (new_stderr < 0) {
          perror(stderr_fname);
          return 1;
       }
       dup2(new_stdout, 1);
       dup2(new_stderr, 2);
    }


    /* spawn worker */
    char str_vm_id[10];
    char str_rt_type[10];
    char str_port[10];
    char str_pb_offset[10];
    char str_pb_size[10];
    char str_rdy_pipe[10];
    pb_info = (struct param_block_info *)msg.reserved_area;
    sprintf(str_vm_id, "%d", msg.vm_id);
    sprintf(str_rt_type, "%d", msg.api_id);
    sprintf(str_port, "%d", worker_id + WORKER_PORT_BASE);
    sprintf(str_pb_offset, "%lu", pb_info->param_local_offset);
    sprintf(str_pb_size, "%lu", pb_info->param_block_size);
    sprintf(str_rdy_pipe, "%d", ready_pipe[1]);
    char *argv_list[] = {"worker.out",
                         str_vm_id, str_rt_type, str_port,
                         str_pb_offset, str_pb_size, str_rdy_pipe, NULL};
    printf("[manager] %s vm_id=%d, rt_type=%d, port=%s, pb_offset=%lx, pb_size=%lx, rdy_pipe=%d\n",
           worker_bin, msg.vm_id, msg.api_id, str_port,
           pb_info->param_local_offset,
           pb_info->param_block_size,  ready_pipe[1]);
    if (execv(worker_bin, argv_list) < 0) {
        perror("execv worker");
    }

    return 0;
}
