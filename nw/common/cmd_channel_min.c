#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"
#include "common/guest_mem.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


struct command_channel_min {
    struct command_channel_base base;
    int sock_fd;
    struct pollfd pfd;
};

static struct command_channel_vtable command_channel_min_vtable;

/**
 * Print a command for debugging.
 */
static void command_channel_min_print_command(struct command_channel* chan, struct command_base* cmd)
{
    DEBUG_PRINT("struct command_base {\n"
                "  command_type=%ld\n"
                "  flags=%d\n"
                "  api_id=%d\n"
                "  command_id=%ld\n"
                "  command_size=%lx\n"
                "  region_size=%lx\n"
                "}\n",
                cmd->command_type,
                cmd->flags,
                cmd->api_id,
                cmd->command_id,
                cmd->command_size,
                cmd->region_size);
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
void command_channel_min_free(struct command_channel* c) {
    struct command_channel_min* chan = (struct command_channel_min*)c;
    close(chan->sock_fd);
    free(chan);
}

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 * For shared memory implementations this should round the size up
 * to a cache line, so as to maintain the alignment of buffers when
 * they are concatinated into the data region.
 */
size_t command_channel_min_buffer_size(struct command_channel* c, size_t size) {
    return size;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
struct command_base* command_channel_min_new_command(struct command_channel* c, size_t command_struct_size, size_t data_region_size) {
    struct command_base *cmd = (struct command_base *)malloc(command_struct_size + data_region_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    memset(cmd, 0, command_struct_size + data_region_size);
    cmd->vm_id = 1;
    cmd->command_size = command_struct_size;
    cmd->data_region = (void *)command_struct_size;
    cmd->region_size = data_region_size;
    seeker->cur_offset = command_struct_size;

    return cmd;
}

/**
 * Attach a buffer to a command and return a location independent
 * buffer ID. `buffer` must be valid until after the call to
 * `command_channel_send_command`.
 *
 * The combined attached buffers must fit within the initially
 * provided `data_region_size` (to `command_channel_new_command`).
 */
void* command_channel_min_attach_buffer(struct command_channel* c, struct command_base* cmd, void* buffer, size_t size) {
    assert(buffer && size != 0);

    struct command_channel_min* chan = (struct command_channel_min *)c;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    void *offset = (void *)seeker->cur_offset;
    void *dst = (void *)((uintptr_t)cmd + seeker->cur_offset);
    seeker->cur_offset += size;
    memcpy(dst, buffer, size);

    return offset;
}

/**
 * Send the message and all its attached buffers.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 */
void command_channel_min_send_command(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_min *chan = (struct command_channel_min *)c;
    cmd->command_type = MSG_NEW_INVOCATION;

    /* vsock interposition does not block send_message */
    send_socket(chan->sock_fd, cmd, cmd->command_size + cmd->region_size);
}

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
struct command_base* command_channel_min_receive_command(struct command_channel* c)
{
    struct command_channel_min *chan = (struct command_channel_min *)c;
    struct command_base cmd_base;
    struct command_base *cmd;
    ssize_t ret;

    ret = poll(&chan->pfd, 1, -1);
    if (ret < 0) {
        fprintf(stderr, "failed to poll\n");
        exit(-1);
    }

    DEBUG_PRINT("revents=%d\n", chan->pfd.revents);
    if (chan->pfd.revents == 0)
        return NULL;

    /* terminate guestlib when worker exits */
    if (chan->pfd.revents & POLLRDHUP) {
        DEBUG_PRINT("worker shutdown\n");
        close(chan->pfd.fd);
        exit(-1);
    }

    if (chan->pfd.revents & POLLIN) {
        memset(&cmd_base, 0, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
        cmd = (struct command_base *)malloc(cmd_base.command_size + cmd_base.region_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
                    cmd_base.command_size + cmd_base.region_size - sizeof(struct command_base));
        DEBUG_PRINT("receive new command:\n");
        command_channel_min_print_command(c, cmd);
        return cmd;
    }

    return NULL;
}

/**
 * Translate a buffer_id (as returned by
 * `command_channel_attach_buffer` in the sender) into a data pointer.
 * The returned pointer will be valid until
 * `command_channel_free_command` is called on `cmd`.
 */
void* command_channel_min_get_buffer(struct command_channel* chan, struct command_base* cmd, void* buffer_id) {
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
void command_channel_min_free_command(struct command_channel* c, struct command_base* cmd) {
    free(cmd);
}

struct command_channel* command_channel_min_new()
{
    struct command_channel_min *chan = (struct command_channel_min *)malloc(sizeof(struct command_channel_min));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_min_vtable);

    /* connect manager to get worker port */
    int ret;
    int manager_fd;

    const char *ip_str = getenv("AVA_IP");
    if (!ip_str)
       ip_str = "127.0.0.1";

    struct addrinfo hints = {
       .ai_family = AF_INET,
       .ai_socktype = SOCK_STREAM,
    };
    struct addrinfo *result, *rp;
    if ((ret = getaddrinfo(ip_str, "4000", &hints, &result))) {
       if (ret == EAI_SYSTEM)
          perror(ip_str);
       else
          fprintf(stderr, "%s: %s\n", ip_str, gai_strerror(ret));
       abort();
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
       manager_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
       if (manager_fd == -1)
          continue;
       if (connect(manager_fd, rp->ai_addr, rp->ai_addrlen) != -1)
          break;
       close(manager_fd);
    }
    if (rp == NULL) {
       fprintf(stderr, "%s:%d couldn't connect!\n", __FILE__, __LINE__);
       abort();
    }
    freeaddrinfo(result);

    struct command_base* msg = command_channel_min_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
    msg->command_type = MSG_NEW_APPLICATION;
    msg->vm_id = 1;
    struct param_block_info *pb_info = (struct param_block_info *)msg->reserved_area;
    pb_info->param_local_offset = 0;
    pb_info->param_block_size = 0;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
    command_channel_min_free_command((struct command_channel *)chan, msg);
    close(manager_fd);

    /* connect worker */
    DEBUG_PRINT("assigned worker at %lu\n", worker_port);

    char port_buf[128];
    sprintf(port_buf, "%lu", worker_port);
    if (getaddrinfo(ip_str, port_buf, &hints, &result)) {
       perror(ip_str);
       exit(1);
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
       chan->sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
       if (chan->sock_fd == -1)
          continue;
       if (connect(chan->sock_fd, rp->ai_addr, rp->ai_addrlen) != -1)
          break;
       close(chan->sock_fd);
    }
    if (rp == NULL) {
       fprintf(stderr, "%s:%d couldn't connect!\n", __FILE__, __LINE__);
       exit(1);
    }

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

static struct command_channel_vtable command_channel_min_vtable = {
  command_channel_min_buffer_size,
  command_channel_min_new_command,
  command_channel_min_attach_buffer,
  command_channel_min_send_command,
  command_channel_min_receive_command,
  command_channel_min_get_buffer,
  NULL,
  command_channel_min_free_command,
  command_channel_min_free,
  command_channel_min_print_command
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the command_channel_min_new and other socket specific APIs.
