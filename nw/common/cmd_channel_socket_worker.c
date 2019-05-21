#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <assert.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct command_channel_socket {
    struct command_channel_base base;
    int guestlib_fd;
    struct pollfd pfd;

    /* netlink */
    int netlink_fd;
    struct sockaddr_nl src_addr;
    struct sockaddr_nl dst_addr;
    struct nlmsghdr *nlh;
    struct msghdr *nl_msg;

    int vm_id;
    int listen_fd;
    int listen_port;
    uint8_t init_command_type;
};

struct block_seeker {
    uintptr_t cur_offset;
};

static struct command_channel_vtable command_channel_socket_vtable;

/**
 * Print a command for debugging.
 */
static void command_channel_socket_print_command(struct command_channel* chan, struct command_base* cmd)
{
    DEBUG_PRINT("struct command_base {\n"
                "  command_type=%ld\n"
                "  flags=%d\n"
                "  api_id=%d\n"
                "  command_id=%ld\n"
                "  command_size=%lx\n"
                "}\n",
                cmd->command_type,
                cmd->flags,
                cmd->api_id,
                cmd->command_id,
                cmd->command_size);
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
void command_channel_socket_free(struct command_channel* c) {
    struct command_channel_socket* chan = (struct command_channel_socket*)c;
    close(chan->guestlib_fd);
    if (chan->nl_msg)
        free_netlink_msg(chan->nl_msg);
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
size_t command_channel_socket_buffer_size(struct command_channel* c, size_t size) {
    return size;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
struct command_base* command_channel_socket_new_command(struct command_channel* c, size_t command_struct_size, size_t data_region_size) {
    struct command_base *cmd = (struct command_base *)malloc(command_struct_size + data_region_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    memset(cmd, 0, command_struct_size + data_region_size);
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
void* command_channel_socket_attach_buffer(struct command_channel* c, struct command_base* cmd, void* buffer, size_t size) {
    assert(buffer && size != 0);

    struct command_channel_socket* chan = (struct command_channel_socket *)c;
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
void command_channel_socket_send_command(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    cmd->command_type = MSG_NEW_INVOCATION;
    cmd->vm_id = chan->vm_id;

    /* vsock interposition does not block send_message */
    send_socket(chan->guestlib_fd, cmd, cmd->command_size + cmd->region_size);
}

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
struct command_base* command_channel_socket_receive_command(struct command_channel* c)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    struct command_base cmd_base;
    struct command_base *cmd;
    ssize_t ret;

    ret = poll(&chan->pfd, 1, -1);
    if (ret < 0) {
        fprintf(stderr, "failed to poll\n");
        abort();
    }

    DEBUG_PRINT("revents=%d\n", chan->pfd.revents);
    if (chan->pfd.revents == 0)
        return NULL;

    /* terminate guestlib when worker exits */
    if (chan->pfd.revents & POLLRDHUP) {
        DEBUG_PRINT("guestlib shutdown\n");
        close(chan->pfd.fd);
        abort();
    }

    if (chan->pfd.revents & POLLIN) {
        memset(&cmd_base, 0, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
        cmd = (struct command_base *)malloc(cmd_base.command_size + cmd_base.region_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
                    cmd_base.command_size + cmd_base.region_size - sizeof(struct command_base));
        DEBUG_PRINT("receive new command:\n");
        command_channel_socket_print_command(c, cmd);
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
void* command_channel_socket_get_buffer(struct command_channel* chan, struct command_base* cmd, void* buffer_id) {
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
static void* command_channel_socket_get_data_region(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)c;
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
void command_channel_socket_free_command(struct command_channel* c, struct command_base* cmd) {
    free(cmd);
}

struct command_channel* command_channel_socket_worker_new(int dummy1, int rt_type, int listen_port,
        uintptr_t dummy2, size_t dummy3)
{
    struct command_channel_socket *chan = (struct command_channel_socket *)malloc(sizeof(struct command_channel_socket));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_socket_vtable);

    // TODO: notify executor when VM created or destroyed
    printf("spawn worker port#%d, rt_type#%d\n", listen_port, chan->init_command_type);
    chan->init_command_type = rt_type;
    chan->listen_port = listen_port;

    /* setup shared memory */
    // TODO: netlink connection between hypervisor may be enough.
    /*
    if ((chan->shm_fd = open("/dev/kvm-vgpu", O_RDWR | O_NONBLOCK)) < 0) {
        printf("failed to open /dev/kvm-vgpu\n");
        abort();
    }

    if (ioctl(chan->shm_fd, KVM_NOTIFY_EXEC_SPAWN, (unsigned long)chan->vm_id) < 0) {
        printf("failed to notify FIFO address\n");
        abort();
    }
    printf("[worker#%d] kvm-vgpu notified\n", chan->vm_id);
    */

    /* connect hypervisor */
#if ENABLE_SWAP | ENABLE_RATE_LIMIT
    chan->netlink_fd = init_netlink_socket(&chan->src_addr, &chan->dst_addr);
    chan->nl_msg = (struct msghdr *)malloc(sizeof(struct msghdr));
    chan->nlh = init_netlink_msg(&chan->dst_addr, chan->nl_msg, sizeof(struct command_base));
#endif

    /* connect guestlib */
    struct sockaddr_vm sa_listen;
    chan->listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, chan->listen_port);
    listen_vm_socket(chan->listen_fd, &sa_listen);

#if ENABLE_SWAP | ENABLE_RATE_LIMIT
    /* connect hypervisor */
    struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);
    raw_msg->api_id = INTERNAL_API;
    raw_msg->command_id = NW_NEW_WORKER;
    *((int *)raw_msg->reserved_area) = chan->listen_port;
    raw_msg->vm_id = chan->vm_id;
    sendmsg(chan->netlink_fd, chan->nl_msg, 0);
    printf("[worker#%d] kvm-vgpu netlink notified\n", chan->vm_id);

    //recvmsg(chan->netlink_fd, &nl_msg, 0);
    //DEBUG_PRINT("receive netlink cmd_id=%lu\n", raw_msg->command_id);
#endif

    printf("[worker#%d] waiting for guestlib connection\n", listen_port);
    chan->guestlib_fd = accept_vm_socket(chan->listen_fd);

    // TODO: also poll netlink socket, and put the swapping task in the same
    // task queue just as the normal invocations.
    chan->pfd.fd = chan->guestlib_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    /*
    if (fcntl(ex_st.client_fd, F_SETFL,
              fcntl(ex_st.client_fd, F_GETFL) & (~O_NONBLOCK)) < 0) {
        perror("fcntl blocking failed");
        return 0;
    }
    */

    return (struct command_channel *)chan;
}

static struct command_channel_vtable command_channel_socket_vtable = {
  command_channel_socket_buffer_size,
  command_channel_socket_new_command,
  command_channel_socket_attach_buffer,
  command_channel_socket_send_command,
  command_channel_socket_receive_command,
  command_channel_socket_get_buffer,
  command_channel_socket_get_data_region,
  command_channel_socket_free_command,
  command_channel_socket_free,
  command_channel_socket_print_command
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the command_channel_socket_new and other socket specific APIs.
