#include "common/cmd_channel_impl.h"
#include "common/debug.h"
#include "common/devconf.h"
#include "common/guest_mem.h"
#include "common/ioctl.h"
#include "common/socket.h"

#include "memory.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <sys/types.h>
#include <assert.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern int vm_id;

struct command_channel_shm {
    struct command_channel_base base;
    int sock_fd;
    int shm_fd;
    struct pollfd pfd;
    //struct desc_slab desc_slab_list;
    struct param_block param_block;
    int vm_id;
};

static struct command_channel_vtable command_channel_shm_vtable;

/**
 * Print a command for debugging.
 */
static void command_channel_shm_print_command(struct command_channel* chan, struct command_base* cmd)
{
    DEBUG_PRINT("struct command_base {\n"
                "  command_type=%ld\n"
                "  vm_id=%d\n"
                "  flags=%d\n"
                "  api_id=%d\n"
                "  command_id=%ld\n"
                "  command_size=%lx\n"
                "  region_size=%lx\n"
                "}\n",
                cmd->command_type,
                cmd->vm_id,
                cmd->flags,
                cmd->api_id,
                cmd->command_id,
                cmd->command_size,
                cmd->region_size);
}

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 */
static size_t command_channel_shm_buffer_size(struct command_channel* chan, size_t size)
{
    // For shared memory implementations this should round the size up
    // to a cache line, so as to maintain the alignment of buffers when
    // they are concatinated into the data region.

    // TODO: alignment (round up to command_channel_shm->alignment)
    return size;
}

/**
 * Reserve a memory region on BAR.
 *
 * Return the offset of the region or NULL if no enough space.
 * Guestlib->worker communication uses the first half of the space, and
 * the reverse communication uses the second half.
 *
 * @size: the size of the memory region.
 */
static uintptr_t reserve_param_block(struct param_block *block, size_t size)
{
    uintptr_t ret_offset;

    // TODO: add lock for multi-threading
    // TODO: implement the **real** memory allocator (mask used regions)
    if (block->cur_offset + size >= (block->size >> 1))
        block->cur_offset = 0;

    ret_offset = (uintptr_t)block->cur_offset;
    block->cur_offset += size;

    return ret_offset;
}

/**
 * Allocate a new command struct with size `command_struct_size` and
 * a (potientially imaginary) data region of size `data_region_size`.
 *
 * `data_region_size` should be computed by adding up the result of
 * calls to `command_channel_buffer_size` on the same channel.
 */
static struct command_base* command_channel_shm_new_command(struct command_channel* c, size_t command_struct_size, size_t data_region_size)
{
    struct command_channel_shm* chan = (struct command_channel_shm *)c;
    struct command_base *cmd = (struct command_base *)malloc(command_struct_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    memset(cmd, 0, command_struct_size);
    cmd->command_size = command_struct_size;
    if (data_region_size) {
        data_region_size += 0x4;
        seeker->local_offset = reserve_param_block(&chan->param_block, data_region_size);
        seeker->cur_offset = seeker->local_offset + 0x4;
        cmd->data_region = (void *)(seeker->local_offset + chan->param_block.offset);
    }
    cmd->region_size = data_region_size;
    cmd->vm_id = chan->vm_id;

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
static void* command_channel_shm_attach_buffer(struct command_channel* c, struct command_base* cmd, void* buffer, size_t size)
{
    assert(buffer && size != 0);

    struct command_channel_shm* chan = (struct command_channel_shm *)c;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    void *offset = (void *)(seeker->cur_offset - seeker->local_offset);
    seeker->cur_offset += size;
    void *dst = (void *)((uintptr_t)chan->param_block.base + seeker->local_offset + (uintptr_t)offset);
    memcpy(dst, buffer, size);

    return offset;
}

/**
 * Send the message and all its attached buffers.
 *
 * This call is asynchronous and does not block for the command to
 * complete execution.
 */
static void command_channel_shm_send_command(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_shm * chan = (struct command_channel_shm *)c;

    cmd->command_type = MSG_NEW_INVOCATION;

    /* vsock interposition does not block send_message */
    send_socket(chan->sock_fd, cmd, cmd->command_size);
}

//! Receiving

/**
 * Receive a command from a channel. The returned Command pointer
 * should be interpreted based on its `command_id` field.
 *
 * This call blocks waiting for a command to be sent along this
 * channel.
 */
static struct command_base* command_channel_shm_receive_command(struct command_channel* c)
{
    struct command_channel_shm * chan = (struct command_channel_shm *)c;
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
        DEBUG_PRINT("worker shutdown\n");
        close(chan->pfd.fd);
        abort();
    }

    if (chan->pfd.revents & POLLIN) {
        memset(&cmd_base, 0, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
        cmd = (struct command_base *)malloc(cmd_base.command_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));
        recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
                    cmd_base.command_size - sizeof(struct command_base));
        DEBUG_PRINT("receive new command:\n");
        command_channel_shm_print_command(c, cmd);
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
static void* command_channel_shm_get_buffer(struct command_channel* c, struct command_base* cmd, void* buffer_id)
{
    struct command_channel_shm * chan = (struct command_channel_shm *)c;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    if (buffer_id)
        return (void *)((uintptr_t)chan->param_block.base + seeker->local_offset + (uintptr_t)buffer_id);
    else
        return NULL;
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
static void command_channel_shm_free_command(struct command_channel* chan, struct command_base* cmd)
{
    free(cmd);
}

/**
 * Initialize a new command channel with vsock as doorbell and shared
 * memory as data transport.
 */
struct command_channel* command_channel_shm_new()
{
    struct command_channel_shm *chan = (struct command_channel_shm *)malloc(sizeof(struct command_channel_shm));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_shm_vtable);

    /* setup shared memory */
    char dev_filename[32];
    sprintf(dev_filename, "/dev/%s%d", VGPU_DEV_NAME, VGPU_DRIVER_MINOR);

    chan->shm_fd = open(dev_filename, O_RDWR);
    if (chan->shm_fd < 0) {
        printf("failed to open device %s\n", dev_filename);
        abort();
    }

    /* acquire vm id */
    chan->vm_id = ioctl(chan->shm_fd, IOCTL_GET_VGPU_ID);
    if (chan->vm_id <= 0) {
        printf("failed to retrieve vm id: %d\n", chan->vm_id);
        abort();
    }
    DEBUG_PRINT("assigned vm_id=%d\n", chan->vm_id);

    chan->param_block.size = DEFAULT_PARAM_BLOCK_SIZE;
    chan->param_block.offset = ioctl(chan->shm_fd, IOCTL_REQUEST_PARAM_BLOCK, chan->param_block.size);
    chan->param_block.base = mmap(NULL, chan->param_block.size,
                                  PROT_READ | PROT_WRITE, MAP_SHARED, chan->shm_fd, 0);

    /* connect worker manager and send vm_id, param_block offset (inside
     * the VM's shared memory region) and param_block size. */
    struct sockaddr_vm sa;
    int manager_fd = init_vm_socket(&sa, VMADDR_CID_HOST, WORKER_MANAGER_PORT);
    conn_vm_socket(manager_fd, &sa);

    struct command_base* msg = command_channel_shm_new_command((struct command_channel *)chan, sizeof(struct command_base), 0);
    msg->command_type = MSG_NEW_APPLICATION;
    struct param_block_info *pb_info = (struct param_block_info *)msg->reserved_area;
    pb_info->param_local_offset = chan->param_block.offset;
    pb_info->param_block_size = chan->param_block.size;
    send_socket(manager_fd, msg, sizeof(struct command_base));

    recv_socket(manager_fd, msg, sizeof(struct command_base));
    uintptr_t worker_port = *((uintptr_t *)msg->reserved_area);
    command_channel_shm_free_command((struct command_channel *)chan, msg);
    close(manager_fd);

    /* connect worker */
    DEBUG_PRINT("assigned worker at %lu\n", worker_port);
    chan->sock_fd = init_vm_socket(&sa, VMADDR_CID_HOST, worker_port);
    // FIXME: connect is always non-blocking for vm socket!
    usleep(5000000);
    conn_vm_socket(chan->sock_fd, &sa);

    chan->pfd.fd = chan->sock_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    return (struct command_channel *)chan;
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
static void command_channel_shm_free(struct command_channel* c) {
    struct command_channel_shm * chan = (struct command_channel_shm *)c;

    munmap(chan->param_block.base, chan->param_block.size);
    // TODO: unmap slabs
    // TODO: destroy sems

    close(chan->sock_fd);
    close(chan->shm_fd);
    free(chan);
}

static struct command_channel_vtable command_channel_shm_vtable = {
  command_channel_shm_buffer_size,
  command_channel_shm_new_command,
  command_channel_shm_attach_buffer,
  command_channel_shm_send_command,
  command_channel_shm_receive_command,
  command_channel_shm_get_buffer,
  NULL,
  command_channel_shm_free_command,
  command_channel_shm_free,
  command_channel_shm_print_command
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the command_channel_socket_new and other socket specific APIs.
