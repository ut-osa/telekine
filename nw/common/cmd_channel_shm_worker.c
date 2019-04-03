#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/guest_mem.h"
#include "common/ioctl.h"
#include "common/socket.h"

#include "worker.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

struct command_channel_shm {
    struct command_channel_base base;
    int guestlib_fd;
    int listen_fd;
    int shm_fd;

    /* netlink */
    int netlink_fd;
    struct sockaddr_nl src_addr;
    struct sockaddr_nl dst_addr;
    struct nlmsghdr *nlh;
    struct msghdr *nl_msg;

    struct pollfd pfd;
    MemoryRegion fifo;
    MemoryRegion dstore;
    struct param_block param_block;

    int vm_id;
    int listen_port;
    uint8_t init_command_type;
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

//! Utilities

static void command_channel_shm_report_storage_resource_allocation(struct command_channel*c, const char* const name, ssize_t amount)
{
    struct command_channel_shm* chan = (struct command_channel_shm *)c;
    struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);

    if (!strcmp(name, "device_memory")) {
        raw_msg->command_id = NW_ALLOCATE_DEVICE_MEMORY;
        *((long *)raw_msg->reserved_area) = (long)amount;
        sendmsg(chan->netlink_fd, chan->nl_msg, 0);
    }
}

static void command_channel_shm_report_throughput_resource_consumption(struct command_channel* c, const char* const name, ssize_t amount)
{
    struct command_channel_shm* chan = (struct command_channel_shm *)c;
    struct command_base *raw_msg = (struct command_base *)NLMSG_DATA(chan->nlh);

    if (!strcmp(name, "command_rate")) {
        raw_msg->command_id = NW_CONSUME_COMMAND_RATE;
        *((int *)raw_msg->reserved_area) = (int)amount;
        sendmsg(chan->netlink_fd, chan->nl_msg, 0);
    }

    if (!strcmp(name, "device_time")) {
        raw_msg->command_id = NW_CONSUME_DEVICE_TIME;
        *((long *)raw_msg->reserved_area) = (long)amount;
        sendmsg(chan->netlink_fd, chan->nl_msg, 0);
    }
}

//! Sending

/**
 * Compute the buffer size that will actually be used for a buffer of
 * `size`. The returned value may be larger than `size`.
 */
static size_t command_channel_shm_buffer_size(struct command_channel* chan, size_t size) {
    // For shared memory implementations this should round the size up
    // to a cache line, so as to maintain the alignment of buffers when
    // they are concatinated into the data region.

    // TODO: alignment (round up to command_channel_shm->alignment)
    return size;
}

/**
 * Reserve a memory region on BAR
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
    if (block->cur_offset + size >= block->size)
        block->cur_offset = (block->size >> 1);

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
        cmd->data_region = (void *)seeker->local_offset;
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

    cmd->command_type = MSG_RESPONSE; // TODO: remove me
    DEBUG_PRINT("[worker#%d] send message to guestlib\n", chan->listen_port);
    command_channel_shm_print_command(c, cmd);

    /* vsock interposition does not block send_message */
    send_socket(chan->guestlib_fd, cmd, cmd->command_size);
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
    struct command_channel_shm *chan = (struct command_channel_shm *)c;
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

    /* terminate worker when guestlib exits */
    if (chan->pfd.revents & POLLRDHUP) {
        printf("[worker#%d] guestlib shutdown\n", chan->listen_port);
        close(chan->pfd.fd);
        exit(-1);
    }

    if (chan->pfd.revents & POLLIN) {
        DEBUG_PRINT("[worker#%d] start to recv guestlib message\n", chan->listen_port);
        recv_socket(chan->guestlib_fd, &cmd_base, sizeof(struct command_base));
        DEBUG_PRINT("[worker#%d] recv guestlib message\n", chan->listen_port);
        cmd = (struct command_base *)malloc(cmd_base.command_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));
        recv_socket(chan->guestlib_fd, (void *)cmd + sizeof(struct command_base),
                    cmd_base.command_size - sizeof(struct command_base));
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
    struct command_channel_shm *chan = (struct command_channel_shm *)c;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    if (buffer_id)
        return (void *)((uintptr_t)chan->param_block.base + seeker->local_offset + (uintptr_t)buffer_id);
    else
        return NULL;
}

/**
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
static void* command_channel_shm_get_data_region(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_shm *chan = (struct command_channel_shm *)c;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    return (void *)((uintptr_t)chan->param_block.base + seeker->local_offset);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
static void command_channel_shm_free_command(struct command_channel* c, struct command_base* cmd)
{
    free(cmd);
}

/**
 * Initialize a new command channel for worker with vsock as doorbell and
 * shared memory as data transport.
 */
struct command_channel* command_channel_shm_worker_new(int vm_id, int rt_type, int listen_port,
        uintptr_t param_block_local_offset, size_t param_block_size)
{
    struct command_channel_shm *chan = (struct command_channel_shm *)malloc(sizeof(struct command_channel_shm));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_shm_vtable);

    /* set up executor worker info */
    chan->fifo.size = EXECUTOR_FIFO_SIZE;
    chan->dstore.size = EXECUTOR_DSTORE_SIZE;

    // TODO: notify executor when VM created or destroyed
    printf("spawn worker vm_id#%d, rt_type#%d\n", chan->vm_id, chan->init_command_type);
    chan->vm_id = vm_id;
    chan->init_command_type = rt_type;
    chan->listen_port = listen_port;

    /* setup shared memory */
    if ((chan->shm_fd = open("/dev/kvm-vgpu", O_RDWR | O_NONBLOCK)) < 0) {
        printf("failed to open /dev/kvm-vgpu\n");
        exit(0);
    }
    chan->fifo.addr = mmap(NULL, chan->fifo.size + chan->dstore.size,
                           PROT_READ | PROT_WRITE, MAP_SHARED, chan->shm_fd, 0);
    if (chan->fifo.addr == MAP_FAILED) {
        printf("mmap shared memory failed: %s\n", strerror(errno));
        // TODO: add exit labels
        exit(0);
    }
    else
        printf("mmap shared memory to 0x%lx\n", (uintptr_t)chan->fifo.addr);
    chan->dstore.addr = chan->fifo.addr + chan->fifo.size;

    /* worker uses the last half of the parameter block.
     *   base: start address of the whole parameter block;
     *   size: size of the block;
     *   offset: offset of the block to the VM's shared memory base;
     *   cur_offset: the moving pointer for attaching buffers. */
    chan->param_block.cur_offset = (param_block_size >> 1);
    chan->param_block.offset = param_block_local_offset;
    chan->param_block.size = param_block_size;
    chan->param_block.base = chan->dstore.addr + (vm_id - 1) * VGPU_DSTORE_SIZE + param_block_local_offset;

    if (ioctl(chan->shm_fd, KVM_NOTIFY_EXEC_SPAWN, (unsigned long)chan->vm_id) < 0) {
        printf("failed to notify FIFO address\n");
        exit(0);
    }
    printf("[worker#%d] kvm-vgpu notified\n", chan->vm_id);

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
    ssize_t retsize;
    retsize = sendmsg(chan->netlink_fd, chan->nl_msg, 0);
    if (retsize < 0) {
        printf("sendmsg failed with errcode %s\n", strerror(errno));
        exit(-1);
    }
    else
        printf("[worker#%d] kvm-vgpu netlink notified\n", chan->vm_id);

    //recvmsg(chan->netlink_fd, chan->nl_msg, 0);
    //DEBUG_PRINT("receive netlink cmd_id=%lu\n", raw_msg->command_id);
#endif

    printf("[worker#%d] waiting for guestlib connection\n", chan->vm_id);
    chan->guestlib_fd = accept_vm_socket(chan->listen_fd);
    printf("[worker#%d] guestlib connection accepted\n", chan->vm_id);

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

/**
 * Disconnect worker's command channel and free all resources associated
 * with it.
 */
static void command_channel_shm_free(struct command_channel* c)
{
    struct command_channel_shm *chan = (struct command_channel_shm *)c;

    munmap(chan->fifo.addr, chan->fifo.size + chan->dstore.size);
    if (chan->shm_fd > 0)
        close(chan->shm_fd);
    if (chan->nl_msg)
        free_netlink_msg(chan->nl_msg);
    free(chan);
}

static struct command_channel_vtable command_channel_shm_vtable = {
    command_channel_shm_buffer_size,
    command_channel_shm_new_command,
    command_channel_shm_attach_buffer,
    command_channel_shm_send_command,
    command_channel_shm_receive_command,
    command_channel_shm_get_buffer,
    command_channel_shm_get_data_region,
    command_channel_shm_free_command,
    command_channel_shm_free,
    command_channel_shm_print_command,
    command_channel_shm_report_storage_resource_allocation,
    command_channel_shm_report_throughput_resource_consumption
};
