#ifndef __VGPU_COMMON_SOCKET_H__
#define __VGPU_COMMON_SOCKET_H__

#ifdef __KERNEL__

#include <linux/types.h>
#include <linux/virtio_vsock.h>

#else

#include <stdint.h>
#include <stdlib.h>

#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <sys/types.h>
#include <linux/netlink.h>

#endif

#include "devconf.h"

typedef enum {
    MSG_UNDEFINED,
    MSG_NEW_INVOCATION,
    MSG_NEW_APPLICATION,
    MSG_SWAPPING,
    MSG_RESPONSE,
    MSG_SHUTDOWN,
} MESSAGE;

#define NW_NEW_WORKER               1
#define NW_CONSUME_DEVICE_TIME      10
#define NW_CONSUME_COMMAND_RATE     13
#define NW_ALLOCATE_DEVICE_MEMORY   11
#define NW_DEALLOCATE_DEVICE_MEMORY 12
#define COMMAND_SWAP_OUT            100
#define COMMAND_SWAP_IN             101

/* replace nw/include/kvm.h
typedef enum { STATUS_TASK_UNDEFINED,
               STATUS_TASK_DONE,
               STATUS_TASK_RUNNING,
               STATUS_TASK_ASYNC,
               STATUS_TASK_CALLBACK,
               STATUS_TASK_ERROR,
               STATUS_TASK_CONTINUE,
               STATUS_CALLBACK_POLL,
               STATUS_CALLBACK_FILLED,
               STATUS_CALLBACK_DONE,
               STATUS_CALLBACK_TIMEOUT,
               STATUS_CALLBACK_ERROR
             } STATUS;
*/

/* invocation modes */
#define MODE_ASYNC (ENABLE_ASYNC ? 1 : 0)

/*
struct message {
    MESSAGE type;
    // STATUS status;
    uint8_t status;
    int payload; // e.g. worker_port

    uint8_t vm_id;
    uint8_t rt_type;
    uintptr_t desc_slab_offset;
    struct desc_slab* desc_slab; // for fast indexing

    // TODO: move these fields from desc to message
    uint8_t cmd_id;
    uint8_t mode; // optimizations

    uint8_t rate_limit;
#if ENABLE_RATE_LIMIT
    uint64_t device_time;
#endif
    char reserved_area[64];
};
*/

#define INIT_MESSAGE(_msg, _slab, _type)                \
    struct message _msg;                                \
    _msg.type = _type;                                  \
    _msg.vm_id = vm_id;                                 \
    _msg.desc_slab = _slab;                             \
    if (_slab != NULL)                                  \
	_msg.desc_slab_offset = _msg.desc_slab->offset; \
    _msg.rt_type = RUNTIME_TYPE;                        \
    _msg.mode = 0;                                      \
    _msg.rate_limit = 0

int init_netlink_socket(struct sockaddr_nl *src_addr, struct sockaddr_nl *dst_addr);
struct nlmsghdr *init_netlink_msg(struct sockaddr_nl *dst_addr, struct msghdr *msg, size_t size);
void free_netlink_msg(struct msghdr *msg);

int init_vm_socket(struct sockaddr_vm* sa, int cid, int port);
int conn_vm_socket(int sockfd, struct sockaddr_vm* sa);
void listen_vm_socket(int listen_fd, struct sockaddr_vm *sa_listen);
int accept_vm_socket(int listen_fd);
int send_socket(int sockfd, void *buf, size_t size);
int recv_socket(int sockfd, void *buf, size_t size);

#endif
