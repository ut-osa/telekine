#include "common/cmd_channel_impl.h"
#include "common/devconf.h"
#include "common/debug.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <netdb.h>

static int ssl_enabled_impl(void)
{
    const char* env = getenv("AVA_ENABLE_SSL");
    if (env == NULL || atoi(env) == 0) {
        return 0;
    } else {
        return 1;
    }
}

static int ssl_enabled(void)
{
    static int enabled = -1;
    if (enabled == -1) enabled = ssl_enabled_impl();
    return enabled;
}


struct command_channel_min {
    struct command_channel_base base;
    int guestlib_fd;
    struct pollfd pfd;
    void* ssl_ctx;
    void* ssl;
    pthread_mutex_t send_mutex;

    /* netlink */
    int netlink_fd;
    struct sockaddr_nl src_addr;
    struct sockaddr_nl dst_addr;
    struct nlmsghdr *nlh;
    struct msghdr *nl_msg;

    int listen_fd;
    uint8_t init_command_type;
};

struct block_seeker {
    uintptr_t cur_offset;
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
 * Utilities
 */
static void command_channel_min_report_storage_resource_allocation(struct command_channel*c, const char* const name, ssize_t amount)
{
    return;
}

static void command_channel_min_report_throughput_resource_consumption(struct command_channel* c, const char* const name, ssize_t amount)
{
    return;
}

/**
 * Disconnect this command channel and free all resources associated
 * with it.
 */
void command_channel_min_free(struct command_channel* c) {
    struct command_channel_min* chan = (struct command_channel_min*)c;
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

    pthread_mutex_lock(&chan->send_mutex);

    /* vsock interposition does not block send_message */
    if (ssl_enabled()) {
        send_ssl_socket(chan->ssl, cmd, cmd->command_size + cmd->region_size);
    } else {
        send_socket(chan->guestlib_fd, cmd, cmd->command_size + cmd->region_size);
    }

    pthread_mutex_unlock(&chan->send_mutex);
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
        fflush(stderr);
        abort();
    }

    DEBUG_PRINT("revents=%d\n", chan->pfd.revents);
    if (chan->pfd.revents == 0)
        return NULL;

    /* terminate guestlib when worker exits */
    if (chan->pfd.revents & POLLRDHUP) {
        fprintf(stderr, "poll HUP\n");
        fflush(stderr);
        DEBUG_PRINT("guestlib shutdown\n");
        close(chan->pfd.fd);
        abort();
    }

    if (chan->pfd.revents & POLLIN) {
        memset(&cmd_base, 0, sizeof(struct command_base));
        if (ssl_enabled()) {
            recv_ssl_socket(chan->ssl, &cmd_base, sizeof(struct command_base));
        } else {
            recv_socket(chan->pfd.fd, &cmd_base, sizeof(struct command_base));
        }
        cmd = (struct command_base *)malloc(cmd_base.command_size + cmd_base.region_size);
        memcpy(cmd, &cmd_base, sizeof(struct command_base));
        if (ssl_enabled()) {
            recv_ssl_socket(chan->ssl, (void *)cmd + sizeof(struct command_base),
                            cmd_base.command_size + cmd_base.region_size - sizeof(struct command_base));
        } else {
            recv_socket(chan->pfd.fd, (void *)cmd + sizeof(struct command_base),
                        cmd_base.command_size + cmd_base.region_size - sizeof(struct command_base));
        }
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
 * Returns the pointer to data region. The returned pointer is mainly
 * used for data extraction for migration.
 */
static void* command_channel_min_get_data_region(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_min *chan = (struct command_channel_min *)c;
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * Free a command returned by `command_channel_receive_command`.
 */
void command_channel_min_free_command(struct command_channel* c, struct command_base* cmd) {
    free(cmd);
}

void set_up_ports(int listen_port, int listen_fds[N_AVA_CHANNELS])
{
    /* connect guestlib */
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;
    struct command_base msg, response;

    for (int i = 0; i < N_AVA_CHANNELS; i++) {
       if ((listen_fds[i] = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
           perror("socket");
       }
       // Forcefully attaching socket to the worker port
       if (setsockopt(listen_fds[i], SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
           perror("setsockopt");
       }
       address.sin_family = AF_INET;
       address.sin_addr.s_addr = INADDR_ANY;
       address.sin_port = htons(listen_port);

       if (bind(listen_fds[i], (struct sockaddr *)&address, sizeof(address)) < 0) {
           perror("bind failed");
       }

       if (listen(listen_fds[i], 10) < 0) {
           perror("listen");
       }

       printf("[worker#%d] waiting for guestlib connection\n", listen_port);
       listen_port += CHANNEL_OFFSET;
    }
}

struct command_channel* command_channel_min_worker_new(int dummy1, int rt_type, int listen_fd,
        uintptr_t port, size_t dummy3)
{
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    struct command_channel_min *chan = (struct command_channel_min *)malloc(sizeof(struct command_channel_min));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_min_vtable);
    chan->init_command_type = rt_type;
    chan->listen_fd = listen_fd;

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    chan->guestlib_fd = accept(chan->listen_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

    chan->pfd.fd = chan->guestlib_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    if (ssl_enabled()) {
        fprintf(stderr, "Init SSL connection\n");
        init_ssl();
        const char* cert_file = getenv("AVA_SSL_CERT_FILE");
        const char* key_file = getenv("AVA_SSL_KEY_FILE");
        if (cert_file == NULL || key_file == NULL) {
            fprintf(stderr, "Empty AVA_SSL_CERT_FILE or AVA_SSL_KEY_FILE!\n");
            abort();
        }
        chan->ssl_ctx = create_ssl_server_context(cert_file, key_file);
        chan->ssl = create_ssl_session(chan->ssl_ctx, chan->guestlib_fd);
        ssl_accept(chan->ssl);
    }
    pthread_mutex_init(&chan->send_mutex, NULL);

    return (struct command_channel *)chan;
}

struct command_channel* command_channel_min_worker_new_reverse_socket(const char* remote, int port)
{
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    struct command_channel_min *chan = (struct command_channel_min *)malloc(sizeof(struct command_channel_min));
    command_channel_preinitialize((struct command_channel *)chan, &command_channel_min_vtable);
    chan->init_command_type = 0;

    struct addrinfo *result, *rp;
    struct addrinfo hints = {
       .ai_family = AF_INET,
       .ai_socktype = SOCK_STREAM,
    };

    fprintf(stderr, "Start command_channel_min_worker_new_reverse_socket\n");

    char port_buf[128];
    sprintf(port_buf, "%d", port);
    if (getaddrinfo(remote, port_buf, &hints, &result)) {
       perror(remote);
       abort();
    }

    int sock_fd;
    for (rp = result; rp != NULL; rp = rp->ai_next) {
       sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
       if (sock_fd == -1)
          continue;
       if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) != -1)
          break;
       close(sock_fd);
    }
    if (rp == NULL) {
       fprintf(stderr, "%s:%d couldn't connect!\n", __FILE__, __LINE__);
       abort();
    }

    chan->listen_fd = -1;
    chan->guestlib_fd = sock_fd;

    chan->pfd.fd = chan->guestlib_fd;
    chan->pfd.events = POLLIN | POLLRDHUP;

    if (ssl_enabled()) {
        fprintf(stderr, "Init SSL connection\n");
        init_ssl();
        chan->ssl_ctx = create_ssl_client_context();
        chan->ssl = create_ssl_session(chan->ssl_ctx, chan->guestlib_fd);
        ssl_connect(chan->ssl);
    }
    pthread_mutex_init(&chan->send_mutex, NULL);

    return (struct command_channel *)chan;
}

static struct command_channel_vtable command_channel_min_vtable = {
  command_channel_min_buffer_size,
  command_channel_min_new_command,
  command_channel_min_attach_buffer,
  command_channel_min_send_command,
  command_channel_min_receive_command,
  command_channel_min_get_buffer,
  command_channel_min_get_data_region,
  command_channel_min_free_command,
  command_channel_min_free,
  command_channel_min_print_command,
  command_channel_min_report_storage_resource_allocation,
  command_channel_min_report_throughput_resource_consumption
};

// warning TODO: Does there need to be a separate socket specific function which handles listening/accepting instead of connecting?

// warning TODO: Make a header file "cmd_channel_socket.h" for the command_channel_min_new and other socket specific APIs.
