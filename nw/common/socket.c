#include "common/socket.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define SOL_NETLINK 270

int init_netlink_socket(struct sockaddr_nl *src_addr, struct sockaddr_nl *dst_addr)
{
    int sock_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_USERSOCK);
    if (sock_fd < 0) {
        perror("ERROR opening netlink socket");
        exit(0);
    }
    if (setsockopt(sock_fd, SOL_NETLINK, NETLINK_NO_ENOBUFS, (int[]){1}, sizeof(int)) != 0) {
        perror("ERROR setsockopt SOL_NETLINK");
    }

    memset(src_addr, 0, sizeof(struct sockaddr_nl));
    src_addr->nl_family = AF_NETLINK;
    src_addr->nl_pid = getpid();
    src_addr->nl_groups = 0;
    if (bind(sock_fd, (struct sockaddr *)src_addr, sizeof(struct sockaddr_nl)) != 0) {
        perror("ERROR bind netlink socket");
        close(sock_fd);
        exit(0);
    }

    memset(dst_addr, 0, sizeof(struct sockaddr_nl));
    dst_addr->nl_family = AF_NETLINK;
    dst_addr->nl_pid = 0;    /* kernel */
    dst_addr->nl_groups = 0; /* unicast */

    return sock_fd;
}

struct nlmsghdr *init_netlink_msg(struct sockaddr_nl *dst_addr, struct msghdr *msg, size_t size)
{
    struct nlmsghdr *nlh;
    struct iovec *iov;

    memset(msg, 0, sizeof(*msg));
    nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(size));
    memset(nlh, 0, NLMSG_SPACE(size));
    iov = (struct iovec *)malloc(sizeof(struct iovec));
    memset(iov, 0, sizeof(struct iovec));

    nlh->nlmsg_len = NLMSG_SPACE(size);
    nlh->nlmsg_pid = getpid();
    nlh->nlmsg_flags = 0;
    iov->iov_base = (void *)nlh;
    iov->iov_len = nlh->nlmsg_len;
    msg->msg_name = (void *)dst_addr;
    msg->msg_namelen = sizeof(*dst_addr);
    msg->msg_iov = iov;
    msg->msg_iovlen = 1;

    return nlh;
}

void free_netlink_msg(struct msghdr *msg)
{
    struct iovec *iov = (struct iovec *)msg->msg_iov;
    free(iov->iov_base);
    free(msg->msg_iov);
    free(msg);
}

int init_vm_socket(struct sockaddr_vm *sa, int cid, int port)
{
    int sockfd;

    memset(sa, 0, sizeof(struct sockaddr_vm));
    sa->svm_family = AF_VSOCK;
    sa->svm_cid = cid;
    sa->svm_port = port;

    sockfd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("ERROR opening socket");
        exit(0);
    }

    return sockfd;
}

void listen_vm_socket(int listen_fd, struct sockaddr_vm *sa_listen)
{
    if (bind(listen_fd, (struct sockaddr *)sa_listen, sizeof(*sa_listen)) != 0) {
        perror("ERROR bind");
        close(listen_fd);
        exit(0);
    }

    if (listen(listen_fd, 10) != 0) {
        perror("ERROR listen");
        close(listen_fd);
        exit(0);
    }
    printf("start vm socket listening at %d\n", sa_listen->svm_port);
}

int accept_vm_socket(int listen_fd)
{
    int client_fd;

    struct sockaddr_vm sa_client;
    socklen_t socklen_client = sizeof(sa_client);

    client_fd = accept(listen_fd, (struct sockaddr *)&sa_client, &socklen_client);
    if (client_fd < 0) {
        perror("ERROR accept");
        close(listen_fd);
        exit(0);
    }
    printf("connection from cid %u port %u\n", sa_client.svm_cid, sa_client.svm_port);

    return client_fd;
}

#if 1
int conn_vm_socket(int sockfd, struct sockaddr_vm *sa)
{
    int ret;
    int num_tries = 0;
    /*
    int sock_flags;

    sock_flags = fcntl(sockfd, F_GETFL);
    if (sock_flags & O_NONBLOCK) {
        DEBUG_PRINT("socket was non-blocking\n");
        if (fcntl(sockfd, F_SETFL, sock_flags & (~O_NONBLOCK)) < 0) {
            perror("fcntl blocking");
            exit(0);
        }
    }
    */

    while (num_tries < 5 && (ret = connect(sockfd, (struct sockaddr *)sa, sizeof(*sa))) < 0) {
        num_tries++;
        DEBUG_PRINT("errcode=%s, retry connection x%d...\n", strerror(errno), num_tries);
        usleep(100000);
    }
    return ret;
}

#else

int conn_vm_socket(int sockfd, struct sockaddr_vm *sa)
{
    int ret;
    int sock_flags;
    struct timeval tv;
    fd_set wset;

    sock_flags = fcntl(sockfd, F_GETFL);
    if ((sock_flags & O_NONBLOCK) == 0) {
        DEBUG_PRINT("socket was blocking\n");
        if (fcntl(sockfd, F_SETFL, sock_flags | O_NONBLOCK) < 0) {
            perror("fcntl non-blocking");
            exit(0);
        }
    }

    ret = connect(sockfd, (struct sockaddr *)sa, sizeof(*sa));
    if (ret < 0) {
        if (errno == EINPROGRESS) {
            DEBUG_PRINT("EINPROGRESS in connect() - selecting\n");

            tv.tv_sec = 5;
            tv.tv_usec = 0;
            FD_ZERO(&wset);
            FD_SET(sockfd, &wset);
            ret = select(sockfd + 1, NULL, &wset, NULL, &tv);
            if (ret > 0) {
                ret = 0;
                DEBUG_PRINT("connect!\n");
                goto connect_exit;
            }
        }
    }
    if (!ret) {
        DEBUG_PRINT("connection failed with errcode=%s...\n", strerror(errno));
    }

connect_exit:
    if (fcntl(sockfd, F_SETFL, sock_flags & (~O_NONBLOCK)) < 0) {
        perror("fcntl blocking");
        exit(0);
    }

    return ret;
}
#endif

int send_socket(int sockfd, void *buf, size_t size)
{
    ssize_t ret;
    while (size > 0) {
        //if ((ret = send(sockfd, buf, size, MSG_DONTWAIT)) <= 0) {
        if ((ret = send(sockfd, buf, size, 0)) <= 0) {
            perror("ERROR sending to socket");
            close(sockfd);
            abort();
        }
        size -= ret;
        buf += ret;
    }
    return ret;
}

int recv_socket(int sockfd, void *buf, size_t size)
{
    ssize_t ret;
    while (size > 0) {
        if ((ret = recv(sockfd, buf, size, 0)) <= 0) {
            perror("ERROR receiving from socket");
            close(sockfd);
            exit(0);
        }
        buf += ret;
        size -= ret;
    }
    return ret;
}
