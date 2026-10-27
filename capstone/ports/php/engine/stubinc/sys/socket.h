/* Minimal stub: PHP main/streams pulls the socket headers in through php.h. A domain has
 * no network stack, so nothing declared here is ever called -- the declarations exist so
 * php.h parses. See sys/stat.h for the same rationale. */
#ifndef _SYS_SOCKET_H
#define _SYS_SOCKET_H 1
#include <sys/types.h>
typedef unsigned int socklen_t;
typedef unsigned short sa_family_t;
struct sockaddr { sa_family_t sa_family; char sa_data[14]; };
struct sockaddr_storage { sa_family_t ss_family; char __ss_pad[126]; };
struct iovec { void *iov_base; size_t iov_len; };
struct msghdr {
    void *msg_name; socklen_t msg_namelen;
    struct iovec *msg_iov; size_t msg_iovlen;
    void *msg_control; size_t msg_controllen; int msg_flags;
};
#define AF_UNSPEC 0
#define AF_UNIX   1
#define AF_INET   2
#define AF_INET6  10
#define SOCK_STREAM 1
#define SOCK_DGRAM  2
#define SOL_SOCKET  1
#define SHUT_RD   0
#define SHUT_WR   1
#define SHUT_RDWR 2
int socket(int, int, int);
int connect(int, const struct sockaddr *, socklen_t);
int bind(int, const struct sockaddr *, socklen_t);
int listen(int, int);
int accept(int, struct sockaddr *, socklen_t *);
int shutdown(int, int);
int getsockname(int, struct sockaddr *, socklen_t *);
int getpeername(int, struct sockaddr *, socklen_t *);
int setsockopt(int, int, int, const void *, socklen_t);
int getsockopt(int, int, int, void *, socklen_t *);
long send(int, const void *, size_t, int);
long recv(int, void *, size_t, int);
long sendto(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
long recvfrom(int, void *, size_t, int, struct sockaddr *, socklen_t *);
#endif
