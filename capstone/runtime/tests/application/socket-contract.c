/* The socket contract of the delegated runtime: one mode per case of
 * docs/plans/delegation-sockets.md. Every mode exits 0 and prints
 * "socket-contract <mode>: PASS", or fails the CHECK that names the broken
 * promise. Linux is the oracle: the same program runs natively first.
 *
 * The image is built with a 16 KiB exchange region on purpose, so that the
 * datagram rule (a message the region cannot hold is EMSGSIZE, a stream send
 * is short) is exercised below the kernel's own limits. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <spawn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "socket-contract:%d: %s (errno %d)\n", __LINE__, #test, errno); return 1; \
} } while (0)

extern char **environ;
static const char *mode;

static int pass(void) { printf("socket-contract %s: PASS\n", mode); return 0; }

static long monotonic_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static socklen_t unix_address(struct sockaddr_un *un, const char *tag) {
  memset(un, 0, sizeof *un);
  un->sun_family = AF_UNIX;
  snprintf(un->sun_path, sizeof un->sun_path, "/tmp/socket-contract-%s-%ld", tag, (long)getpid());
  return (socklen_t)(offsetof(struct sockaddr_un, sun_path) + strlen(un->sun_path) + 1);
}

/* a TCP listener on the loopback, port chosen by the kernel */
static int listener(struct sockaddr_in *bound) {
  int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = 0, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
  socklen_t len = sizeof *bound;
  int one = 1;
  if (fd < 0 || setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) ||
      bind(fd, (struct sockaddr *)&in, sizeof in) || listen(fd, 4) ||
      getsockname(fd, (struct sockaddr *)bound, &len))
    return -1;
  return fd;
}

static int reap(pid_t pid, int *status) {
  pid_t r;
  do r = waitpid(pid, status, 0); while (r < 0 && errno == EINTR);
  return r == pid ? 0 : -1;
}

int main(int argc, char **argv) {
  int status = 0;
  if (argc < 2) return 2;
  mode = argv[1];
  if (!strcmp(mode, "unix-stream")) {
    /* socket, bind, listen, connect, accept4 with an address and a length
       word; names through getsockname and getpeername; bytes both ways;
       shutdown of the writing side is end of file for the peer */
    struct sockaddr_un un, name;
    socklen_t unlen = unix_address(&un, "stream"), namelen;
    int server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    CHECK(server >= 0);
    CHECK(!bind(server, (struct sockaddr *)&un, unlen));
    CHECK(!listen(server, 2));
    int client = socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(client >= 0 && !connect(client, (struct sockaddr *)&un, unlen));
    memset(&name, 0xee, sizeof name);
    namelen = sizeof name;
    int accepted = accept4(server, (struct sockaddr *)&name, &namelen, SOCK_CLOEXEC);
    CHECK(accepted >= 0);
    CHECK(namelen == sizeof(sa_family_t) && name.sun_family == AF_UNIX);   /* an unbound peer */
    CHECK((unsigned char)name.sun_path[0] == 0xee);                          /* untouched beyond */
    namelen = sizeof name;
    CHECK(!getsockname(accepted, (struct sockaddr *)&name, &namelen));
    CHECK(namelen == unlen && !strcmp(name.sun_path, un.sun_path));
    namelen = sizeof name;
    CHECK(!getpeername(client, (struct sockaddr *)&name, &namelen));
    CHECK(namelen == unlen && !strcmp(name.sun_path, un.sun_path));
    CHECK(send(accepted, "ping", 4, 0) == 4);
    char buf[16] = {0};
    CHECK(recv(client, buf, sizeof buf, 0) == 4 && !memcmp(buf, "ping", 4));
    CHECK(send(client, "pong!", 5, 0) == 5);
    CHECK(recv(accepted, buf, sizeof buf, 0) == 5 && !memcmp(buf, "pong!", 5));
    CHECK(!shutdown(accepted, SHUT_WR));
    CHECK(recv(client, buf, sizeof buf, 0) == 0);
    CHECK(!close(client) && !close(accepted) && !close(server));
    unlink(un.sun_path);
    return pass();
  }
  if (!strcmp(mode, "unix-dgram")) {
    /* two bound datagram sockets: sendto and recvfrom with the address, and a
       receive into a too-small address: the word says the full length, the
       name is cut, the data is intact */
    struct sockaddr_un a, b, from;
    socklen_t alen = unix_address(&a, "dgram-a"), blen = unix_address(&b, "dgram-b"), fromlen;
    int sa = socket(AF_UNIX, SOCK_DGRAM, 0), sb = socket(AF_UNIX, SOCK_DGRAM, 0);
    CHECK(sa >= 0 && sb >= 0);
    CHECK(!bind(sa, (struct sockaddr *)&a, alen) && !bind(sb, (struct sockaddr *)&b, blen));
    CHECK(sendto(sa, "datagram", 8, 0, (struct sockaddr *)&b, blen) == 8);
    char buf[32];
    fromlen = sizeof from;
    CHECK(recvfrom(sb, buf, sizeof buf, 0, (struct sockaddr *)&from, &fromlen) == 8);
    CHECK(!memcmp(buf, "datagram", 8) && fromlen == alen && !strcmp(from.sun_path, a.sun_path));
    CHECK(sendto(sa, "datagram", 8, 0, (struct sockaddr *)&b, blen) == 8);
    memset(&from, 0xee, sizeof from);
    fromlen = 4;
    CHECK(recvfrom(sb, buf, sizeof buf, 0, (struct sockaddr *)&from, &fromlen) == 8);
    CHECK(fromlen == alen && from.sun_family == AF_UNIX && from.sun_path[0] == '/' &&
          (unsigned char)from.sun_path[2] == 0xee && !memcmp(buf, "datagram", 8));
    CHECK(!close(sa) && !close(sb));
    unlink(a.sun_path); unlink(b.sun_path);
    return pass();
  }
  if (!strcmp(mode, "inet-stream")) {
    /* TCP on the loopback: the port learned through getsockname, an option
       read back through a word, the type, and a refused non-blocking
       connect reported through SO_ERROR after ppoll */
    struct sockaddr_in bound;
    int server = listener(&bound);
    CHECK(server >= 0 && bound.sin_port != 0);
    int value = 0;
    socklen_t vlen = sizeof value;
    CHECK(!getsockopt(server, SOL_SOCKET, SO_REUSEADDR, &value, &vlen) && vlen == 4 && value == 1);
    CHECK(!getsockopt(server, SOL_SOCKET, SO_TYPE, &value, &vlen) && value == SOCK_STREAM);
    int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0 && !connect(client, (struct sockaddr *)&bound, sizeof bound));
    struct sockaddr_in peer;
    socklen_t plen = sizeof peer;
    int accepted = accept(server, (struct sockaddr *)&peer, &plen);
    CHECK(accepted >= 0 && plen == sizeof peer && peer.sin_addr.s_addr == htonl(INADDR_LOOPBACK));
    int nodelay = 1;
    CHECK(!setsockopt(accepted, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof nodelay));
    CHECK(send(accepted, "tcp", 3, 0) == 3);
    char buf[8];
    CHECK(recv(client, buf, sizeof buf, 0) == 3 && !memcmp(buf, "tcp", 3));
    CHECK(!close(accepted) && !close(client) && !close(server));
    /* the port is free now: a non-blocking connect to it is refused */
    int probe = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    CHECK(probe >= 0);
    int r = connect(probe, (struct sockaddr *)&bound, sizeof bound);
    CHECK(r == 0 || errno == EINPROGRESS || errno == ECONNREFUSED);
    if (r < 0 && errno == EINPROGRESS) {
      struct pollfd p = {.fd = probe, .events = POLLOUT};
      CHECK(ppoll(&p, 1, &(struct timespec){2, 0}, NULL) == 1);
      value = 0; vlen = sizeof value;
      CHECK(!getsockopt(probe, SOL_SOCKET, SO_ERROR, &value, &vlen) && value == ECONNREFUSED);
    }
    CHECK(!close(probe));
    return pass();
  }
  if (!strcmp(mode, "inet-dgram")) {
    /* UDP on the loopback. The image's exchange region is 16 KiB: a datagram
       that fits crosses whole, one the region cannot hold is EMSGSIZE from
       the libc, as the kernel answers one too long for the protocol; a
       receive into a buffer shorter than the datagram is MSG_TRUNC through
       recvmsg, the kernel's own report. A stream send of the same size is
       short and completes in pieces. */
    struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = 0, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)}, to;
    socklen_t tolen = sizeof to;
    int a = socket(AF_INET, SOCK_DGRAM, 0), b = socket(AF_INET, SOCK_DGRAM, 0);
    CHECK(a >= 0 && b >= 0);
    CHECK(!bind(a, (struct sockaddr *)&in, sizeof in) && !bind(b, (struct sockaddr *)&in, sizeof in));
    CHECK(!getsockname(b, (struct sockaddr *)&to, &tolen));
    static char big[70000], back[70000];
    for (size_t i = 0; i < sizeof big; ++i) big[i] = (char)(i * 7);
    /* the largest datagram that crosses, found from above: printed, not assumed */
    size_t fits = 0;
    for (size_t n = 16000; n >= 1000; n -= 1000) {
      ssize_t s = sendto(a, big, n, 0, (struct sockaddr *)&to, tolen);
      if (s == (ssize_t)n) { fits = n; break; }
      if (!(s < 0 && errno == EMSGSIZE))
        fprintf(stderr, "socket-contract inet-dgram: sendto(%zu) = %zd, errno %d\n", n, s, errno);
      CHECK(s < 0 && errno == EMSGSIZE);
    }
    CHECK(fits >= 8000);
    printf("socket-contract inet-dgram: %zu-byte datagram crosses\n", fits);
    CHECK(recv(b, back, sizeof back, 0) == (ssize_t)fits && !memcmp(back, big, fits));
    /* 40000 bytes: Linux takes it; a region that cannot hold it answers
       EMSGSIZE, the deviation the plan names, printed for the record and
       expected by the guest runner, never by Linux */
    errno = 0;
    ssize_t forty = sendto(a, big, 40000, 0, (struct sockaddr *)&to, tolen);
    CHECK(forty == 40000 || (forty < 0 && errno == EMSGSIZE));
    printf("socket-contract inet-dgram: 40000-byte datagram: %s\n", forty == 40000 ? "crosses" : "EMSGSIZE");
    if (forty == 40000) CHECK(recv(b, back, sizeof back, 0) == 40000 && !memcmp(back, big, 40000));
    errno = 0;
    CHECK(sendto(a, big, sizeof big, 0, (struct sockaddr *)&to, tolen) < 0 && errno == EMSGSIZE);
    CHECK(sendto(a, big, 4000, 0, (struct sockaddr *)&to, tolen) == 4000);
    struct iovec iov = {.iov_base = back, .iov_len = 100};
    struct msghdr m = {.msg_iov = &iov, .msg_iovlen = 1};
    CHECK(recvmsg(b, &m, 0) == 100 && (m.msg_flags & MSG_TRUNC) && !memcmp(back, big, 100));
    /* the same bytes over a stream: short sends, all of it arrives */
    struct sockaddr_in bound;
    int server = listener(&bound);
    CHECK(server >= 0);
    int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0 && !connect(client, (struct sockaddr *)&bound, sizeof bound));
    int accepted = accept(server, NULL, NULL);
    CHECK(accepted >= 0);
    size_t sent = 0;
    int pieces = 0;
    while (sent < 40000) {
      ssize_t s = send(client, big + sent, 40000 - sent, 0);
      CHECK(s > 0);
      sent += (size_t)s;
      ++pieces;
    }
    size_t got = 0;
    while (got < 40000) {
      ssize_t r = recv(accepted, back + got, sizeof back - got, 0);
      CHECK(r > 0);
      got += (size_t)r;
    }
    CHECK(!memcmp(back, big, 40000));
    printf("socket-contract inet-dgram: 40000 stream bytes in %d sends\n", pieces);
    CHECK(!close(a) && !close(b) && !close(client) && !close(accepted) && !close(server));
    return pass();
  }
  if (!strcmp(mode, "scm-rights")) {
    /* a sequenced-packet pair; sendmsg with two iovecs and a pipe end in
       SCM_RIGHTS; recvmsg receives the bytes across two buffers and a
       descriptor that writes into the pipe, close-on-exec as asked; a control
       buffer too small is MSG_CTRUNC */
    int pair[2], pipefd[2];
    CHECK(!socketpair(AF_UNIX, SOCK_SEQPACKET, 0, pair) && !pipe(pipefd));
    char control[CMSG_SPACE(sizeof(int))];
    struct iovec out[2] = {{"abc", 3}, {"def", 3}};
    struct msghdr m = {.msg_iov = out, .msg_iovlen = 2, .msg_control = control, .msg_controllen = sizeof control};
    memset(control, 0, sizeof control);
    struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    CHECK(sendmsg(pair[0], &m, 0) == 6);
    char one[4] = {0}, two[4] = {0}, rcontrol[CMSG_SPACE(sizeof(int))];
    struct iovec in[2] = {{one, 4}, {two, 4}};
    struct msghdr r = {.msg_iov = in, .msg_iovlen = 2, .msg_control = rcontrol, .msg_controllen = sizeof rcontrol};
    CHECK(recvmsg(pair[1], &r, MSG_CMSG_CLOEXEC) == 6);
    CHECK(!memcmp(one, "abcd", 4) && !memcmp(two, "ef", 2) && !(r.msg_flags & (MSG_TRUNC | MSG_CTRUNC)));
    cm = CMSG_FIRSTHDR(&r);
    CHECK(cm && cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS);
    int received;
    memcpy(&received, CMSG_DATA(cm), sizeof received);
    CHECK(received >= 0 && received != pipefd[1]);
    CHECK(fcntl(received, F_GETFD) & FD_CLOEXEC);
    CHECK(write(received, "x", 1) == 1);
    char x = 0;
    CHECK(read(pipefd[0], &x, 1) == 1 && x == 'x');
    CHECK(!close(received));
    /* too small a control buffer: the descriptor is dropped, the flag says so */
    memset(control, 0, sizeof control);
    m.msg_controllen = sizeof control;
    cm = CMSG_FIRSTHDR(&m);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    CHECK(sendmsg(pair[0], &m, 0) == 6);
    r.msg_controllen = 8;
    r.msg_flags = 0;
    CHECK(recvmsg(pair[1], &r, 0) == 6 && (r.msg_flags & MSG_CTRUNC));
    CHECK(!close(pair[0]) && !close(pair[1]) && !close(pipefd[0]) && !close(pipefd[1]));
    return pass();
  }
  if (!strcmp(mode, "epoll")) {
    /* create, add a socket, a wait with timeout 0 is empty, after a write it
       reports the socket with the data word intact; modify; delete with a
       null event */
    int pair[2];
    CHECK(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair));
    int ep = epoll_create1(EPOLL_CLOEXEC);
    CHECK(ep >= 0);
    struct epoll_event ev = {.events = EPOLLIN, .data.u64 = UINT64_C(0x1234567890abcdef)}, got[4];
    CHECK(!epoll_ctl(ep, EPOLL_CTL_ADD, pair[0], &ev));
    CHECK(epoll_wait(ep, got, 4, 0) == 0);
    CHECK(send(pair[1], "z", 1, 0) == 1);
    CHECK(epoll_wait(ep, got, 4, 1000) == 1);
    CHECK((got[0].events & EPOLLIN) && got[0].data.u64 == UINT64_C(0x1234567890abcdef));
    ev.events = EPOLLOUT;
    CHECK(!epoll_ctl(ep, EPOLL_CTL_MOD, pair[0], &ev));
    CHECK(epoll_wait(ep, got, 4, 0) == 1 && (got[0].events & EPOLLOUT));
    CHECK(!epoll_ctl(ep, EPOLL_CTL_DEL, pair[0], NULL));
    CHECK(epoll_wait(ep, got, 4, 0) == 0);
    CHECK(!close(ep) && !close(pair[0]) && !close(pair[1]));
    return pass();
  }
  if (!strcmp(mode, "select-poll")) {
    /* pselect and ppoll report a readable socket, and nothing before */
    int pair[2];
    CHECK(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair));
    fd_set readable;
    FD_ZERO(&readable); FD_SET(pair[0], &readable);
    struct timespec zero = {0, 0};
    CHECK(pselect(pair[0] + 1, &readable, NULL, NULL, &zero, NULL) == 0);
    struct pollfd p = {.fd = pair[0], .events = POLLIN};
    CHECK(ppoll(&p, 1, &zero, NULL) == 0);
    CHECK(send(pair[1], "r", 1, 0) == 1);
    FD_ZERO(&readable); FD_SET(pair[0], &readable);
    CHECK(pselect(pair[0] + 1, &readable, NULL, NULL, &zero, NULL) == 1 && FD_ISSET(pair[0], &readable));
    CHECK(ppoll(&p, 1, &zero, NULL) == 1 && (p.revents & POLLIN));
    CHECK(!close(pair[0]) && !close(pair[1]));
    return pass();
  }
  if (!strcmp(mode, "nonblock")) {
    /* a non-blocking accept is EAGAIN; FIONBIO and FIONREAD; a receive
       timeout makes recv return EAGAIN after at least that long */
    struct sockaddr_in bound;
    int server = listener(&bound);
    CHECK(server >= 0);
    int on = 1;
    CHECK(!ioctl(server, FIONBIO, &on));
    errno = 0;
    CHECK(accept4(server, NULL, NULL, 0) < 0 && (errno == EAGAIN || errno == EWOULDBLOCK));
    int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0 && !connect(client, (struct sockaddr *)&bound, sizeof bound));
    int accepted = -1;
    for (int i = 0; i < 100 && accepted < 0; ++i) {
      accepted = accept4(server, NULL, NULL, 0);
      if (accepted < 0) { CHECK(errno == EAGAIN); usleep(10000); }
    }
    CHECK(accepted >= 0);
    CHECK(send(client, "12345", 5, 0) == 5);
    int pending = -1;
    for (int i = 0; i < 100 && pending < 5; ++i) { CHECK(!ioctl(accepted, FIONREAD, &pending)); usleep(10000); }
    CHECK(pending == 5);
    char buf[8];
    CHECK(recv(accepted, buf, sizeof buf, 0) == 5);
    struct timeval tv = {.tv_sec = 0, .tv_usec = 100000};
    CHECK(!setsockopt(accepted, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv));
    long t0 = monotonic_ms();
    errno = 0;
    CHECK(recv(accepted, buf, sizeof buf, 0) < 0 && (errno == EAGAIN || errno == EWOULDBLOCK));
    CHECK(monotonic_ms() - t0 >= 90);
    CHECK(!close(accepted) && !close(client) && !close(server));
    return pass();
  }
  if (!strcmp(mode, "inherit")) {
    /* the listening socket on descriptor 3 of a child running this image:
       the child accepts, the parent connects, one byte crosses */
    struct sockaddr_in bound;
    int server = listener(&bound);
    CHECK(server >= 0);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, server, 3);
    char *args[] = {argv[0], "inherit-child", NULL};
    pid_t p;
    CHECK(!posix_spawn(&p, argv[0], &actions, NULL, args, environ));
    posix_spawn_file_actions_destroy(&actions);
    int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0 && !connect(client, (struct sockaddr *)&bound, sizeof bound));
    char c = 0;
    CHECK(recv(client, &c, 1, 0) == 1 && c == 'k');
    CHECK(!reap(p, &status) && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(!close(client) && !close(server));
    return pass();
  }
  if (!strcmp(mode, "inherit-child")) {
    int accepted = accept4(3, NULL, NULL, 0);
    if (accepted < 0 || send(accepted, "k", 1, 0) != 1) return 3;
    close(accepted);
    return 0;
  }
  if (!strcmp(mode, "hosts")) {
    /* name resolution without a network: localhost through /etc/hosts, a
       numeric host, and a numeric name back */
    struct addrinfo hints = {.ai_family = AF_INET, .ai_socktype = SOCK_STREAM}, *res = NULL;
    CHECK(getaddrinfo("localhost", NULL, &hints, &res) == 0 && res);
    CHECK(((struct sockaddr_in *)res->ai_addr)->sin_addr.s_addr == htonl(INADDR_LOOPBACK));
    freeaddrinfo(res);
    hints.ai_flags = AI_NUMERICHOST;
    CHECK(getaddrinfo("127.0.0.1", "80", &hints, &res) == 0 && res);
    CHECK(((struct sockaddr_in *)res->ai_addr)->sin_port == htons(80));
    char host[64], service[16];
    CHECK(getnameinfo(res->ai_addr, res->ai_addrlen, host, sizeof host, service, sizeof service,
                      NI_NUMERICHOST | NI_NUMERICSERV) == 0);
    CHECK(!strcmp(host, "127.0.0.1") && !strcmp(service, "80"));
    freeaddrinfo(res);
    return pass();
  }
  if (!strcmp(mode, "tagged-buffer")) {
    /* A buffer that held a pointer before the call: an uninitialized address
       struct on a stack that held one, pymalloc's free list in CPython. The
       exchange region carries data, never capabilities; the bytes the kernel
       wrote must come back, not the pointer that was there. Found through
       CPython's recv_fds and inet-dgram on capstone-qemu, 2026-09-30. */
    union { struct sockaddr_in addr; void *pointer[2]; } holder;
    holder.pointer[0] = &holder;
    holder.pointer[1] = &mode;
    socklen_t len = sizeof holder.addr;
    struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = 0, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    int a = socket(AF_INET, SOCK_DGRAM, 0);
    CHECK(a >= 0 && !bind(a, (struct sockaddr *)&in, sizeof in));
    CHECK(!getsockname(a, (struct sockaddr *)&holder.addr, &len));
    CHECK(len == sizeof in && holder.addr.sin_family == AF_INET && holder.addr.sin_addr.s_addr == htonl(INADDR_LOOPBACK));
    CHECK(sendto(a, "x", 1, 0, (struct sockaddr *)&holder.addr, len) == 1);
    /* the same for a plain read: a pipe's bytes into a buffer that held pointers */
    union { char bytes[32]; void *pointer[2]; } buffer;
    buffer.pointer[0] = &buffer;
    buffer.pointer[1] = &in;
    int p[2];
    CHECK(!pipe(p) && write(p[1], "0123456789abcdefghijklmnopqrstuv", 32) == 32);
    CHECK(read(p[0], buffer.bytes, 32) == 32 && !memcmp(buffer.bytes, "0123456789abcdefghijklmnopqrstuv", 32));
    /* and a control message into such a buffer */
    int pair[2], pipefd[2];
    CHECK(!socketpair(AF_UNIX, SOCK_STREAM, 0, pair) && !pipe(pipefd));
    union { char bytes[CMSG_SPACE(sizeof(int))]; void *pointer[2]; } control;
    control.pointer[0] = &control; control.pointer[1] = &buffer;
    struct iovec out = {"fd", 2};
    char scontrol[CMSG_SPACE(sizeof(int))];
    struct msghdr m = {.msg_iov = &out, .msg_iovlen = 1, .msg_control = scontrol, .msg_controllen = sizeof scontrol};
    memset(scontrol, 0, sizeof scontrol);
    struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    CHECK(sendmsg(pair[0], &m, 0) == 2);
    char data[4];
    struct iovec inv = {data, sizeof data};
    struct msghdr r = {.msg_iov = &inv, .msg_iovlen = 1, .msg_control = control.bytes, .msg_controllen = CMSG_LEN(sizeof(int))};
    CHECK(recvmsg(pair[1], &r, 0) == 2);
    cm = CMSG_FIRSTHDR(&r);
    CHECK(cm && cm->cmsg_len == CMSG_LEN(sizeof(int)) && cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS);
    return pass();
  }
  fprintf(stderr, "socket-contract: unknown mode %s\n", mode);
  return 2;
}
