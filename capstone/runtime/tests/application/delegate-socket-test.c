/* Native cases for the socket rows through the launcher's dispatcher: Unix
 * and loopback sockets, address and option lengths given as words, the
 * msghdr block with SCM_RIGHTS, epoll, and the launcher's private descriptors
 * refused in every position, the control message included. */
#define _GNU_SOURCE
#include "../../linux/delegate-service.h"
#include "capstone/msghdr.h"
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static char exchange[32768];
static struct capstone_delegate_host host = {.exchange = exchange, .exchange_bytes = sizeof exchange};

static long call6(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
  struct capstone_delegate_entry entry;
  uint64_t args[6] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&entry, nr, args));
  capstone_delegate_serve(&host, &entry);
  return entry.result;
}
static long call(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return call6(nr, a, b, c, d, 0, 0);
}
static void put(uint64_t offset, const void *data, size_t bytes) { memcpy(exchange + offset, data, bytes); }
static uint32_t word_at(uint64_t offset) { uint32_t w; memcpy(&w, exchange + offset, 4); return w; }
static void set_word(uint64_t offset, uint32_t w) { memcpy(exchange + offset, &w, 4); }

/* a Unix address for a path in /tmp, unique to this process */
static socklen_t unix_address(struct sockaddr_un *un, const char *tag) {
  memset(un, 0, sizeof *un);
  un->sun_family = AF_UNIX;
  snprintf(un->sun_path, sizeof un->sun_path, "/tmp/capstone-socket-%s-%ld", tag, (long)getpid());
  return (socklen_t)(offsetof(struct sockaddr_un, sun_path) + strlen(un->sun_path) + 1);
}

/* the block at 64, its pairs at 128, buffers from 1024 */
static void block(struct capstone_msghdr_block *b, uint64_t name, uint64_t namelen,
                  const uint64_t *pairs, uint64_t count, uint64_t control, uint64_t controllen) {
  memset(b, 0, sizeof *b);
  b->name = name; b->namelen = namelen;
  b->iov = count ? 128 : 0; b->iovlen = count;
  b->control = control; b->controllen = controllen;
  put(64, b, sizeof *b);
  if (count) put(128, pairs, count * 16);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *c = argv[1];
  struct sockaddr_un un, peer;
  socklen_t unlen;
  if (!strcmp(c, "unix-stream")) {
    /* socket, bind, listen and accept4 with an address word; the peer's name
       through getpeername; bytes both ways through sendto and recvfrom with
       no address; shutdown seen as end of file by the peer */
    long server = call(CAPSTONE_SYS_socket, AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, 0);
    assert(server >= 0);
    unlen = unix_address(&un, "stream");
    put(256, &un, unlen);
    assert(call(CAPSTONE_SYS_bind, (uint64_t)server, 256, unlen, 0) == 0);
    assert(call(CAPSTONE_SYS_listen, (uint64_t)server, 4, 0, 0) == 0);
    int client = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(client >= 0 && connect(client, (struct sockaddr *)&un, unlen) == 0);
    set_word(512, sizeof peer);
    memset(exchange + 1024, 0xee, sizeof peer);
    long accepted = call(CAPSTONE_SYS_accept4, (uint64_t)server, 1024, 512, SOCK_CLOEXEC);
    assert(accepted >= 0);
    /* an unbound Unix peer: the kernel reports the family only */
    assert(word_at(512) == sizeof(sa_family_t));
    memcpy(&peer, exchange + 1024, sizeof peer);
    assert(peer.sun_family == AF_UNIX);
    assert((unsigned char)exchange[1024 + 2] == 0xee);   /* beyond what the kernel wrote: ours */
    /* getsockname on the accepted descriptor: the bound path, and the word grows to it */
    set_word(512, sizeof peer);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)accepted, 1024, 512, 0) == 0);
    assert(word_at(512) == unlen);
    memcpy(&peer, exchange + 1024, unlen);
    assert(!strcmp(peer.sun_path, un.sun_path));
    /* the peer's view of the same address */
    struct sockaddr_un mine; socklen_t minelen = sizeof mine;
    assert(getpeername(client, (struct sockaddr *)&mine, &minelen) == 0 && minelen == unlen);
    /* data: sendto without an address, recvfrom without one */
    put(2048, "ping", 4);
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)accepted, 2048, 4, 0, 0, 0) == 4);
    char got[8] = {0};
    assert(read(client, got, sizeof got) == 4 && !memcmp(got, "ping", 4));
    assert(write(client, "pong!", 5) == 5);
    memset(exchange + 3072, 0x11, 16);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)accepted, 3072, 16, 0, 0, 0) == 5);
    assert(!memcmp(exchange + 3072, "pong!", 5) && (unsigned char)exchange[3072 + 5] == 0x11);
    assert(call(CAPSTONE_SYS_shutdown, (uint64_t)accepted, SHUT_WR, 0, 0) == 0);
    assert(read(client, got, sizeof got) == 0);
    close(client); close((int)accepted); close((int)server);
    unlink(un.sun_path);
  } else if (!strcmp(c, "word-truncated")) {
    /* a word smaller than the address: the kernel writes that many bytes and
       reports the full length; the rest of the buffer stays as it was */
    long server = call(CAPSTONE_SYS_socket, AF_UNIX, SOCK_STREAM, 0, 0);
    assert(server >= 0);
    unlen = unix_address(&un, "trunc");
    put(256, &un, unlen);
    assert(call(CAPSTONE_SYS_bind, (uint64_t)server, 256, unlen, 0) == 0);
    set_word(512, 4);
    memset(exchange + 1024, 0xee, 32);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)server, 1024, 512, 0) == 0);
    assert(word_at(512) == unlen);
    memcpy(&peer, exchange + 1024, 4);
    assert(peer.sun_family == AF_UNIX && peer.sun_path[0] == '/' && peer.sun_path[1] == 't');
    assert((unsigned char)exchange[1024 + 4] == 0xee);
    /* getsockname's word is not optional: a null word never reaches the wire,
       the libc answers EFAULT itself; recvfrom's is, and the kernel answers
       (the dgram case). A null buffer with a word: accept4 takes it; here
       nothing to accept */
    set_word(512, 16);
    assert(call(CAPSTONE_SYS_listen, (uint64_t)server, 1, 0, 0) == 0);
    assert(fcntl((int)server, F_SETFL, O_NONBLOCK) == 0);
    assert(call(CAPSTONE_SYS_accept4, (uint64_t)server, 0, 512, 0) == -EAGAIN);
    /* a word the region cannot hold is refused before the kernel sees it */
    set_word(512, (uint32_t)sizeof exchange);
    uint64_t before = host.refused;
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)server, 1024, 512, 0) == -EFAULT);
    assert(host.refused == before + 1);
    close((int)server);
    unlink(un.sun_path);
  } else if (!strcmp(c, "inet")) {
    /* TCP on the loopback: a reusable listener on port 0, the port learned
       through getsockname, an option read back through a word, a receive
       timeout as a 16-byte timeval, and a refused non-blocking connect
       reported through SO_ERROR */
    long server = call(CAPSTONE_SYS_socket, AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0, 0);
    assert(server >= 0);
    int one = 1;
    put(256, &one, sizeof one);
    assert(call6(CAPSTONE_SYS_setsockopt, (uint64_t)server, SOL_SOCKET, SO_REUSEADDR, 256, sizeof one, 0) == 0);
    struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = 0, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    put(256, &in, sizeof in);
    assert(call(CAPSTONE_SYS_bind, (uint64_t)server, 256, sizeof in, 0) == 0);
    set_word(512, sizeof in);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)server, 1024, 512, 0) == 0);
    assert(word_at(512) == sizeof in);
    memcpy(&in, exchange + 1024, sizeof in);
    assert(in.sin_port != 0);
    assert(call(CAPSTONE_SYS_listen, (uint64_t)server, 1, 0, 0) == 0);
    set_word(512, 4);
    memset(exchange + 1024, 0, 4);
    assert(call6(CAPSTONE_SYS_getsockopt, (uint64_t)server, SOL_SOCKET, SO_REUSEADDR, 1024, 512, 0) == 0);
    assert(word_at(512) == 4 && word_at(1024) == 1);
    set_word(512, 4);
    assert(call6(CAPSTONE_SYS_getsockopt, (uint64_t)server, SOL_SOCKET, SO_TYPE, 1024, 512, 0) == 0);
    assert(word_at(1024) == SOCK_STREAM);
    int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0 && connect(client, (struct sockaddr *)&in, sizeof in) == 0);
    set_word(512, sizeof in);
    long accepted = call(CAPSTONE_SYS_accept, (uint64_t)server, 1024, 512, 0);
    assert(accepted >= 0 && word_at(512) == sizeof in);
    struct timeval tv = {.tv_sec = 0, .tv_usec = 100000};
    put(256, &tv, sizeof tv);
    assert(call6(CAPSTONE_SYS_setsockopt, (uint64_t)accepted, SOL_SOCKET, SO_RCVTIMEO, 256, sizeof tv, 0) == 0);
    set_word(512, sizeof tv);
    memset(exchange + 1024, 0, sizeof tv);
    assert(call6(CAPSTONE_SYS_getsockopt, (uint64_t)accepted, SOL_SOCKET, SO_RCVTIMEO, 1024, 512, 0) == 0);
    memcpy(&tv, exchange + 1024, sizeof tv);
    assert(word_at(512) == sizeof tv && tv.tv_sec == 0 && tv.tv_usec >= 90000 && tv.tv_usec <= 110000);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)accepted, 2048, 8, 0, 0, 0) == -EAGAIN);   /* timed out */
    /* a non-blocking connect to a port nobody listens on */
    close((int)accepted); close(client);
    long probe = call(CAPSTONE_SYS_socket, AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0, 0);
    assert(probe >= 0);
    close((int)server);   /* the port is free now */
    put(256, &in, sizeof in);
    long r = call(CAPSTONE_SYS_connect, (uint64_t)probe, 256, sizeof in, 0);
    assert(r == 0 || r == -EINPROGRESS || r == -ECONNREFUSED);
    if (r == -EINPROGRESS) {
      struct pollfd p = {.fd = (int)probe, .events = POLLOUT};
      assert(poll(&p, 1, 2000) == 1);
      set_word(512, 4);
      assert(call6(CAPSTONE_SYS_getsockopt, (uint64_t)probe, SOL_SOCKET, SO_ERROR, 1024, 512, 0) == 0);
      assert(word_at(1024) == ECONNREFUSED);
    }
    close((int)probe);
  } else if (!strcmp(c, "dgram")) {
    /* UDP on the loopback: sendto with an address, recvfrom with the source
       address through a word, a short word cuts the name and reports the full
       length; only the bytes received are copied back */
    long a = call(CAPSTONE_SYS_socket, AF_INET, SOCK_DGRAM, 0, 0);
    long b = call(CAPSTONE_SYS_socket, AF_INET, SOCK_DGRAM, 0, 0);
    assert(a >= 0 && b >= 0);
    struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = 0, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    put(256, &in, sizeof in);
    assert(call(CAPSTONE_SYS_bind, (uint64_t)a, 256, sizeof in, 0) == 0);
    assert(call(CAPSTONE_SYS_bind, (uint64_t)b, 256, sizeof in, 0) == 0);
    set_word(512, sizeof in);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)b, 1024, 512, 0) == 0);
    struct sockaddr_in to, from;
    memcpy(&to, exchange + 1024, sizeof to);
    put(2048, "datagram", 8);
    put(256, &to, sizeof to);
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)a, 2048, 8, 0, 256, sizeof to) == 8);
    set_word(512, sizeof from);
    memset(exchange + 3072, 0x22, 32);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)b, 3072, 32, 0, 1024, 512) == 8);
    assert(!memcmp(exchange + 3072, "datagram", 8) && (unsigned char)exchange[3072 + 8] == 0x22);
    assert(word_at(512) == sizeof from);
    memcpy(&from, exchange + 1024, sizeof from);
    set_word(768, sizeof in);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)a, 1536, 768, 0) == 0);
    assert(!memcmp(&from, exchange + 1536, sizeof from));   /* the sender's own address */
    /* a two-byte word: the family only, and the length the kernel would need */
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)a, 2048, 8, 0, 256, sizeof to) == 8);
    set_word(512, 2);
    memset(exchange + 1024, 0xee, 16);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)b, 3072, 32, 0, 1024, 512) == 8);
    assert(word_at(512) == sizeof from && (unsigned char)exchange[1024 + 2] == 0xee);
    /* a large datagram with its address high in the region, as the libc
       places them: the family must survive the distance */
    {
      static char big[16000];
      memset(big, 7, sizeof big);
      put(4096, big, sizeof big);
      put(4096 + 16000 + 16, &to, sizeof to);
      assert(call6(CAPSTONE_SYS_sendto, (uint64_t)a, 4096, 16000, 0, 4096 + 16000 + 16, sizeof to) == 16000);
      set_word(512, sizeof from);
      assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)b, 4096, 16000, 0, 1024, 512) == 16000);
      assert(!memcmp(exchange + 4096, big, 16000));
    }
    /* an address buffer with a null word: the kernel's own answer, EFAULT,
       given after it took the datagram, as Linux does; the socket goes on */
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)a, 2048, 8, 0, 256, sizeof to) == 8);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)b, 3072, 32, 0, 1024, 0) == -EFAULT);
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)a, 2048, 8, 0, 256, sizeof to) == 8);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)b, 3072, 32, 0, 0, 0) == 8);
    close((int)a); close((int)b);
  } else if (!strcmp(c, "socketpair-msg")) {
    /* a sequenced-packet pair from the row; sendmsg with two iovecs and a
       pipe end in SCM_RIGHTS; recvmsg receives the bytes across two buffers
       and a descriptor that writes into the pipe; the lengths and flags come
       back in the block; a control buffer too small is MSG_CTRUNC */
    assert(call(CAPSTONE_SYS_socketpair, AF_UNIX, SOCK_SEQPACKET, 0, 256) == 0);
    int pair[2];
    memcpy(pair, exchange + 256, sizeof pair);
    int pipefd[2];
    assert(!pipe(pipefd));
    struct capstone_msghdr_block b;
    uint64_t pairs[4] = {1024, 3, 1040, 3};
    put(1024, "abc", 3); put(1040, "def", 3);
    char control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof control);
    struct cmsghdr *cm = (struct cmsghdr *)control;
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    put(2048, control, sizeof control);
    block(&b, 0, 0, pairs, 2, 2048, sizeof control);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == 6);
    /* receive: two buffers of four, a control buffer, the peer's name asked for */
    uint64_t rpairs[4] = {3072, 4, 3088, 4};
    memset(exchange + 3072, 0x33, 32);
    memset(exchange + 2560, 0, 64);
    block(&b, 4096, 32, rpairs, 2, 2560, 64);
    assert(call(CAPSTONE_SYS_recvmsg, (uint64_t)pair[1], 64, MSG_CMSG_CLOEXEC, 0) == 6);
    assert(!memcmp(exchange + 3072, "abcd", 4) && !memcmp(exchange + 3088, "ef", 2) &&
           (unsigned char)exchange[3088 + 2] == 0x33);
    memcpy(&b, exchange + 64, sizeof b);
    assert(b.controllen == CMSG_SPACE(sizeof(int)) || b.controllen == CMSG_LEN(sizeof(int)));
    assert(b.namelen == 0 || b.namelen == sizeof(sa_family_t));   /* an unnamed peer */
    assert((b.flags & MSG_CTRUNC) == 0);
    memcpy(control, exchange + 2560, sizeof control);
    cm = (struct cmsghdr *)control;
    assert(cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS);
    int received;
    memcpy(&received, CMSG_DATA(cm), sizeof received);
    assert(received >= 0 && received != pipefd[1]);
    assert(fcntl(received, F_GETFD) & FD_CLOEXEC);
    assert(write(received, "x", 1) == 1);
    char x = 0;
    assert(read(pipefd[0], &x, 1) == 1 && x == 'x');
    close(received);
    /* a control buffer too small for the descriptor: truncated */
    memset(control, 0, sizeof control);
    cm = (struct cmsghdr *)control;
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    put(2048, control, sizeof control);
    block(&b, 0, 0, pairs, 2, 2048, sizeof control);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == 6);
    block(&b, 0, 0, rpairs, 2, 2560, 8);
    assert(call(CAPSTONE_SYS_recvmsg, (uint64_t)pair[1], 64, 0, 0) == 6);
    memcpy(&b, exchange + 64, sizeof b);
    assert(b.flags & MSG_CTRUNC);
    /* a control buffer of exactly CMSG_LEN(int), as CPython's recv_fds
       passes it: the header's level and type survive, the descriptor too */
    memset(control, 0, sizeof control);
    cm = (struct cmsghdr *)control;
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &pipefd[1], sizeof(int));
    put(2048, control, sizeof control);
    block(&b, 0, 0, pairs, 2, 2048, sizeof control);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == 6);
    memset(exchange + 2560, 0xee, 64);
    block(&b, 0, 0, rpairs, 2, 2560, CMSG_LEN(sizeof(int)));
    assert(call(CAPSTONE_SYS_recvmsg, (uint64_t)pair[1], 64, 0, 0) == 6);
    memcpy(&b, exchange + 64, sizeof b);
    assert(b.controllen == CMSG_LEN(sizeof(int)) && !(b.flags & MSG_CTRUNC));
    memcpy(control, exchange + 2560, CMSG_LEN(sizeof(int)));
    cm = (struct cmsghdr *)control;
    assert(cm->cmsg_len == CMSG_LEN(sizeof(int)) && cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS);
    memcpy(&received, CMSG_DATA(cm), sizeof received);
    assert(received >= 0 && write(received, "y", 1) == 1 && read(pipefd[0], &x, 1) == 1 && x == 'y');
    assert((unsigned char)exchange[2560 + CMSG_LEN(sizeof(int))] == 0xee);   /* beyond the control: ours */
    close(received);
    /* a message with no buffers at all is legal and empty */
    block(&b, 0, 0, NULL, 0, 0, 0);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == 0);
    close(pair[0]); close(pair[1]); close(pipefd[0]); close(pipefd[1]);
  } else if (!strcmp(c, "private")) {
    /* the launcher's own descriptor in every position, and inside SCM_RIGHTS */
    int mine = open("/dev/null", O_RDONLY);
    assert(mine >= 0);
    host.private_fds[host.private_count++] = mine;
    assert(call(CAPSTONE_SYS_socketpair, AF_UNIX, SOCK_DGRAM, 0, 256) == 0);
    int pair[2];
    memcpy(pair, exchange + 256, sizeof pair);
    assert(call(CAPSTONE_SYS_listen, (uint64_t)mine, 1, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_shutdown, (uint64_t)mine, SHUT_RDWR, 0, 0) == -EBADF);
    set_word(512, 16);
    assert(call(CAPSTONE_SYS_getsockname, (uint64_t)mine, 1024, 512, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_accept4, (uint64_t)mine, 1024, 512, 0) == -EBADF);
    put(2048, "x", 1);
    assert(call6(CAPSTONE_SYS_sendto, (uint64_t)mine, 2048, 1, 0, 0, 0) == -EBADF);
    assert(call6(CAPSTONE_SYS_recvfrom, (uint64_t)mine, 2048, 1, 0, 0, 0) == -EBADF);
    long ep = call(CAPSTONE_SYS_epoll_create1, EPOLL_CLOEXEC, 0, 0, 0);
    assert(ep >= 0);
    uint64_t ev[2] = {EPOLLIN, 7};
    put(256, ev, sizeof ev);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)ep, EPOLL_CTL_ADD, (uint64_t)mine, 256) == -EBADF);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)mine, EPOLL_CTL_ADD, (uint64_t)pair[0], 256) == -EBADF);
    /* inside a control message: refused before anything is sent */
    struct capstone_msghdr_block b;
    uint64_t pairs[2] = {1024, 1};
    put(1024, "x", 1);
    char control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof control);
    struct cmsghdr *cm = (struct cmsghdr *)control;
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &mine, sizeof(int));
    put(2048, control, sizeof control);
    block(&b, 0, 0, pairs, 1, 2048, sizeof control);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)mine, 64, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_recvmsg, (uint64_t)mine, 64, 0, 0) == -EBADF);
    assert(fcntl(pair[1], F_SETFL, O_NONBLOCK) == 0);
    assert(recv(pair[1], control, 1, 0) == -1 && errno == EAGAIN);   /* nothing was sent */
    /* a descriptor of the application's own crosses */
    memcpy(CMSG_DATA(cm), &pair[1], sizeof(int));
    put(2048, control, sizeof control);
    assert(call(CAPSTONE_SYS_sendmsg, (uint64_t)pair[0], 64, 0, 0) == 1);
    assert(fcntl(mine, F_GETFD) >= 0);
    close(pair[0]); close(pair[1]); close((int)ep); close(mine);
  } else if (!strcmp(c, "epoll")) {
    /* create, add a pipe with a 16-byte event, wait with timeout 0, then
       after a write; the data word comes back as given; modify, delete with
       a null event; a mask with the wrong set size is refused */
    long ep = call(CAPSTONE_SYS_epoll_create1, 0, 0, 0, 0);
    assert(ep >= 0);
    int p[2];
    assert(!pipe(p));
    uint64_t ev[2] = {EPOLLIN, UINT64_C(0x1234567890abcdef)};
    put(256, ev, sizeof ev);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)ep, EPOLL_CTL_ADD, (uint64_t)p[0], 256) == 0);
    memset(exchange + 1024, 0x44, 64);
    assert(call6(CAPSTONE_SYS_epoll_pwait, (uint64_t)ep, 1024, 4, 0, 0, 8) == 0);
    assert((unsigned char)exchange[1024] == 0x44);   /* nothing counted, nothing written */
    assert(write(p[1], "z", 1) == 1);
    assert(call6(CAPSTONE_SYS_epoll_pwait, (uint64_t)ep, 1024, 4, 100, 0, 8) == 1);
    uint64_t out[2];
    memcpy(out, exchange + 1024, sizeof out);
    assert((out[0] & EPOLLIN) && out[1] == UINT64_C(0x1234567890abcdef));
    assert((unsigned char)exchange[1024 + 16] == 0x44);   /* the second slot untouched */
    ev[0] = EPOLLOUT;
    put(256, ev, sizeof ev);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)ep, EPOLL_CTL_MOD, (uint64_t)p[0], 256) == 0);
    assert(call6(CAPSTONE_SYS_epoll_pwait, (uint64_t)ep, 1024, 4, 0, 0, 8) == 0);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)ep, EPOLL_CTL_DEL, (uint64_t)p[0], 0) == 0);
    assert(call(CAPSTONE_SYS_epoll_ctl, (uint64_t)ep, EPOLL_CTL_DEL, (uint64_t)p[0], 0) == -ENOENT);
    /* a masked wait: the mask is 8 bytes, or the request is malformed */
    uint64_t mask = 0;
    put(512, &mask, sizeof mask);
    assert(call6(CAPSTONE_SYS_epoll_pwait, (uint64_t)ep, 1024, 4, 0, 512, 8) == 0);
    assert(call6(CAPSTONE_SYS_epoll_pwait, (uint64_t)ep, 1024, 4, 0, 512, 4) == -EINVAL);
    close(p[0]); close(p[1]); close((int)ep);
  } else {
    abort();
  }
  capstone_delegate_host_free(&host);
  return 0;
}
