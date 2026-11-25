/* msgt: sendmsg/recvmsg in a protected program on the trap route. A socketpair; sendmsg of two
 * iovecs with SCM_RIGHTS (an fd); recvmsg into two iovecs with a control buffer and a name buffer;
 * then the readback: bytes, split, MSG_TRUNC on a short buffer, controllen, the passed fd works. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/uio.h>
int main(void) {
    int sv[2], fails = 0;
    if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sv)) { printf("MSGT socketpair FAIL\n"); return 1; }
    char a[] = "hello ", b[] = "world";
    struct iovec ov[2] = {{a, 6}, {b, 5}};
    int pfd[2]; if (pipe(pfd)) return 2;
    union { char buf[CMSG_SPACE(sizeof(int))]; struct cmsghdr al; } cu;
    memset(&cu, 0, sizeof cu);
    struct msghdr om = {0}; om.msg_iov = ov; om.msg_iovlen = 2; om.msg_control = cu.buf; om.msg_controllen = sizeof cu.buf;
    struct cmsghdr *c = CMSG_FIRSTHDR(&om); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &pfd[1], sizeof(int));
    ssize_t n = sendmsg(sv[0], &om, 0);
    printf("MSGT sendmsg %zd\n", n); if (n != 11) fails++;
    char x[4], y[16]; struct iovec iv[2] = {{x, 4}, {y, 16}};
    union { char buf[CMSG_SPACE(sizeof(int))]; struct cmsghdr al; } ru; memset(&ru, 0, sizeof ru);
    struct sockaddr_storage nm; memset(&nm, 0, sizeof nm);
    struct msghdr im = {0}; im.msg_name = &nm; im.msg_namelen = sizeof nm; im.msg_iov = iv; im.msg_iovlen = 2;
    im.msg_control = ru.buf; im.msg_controllen = sizeof ru.buf;
    n = recvmsg(sv[1], &im, 0);
    printf("MSGT recvmsg %zd x=%.4s y=%.7s flags=%#x controllen=%u namelen=%u\n", n, x, y, im.msg_flags,
           (unsigned)im.msg_controllen, (unsigned)im.msg_namelen);
    if (n != 11 || memcmp(x, "hell", 4) || memcmp(y, "o world", 7)) fails++;
    struct cmsghdr *rc = CMSG_FIRSTHDR(&im); int got = -1;
    if (rc && rc->cmsg_type == SCM_RIGHTS) memcpy(&got, CMSG_DATA(rc), sizeof(int)); else fails++;
    if (got >= 0) { ssize_t w = write(got, "z", 1); char z = 0; ssize_t r = read(pfd[0], &z, 1);
        printf("MSGT passed-fd write %zd read %zd z=%c\n", w, r, z); if (w != 1 || r != 1 || z != 'z') fails++; }
    /* MSG_TRUNC: a 3-byte buffer for an 11-byte datagram */
    sendmsg(sv[0], &(struct msghdr){.msg_iov = ov, .msg_iovlen = 2}, 0);
    char t[3]; struct iovec tv = {t, 3}; struct msghdr tm = {0}; tm.msg_iov = &tv; tm.msg_iovlen = 1;
    n = recvmsg(sv[1], &tm, 0);
    printf("MSGT trunc %zd flags=%#x\n", n, tm.msg_flags); if (n != 3 || !(tm.msg_flags & MSG_TRUNC)) fails++;
    printf("MSGT done fails=%d\n", fails);
    return fails;
}
