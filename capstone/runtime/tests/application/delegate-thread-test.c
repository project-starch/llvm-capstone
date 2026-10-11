/* A blocked message receive must retain its own validated view and iovecs
 * while a second native worker completes a different delegated receive. */
#define _GNU_SOURCE
#include "../../linux/delegate-service.h"
#include "capstone/msghdr.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

struct receiver {
    struct capstone_delegate_host host;
    struct capstone_delegate_entry request;
    char exchange[4096];
    _Atomic int entered;
};
static void setup(struct receiver *r, int fd, size_t target)
{
    memset(r, 0, sizeof *r);
    r->host.exchange = r->exchange;
    r->host.exchange_bytes = sizeof r->exchange;
    struct capstone_msghdr_block b = {.iov = 128, .iovlen = 1};
    uint64_t pair[2] = {target, 1};
    memcpy(r->exchange + 64, &b, sizeof b);
    memcpy(r->exchange + 128, pair, sizeof pair);
    uint64_t args[6] = {(uint64_t)fd, 64, 0, 0, 0, 0};
    assert(!capstone_delegate_pack(&r->request, CAPSTONE_SYS_recvmsg, args));
}
static void *receive(void *opaque)
{
    struct receiver *r = opaque;
    atomic_store(&r->entered, 1);
    capstone_delegate_serve(&r->host, &r->request);
    assert(r->request.result == 1);
    return NULL;
}
int main(void)
{
    struct receiver *a = malloc(sizeof *a), *b = malloc(sizeof *b);
    assert(a && b);
    for (unsigned i = 0; i < 32; ++i) {
        int left[2], right[2];
        pthread_t t;
        assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, left));
        assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, right));
        setup(a, left[0], 1024); setup(b, right[0], 2048);
        assert(!pthread_create(&t, NULL, receive, a));
        while (!atomic_load(&a->entered)) usleep(100);
        usleep(10000); /* let the first receive block in Linux */
        assert(write(right[1], "b", 1) == 1);
        receive(b);
        assert(write(left[1], "a", 1) == 1);
        assert(!pthread_join(t, NULL));
        assert(a->exchange[1024] == 'a' && !a->exchange[2048]);
        assert(b->exchange[2048] == 'b' && !b->exchange[1024]);
        capstone_delegate_host_free(&a->host);
        capstone_delegate_host_free(&b->host);
        close(left[0]); close(left[1]); close(right[0]); close(right[1]);
    }
    free(a); free(b);
    puts("delegate thread isolation passed");
    return 0;
}
