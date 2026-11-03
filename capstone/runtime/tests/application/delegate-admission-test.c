/* Trusted output admission precedes I/O side effects, including nested spans. */
#include "../../linux/delegate-service.h"
#include "capstone/msghdr.h"
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

enum { BYTES = 12288, DEST = 4093 };
static size_t denied, seen[16], seen_bytes[16], calls;

static int admit(struct capstone_delegate_host *h, size_t at, size_t bytes)
{
    assert(at <= BYTES && bytes && bytes <= BYTES - at && calls < 16);
    seen[calls] = at; seen_bytes[calls++] = bytes;
    if (at == denied) return ENOMEM;
    size_t page = at & ~(size_t)4095;
    assert(!mprotect(h->exchange + page, ((at + bytes + 4095) & ~(size_t)4095) - page,
                     PROT_READ | PROT_WRITE));
    return 0;
}

static long serve(struct capstone_delegate_host *h, uint64_t nr, uint64_t a,
                  uint64_t b, uint64_t c, uint64_t d)
{
    struct capstone_delegate_entry e;
    uint64_t args[6] = {a, b, c, d, 0, 0};
    assert(!capstone_delegate_pack(&e, nr, args));
    calls = 0;
    capstone_delegate_serve(h, &e);
    return (long)e.result;
}

static int saw(size_t at, size_t bytes)
{
    for (size_t i = 0; i < calls; ++i)
        if (seen[i] == at && seen_bytes[i] == bytes) return 1;
    return 0;
}

int main(void)
{
    char *p = mmap(NULL, BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    struct capstone_delegate_host h = {.exchange = p, .exchange_bytes = BYTES, .prepare_write = admit};
    int fds[2];
    assert(!pipe(fds));
    for (unsigned vector = 0; vector < 2; ++vector) {
        uint64_t pair[] = {DEST, 6};
        memcpy(p + 64, pair, sizeof pair);
        memset(p + DEST, 'x', 6);
        assert(write(fds[1], "abcdef", 6) == 6);
        assert(!mprotect(p, BYTES, PROT_READ));
        denied = DEST;
        uint64_t nr = vector ? CAPSTONE_SYS_readv : CAPSTONE_SYS_read;
        assert(serve(&h, nr, fds[0], vector ? 64 : DEST, vector ? 1 : 6, 0) == -ENOMEM);
        assert(saw(DEST, 6) && !memcmp(p + DEST, "xxxxxx", 6));
        denied = BYTES;
        /* The denied read did not consume the pipe; the same call succeeds. */
        assert(serve(&h, nr, fds[0], vector ? 64 : DEST, vector ? 1 : 6, 0) == 6);
        assert(saw(DEST, 6) && !memcmp(p + DEST, "abcdef", 6));
    }
    close(fds[0]); close(fds[1]);
    assert(!socketpair(AF_UNIX, SOCK_DGRAM, 0, fds));
    struct capstone_msghdr_block msg = {.name = 8200, .namelen = 32, .iov = 64,
        .iovlen = 1, .control = 8300, .controllen = 128};
    memcpy(p + 128, &msg, sizeof msg);
    assert(write(fds[0], "ghijkl", 6) == 6);
    assert(!mprotect(p, BYTES, PROT_READ));
    denied = DEST;
    assert(serve(&h, CAPSTONE_SYS_recvmsg, fds[1], 128, 0, 0) == -ENOMEM);
    assert(saw(DEST, 6));
    denied = BYTES;
    assert(serve(&h, CAPSTONE_SYS_recvmsg, fds[1], 128, 0, 0) == 6);
    assert(!memcmp(p + DEST, "ghijkl", 6));
    assert(saw(128, sizeof msg) && saw(8200, 32) && saw(8300, 128));
    close(fds[0]); close(fds[1]);
    denied = 8192;
    assert(serve(&h, CAPSTONE_NR_THREAD_NAME, CAPSTONE_THREAD_NAME_GET, denied, 0, 0) == -ENOMEM);
    assert(saw(denied, CAPSTONE_THREAD_NAME_BYTES));
    assert(serve(&h, CAPSTONE_SYS_rt_sigprocmask, SIG_BLOCK, 0, denied, 8) == -ENOMEM);
    assert(saw(denied, 8));
    capstone_delegate_host_free(&h);
    assert(!munmap(p, BYTES));
    puts("delegate-admission-test: scalar/vector cross-page output, message spans, signal/name denial: PASS");
    return 0;
}
