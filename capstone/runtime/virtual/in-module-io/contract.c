/* Exchange I/O lifetime and signal contract. Each faulting case is a separate
 * process. The raw exchange cases use only CV_TEST_HOOKS' test transport opt-in;
 * they never submit a direct-buffer request. See README.md for expected results. */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <capstone/delegate.h>

extern void __capstone_vm_delegate_transport(void *, void *, size_t);
extern long __capstone_vm_transport_detach(void);
extern void __capstone_virtual_transport_release(void);
static unsigned long cursor(void *p) { return __builtin_capstone_cap_get_cursor(p); }
static void nap(void) { struct timespec t = {0, 10000000}; nanosleep(&t, 0); }
static long counter(const char *name)
{
    char path[128], value[40] = {0};
    snprintf(path, sizeof path, "/sys/module/capstone_vm/parameters/%s", name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    long n = read(fd, value, sizeof value - 1);
    close(fd);
    if (n <= 0) return -1;
    if (value[0] == 'Y') return 1;
    if (value[0] == 'N') return 0;
    return atol(value);
}
static void *mapping(size_t n)
{
    void *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    return p;
}
struct request {
    int fd, error;
    char *buffer, *exchange;
    struct capstone_delegate_entry *meta;
    size_t bytes, exchange_bytes, offset;
    long result;
    int started, done;
};
static pthread_mutex_t state_lock = PTHREAD_MUTEX_INITIALIZER;
static void state_store(int *p, int value)
{
    pthread_mutex_lock(&state_lock); *p = value; pthread_mutex_unlock(&state_lock);
}
static int state_load(int *p)
{
    pthread_mutex_lock(&state_lock); int value = *p; pthread_mutex_unlock(&state_lock); return value;
}
static long raw(struct request *r)
{
    struct capstone_delegate_entry *e = r->meta;
    *e = (struct capstone_delegate_entry){.version = 2, .count = 1, .nr = 63,
        .args = {r->fd, r->offset, r->bytes}, .flags = 2, .result = -ENOSYS};
    __capstone_vm_delegate_transport(e, r->exchange, r->exchange_bytes);
    return e->result;
}
static void *reader(void *opaque)
{
    struct request *r = opaque;
    state_store(&r->started, 1);
    r->result = r->meta ? raw(r) : read(r->fd, r->buffer, r->bytes);
    r->error = errno;
    state_store(&r->done, 1);
    return 0;
}
static void wait_started(struct request *r)
{
    while (!state_load(&r->started)) nap();
    for (int i = 0; i < 30; ++i) nap();
    assert(!state_load(&r->done));
}
static volatile sig_atomic_t signals_seen;
static void handler(int signo)
{
    if (signo == SIGUSR1 || signo == SIGPIPE || signo == SIGALRM)
        ++signals_seen;
}
static void *signal_sender(void *opaque)
{
    int fd = *(int *)opaque;
    sigset_t blocked; sigemptyset(&blocked); sigaddset(&blocked, SIGUSR1);
    assert(!pthread_sigmask(SIG_BLOCK, &blocked, 0));
    for (int i = 0; i < 30; ++i) nap();
    /* The main Linux task is blocked in read; this worker masks SIGUSR1, so
     * the process-directed signal must interrupt the main task's CV_STEP. */
    assert(!kill(getpid(), SIGUSR1));
    for (int i = 0; i < 300 && !signals_seen; ++i) nap();
    assert(signals_seen == 1);
    assert(write(fd, "s", 1) == 1);
    return 0;
}
static void interrupted(int restart)
{
    int p[2]; char buf = 0;
    struct sigaction sa = {.sa_handler = handler, .sa_flags = restart ? SA_RESTART : 0};
    sigemptyset(&sa.sa_mask);
    signals_seen = 0;
    assert(!sigaction(SIGUSR1, &sa, 0) && !pipe(p));
    long before = counter("inline_interrupted");
    pthread_t thread;
    assert(!pthread_create(&thread, 0, signal_sender, &p[1]));
    long result = read(p[0], &buf, 1); int error = errno;
    int seen = signals_seen;
    assert(!pthread_join(thread, 0));
    long after = counter("inline_interrupted");
    if (before >= 0) assert(after > before);
    assert(seen == 1 && signals_seen == 1);
    if (restart) assert(result == 1 && buf == 's');
    else {
        assert(result == -1 && error == EINTR && buf == 0);
        assert(read(p[0], &buf, 1) == 1 && buf == 's');
    }
    printf("IOV1 signal restart=%d signals=%d read=%ld errno=%d interrupted_delta=%ld PASS\n",
           restart, seen, result, error, after - before);
    close(p[0]); close(p[1]);
}
static void sigpipe_case(void)
{
    int p[2];
    struct sigaction sa = {.sa_handler = handler};
    sigemptyset(&sa.sa_mask);
    signals_seen = 0;
    assert(!sigaction(SIGPIPE, &sa, 0) && !pipe(p));
    close(p[0]);
    long served0 = counter("inline_served"), stopped0 = counter("inline_stopped");
    long result = write(p[1], "x", 1);
    int error = errno;
    int seen = signals_seen;
    long served1 = counter("inline_served"), stopped1 = counter("inline_stopped");
    close(p[1]);
    assert(result == -1 && error == EPIPE && seen == 1);
    if (served0 >= 0 && counter("in_module_io"))
        assert(served1 > served0 && stopped1 > stopped0);
    printf("IOV1 sigpipe signals=%d write=%ld errno=%d served_delta=%ld stopped_delta=%ld PASS\n",
           seen, result, error, served1 - served0, stopped1 - stopped0);
}
struct partial_req {
    int rfd, wfd;
    int started;
    long r1, r2;
    uint32_t count;
    uint64_t pending;
};
static void *partial_worker(void *opaque)
{
    struct partial_req *q = opaque;
    struct capstone_delegate_entry *meta = mapping(CAPSTONE_DELEGATE_META_BYTES);
    char *exch = mapping(8192);
    memset(meta, 0, CAPSTONE_DELEGATE_META_BYTES);
    memset(exch, 0x3c, 8192);
    __capstone_virtual_transport_release();
    struct capstone_signal_block *block = (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET);
    /* First round uses a non-inline syscall (lseek) so the launcher attaches and
     * registers this worker transport and its signal state even under test hooks. */
    *meta = (struct capstone_delegate_entry){
        .version = 2, .count = 1, .nr = 62, .args = {(uint64_t)-1, 0, 0}, .flags = 0, .result = -ENOSYS
    };
    __capstone_vm_delegate_transport(meta, exch, 8192);
    q->r1 = meta->result;
    __atomic_store_n(&q->started, 1, __ATOMIC_SEQ_CST);
    /* 61440 bytes are now in the pipe (4096 bytes of space left); this 8192-byte write
     * transfers 4096 bytes and blocks in pipe_write until the main thread sends SIGUSR1. */
    *meta = (struct capstone_delegate_entry){
        .version = 2, .count = 1, .nr = 64, .args = {q->wfd, 0, 8192}, .flags = 2, .result = -ENOSYS
    };
    __capstone_vm_delegate_transport(meta, exch, 8192);
    q->r2 = meta->result;
    q->count = block->count;
    q->pending = meta->pending;
    if (q->count) {
        *meta = (struct capstone_delegate_entry){
            .version = 2, .count = 1, .nr = CAPSTONE_NR_SIGDONE,
            .args = {block->events[0].seq, 0, 0}, .flags = 0, .result = -ENOSYS
        };
        __capstone_vm_delegate_transport(meta, exch, 8192);
    }
    __capstone_vm_transport_detach();
    assert(!munmap(meta, CAPSTONE_DELEGATE_META_BYTES) && !munmap(exch, 8192));
    return 0;
}
static void partial_case(void)
{
    static char out[69632], in[69632];
    int p[2];
    memset(out, 0x5a, sizeof out);
    memset(in, 0, sizeof in);
    struct sigaction sa = {.sa_handler = handler, .sa_flags = SA_RESTART};
    sigemptyset(&sa.sa_mask);
    signals_seen = 0;
    assert(!sigaction(SIGALRM, &sa, 0) && !sigaction(SIGUSR1, &sa, 0) && !pipe(p));
    /* Warm up the exchange and pipe pages so the timer interrupts pipe_write, not page faults. */
    assert(write(p[1], out, 65536) == 65536);
    size_t warm = 0;
    while (warm < 65536) {
        long r = read(p[0], in + warm, 65536 - warm);
        assert(r > 0);
        warm += (size_t)r;
    }
    long served0 = counter("inline_served"), stopped0 = counter("inline_stopped");
    struct itimerval it = {.it_value = {0, 30000}};
    assert(!setitimer(ITIMER_REAL, &it, 0));
    long result = write(p[1], out, sizeof out);
    int error = errno;
    int seen = signals_seen;
    long served1 = counter("inline_served"), stopped1 = counter("inline_stopped");
    assert(result == 65536 && seen == 1);
    size_t got = 0;
    while (got < (size_t)result) {
        long r = read(p[0], in + got, (size_t)result - got);
        assert(r > 0);
        got += (size_t)r;
    }
    assert(!fcntl(p[0], F_SETFL, O_NONBLOCK));
    assert(read(p[0], in, 1) == -1 && errno == EAGAIN);
    assert(!memcmp(out, in, (size_t)result));
    if (served0 >= 0 && counter("in_module_io"))
        assert(served1 > served0 && stopped1 > stopped0);
    close(p[0]); close(p[1]);
    assert(!pipe(p));
    assert(write(p[1], out, 61440) == 61440);
    sigset_t usr1;
    sigemptyset(&usr1);
    sigaddset(&usr1, SIGUSR1);
    assert(!pthread_sigmask(SIG_BLOCK, &usr1, 0));
    struct partial_req q = {.rfd = p[0], .wfd = p[1]};
    pthread_t wthread;
    assert(!pthread_create(&wthread, 0, partial_worker, &q));
    while (!__atomic_load_n(&q.started, __ATOMIC_SEQ_CST)) nap();
    for (int i = 0; i < 30; ++i) nap();
    assert(!kill(getpid(), SIGUSR1));
    assert(!pthread_join(wthread, 0));
    assert(!pthread_sigmask(SIG_UNBLOCK, &usr1, 0));
    assert(q.r1 == -EBADF && q.r2 == 4096 && q.count == 1 && q.pending != 0);
    got = 0;
    while (got < 65536) {
        long r = read(p[0], in + got, 65536 - got);
        assert(r > 0);
        got += (size_t)r;
    }
    assert(!fcntl(p[0], F_SETFL, O_NONBLOCK));
    assert(read(p[0], in, 1) == -1 && errno == EAGAIN);
    close(p[0]); close(p[1]);
    printf("IOV1 partial signals=%d write=%ld raw_partial=%ld raw_published=%u served_delta=%ld stopped_delta=%ld PASS\n",
           seen, result, q.r2, q.count, served1 - served0, stopped1 - stopped0);
}
static void *boundary_worker(void *opaque)
{
    int fd = *(int *)opaque;
    struct capstone_delegate_entry *meta = mapping(CAPSTONE_DELEGATE_META_BYTES);
    char *exch = mapping(4096);
    memset(meta, 0, CAPSTONE_DELEGATE_META_BYTES);
    memset(exch, 0, 4096);
    __capstone_virtual_transport_release();
    struct request r = {
        .fd = fd, .meta = meta, .exchange = exch, .exchange_bytes = 4096,
        .offset = 0, .bytes = 1
    };
    /* First round attaches and registers this worker transport in the launcher. */
    long r1 = raw(&r);
    char c1 = exch[0];
    /* Second round runs on the now-registered transport in the module. */
    long r2 = raw(&r);
    char c2 = exch[0];
    /* P3: offset == exchange_bytes with length == 0 must return -EFAULT, never 0. */
    r.offset = 4096; r.bytes = 0;
    long r3 = raw(&r);
    /* Overflowing offset + length on an already-registered transport must return -EFAULT. */
    r.offset = 4096; r.bytes = 1;
    long r4 = raw(&r);
    r.offset = 100; r.bytes = 4000;
    long r5 = raw(&r);
    /* Malformed flags on an already-registered transport must return -EINVAL. */
    *meta = (struct capstone_delegate_entry){
        .version = 2, .count = 1, .nr = 63, .args = {fd, 0, 1}, .flags = 0, .result = -ENOSYS
    };
    __capstone_vm_delegate_transport(meta, exch, 4096);
    long r6 = meta->result;
    __capstone_vm_transport_detach();
    assert(!munmap(meta, CAPSTONE_DELEGATE_META_BYTES) && !munmap(exch, 4096));
    assert(r1 == 1 && c1 == 'a' && r2 == 1 && c2 == 'b');
    assert(r3 == -EFAULT && r4 == -EFAULT && r5 == -EFAULT && r6 == -EINVAL);
    return 0;
}
static void functional(void)
{
    char buf[8] = {0}; int p[2], sv[2];
    int fd = open("/tmp/iov1-contract", O_CREAT | O_TRUNC | O_RDWR, 0600);
    assert(fd >= 0);
    long before = counter("inline_served");
    assert(write(fd, "abcdef", 6) == 6);
    assert(pwrite(fd, "XY", 2, 2) == 2 && lseek(fd, 0, SEEK_CUR) == 6);
    assert(pread(fd, buf, 6, 0) == 6 && !memcmp(buf, "abXYef", 6));
    assert(lseek(fd, 0, SEEK_CUR) == 6);
    assert(pread(fd, buf, 1, -1) == -1 && errno == EINVAL);
    assert(!pipe(p));
    assert(pread(p[0], buf, 1, 0) == -1 && errno == ESPIPE);
    assert(read(-1, buf, 1) == -1 && errno == EBADF);
    assert(write(p[1], "z", 1) == 1 && read(p[0], buf, 1) == 1 && buf[0] == 'z');
    assert(read(p[0], buf, 0) == 0);
    close(p[1]); assert(read(p[0], buf, 1) == 0); close(p[0]);
    int efd = eventfd(1, EFD_NONBLOCK);
    assert(efd >= 0);
    assert(read(efd, buf, 0) == -1 && errno == EINVAL);
    close(efd);
    assert(!socketpair(AF_UNIX, SOCK_DGRAM, 0, sv));
    assert(write(sv[0], buf, 0) == 0);
    assert(recv(sv[1], buf, sizeof buf, MSG_DONTWAIT) == 0);
    assert(recv(sv[1], buf, sizeof buf, MSG_DONTWAIT) == -1 && errno == EAGAIN);
    close(sv[0]); close(sv[1]);
    assert(!pipe(p) && write(p[1], "ab", 2) == 2);
    pthread_t bthread;
    assert(!pthread_create(&bthread, 0, boundary_worker, &p[0]));
    assert(!pthread_join(bthread, 0));
    close(p[0]); close(p[1]);
    long after = counter("inline_served");
    if (before >= 0 && counter("in_module_io")) assert(after >= before + 9);
    close(fd); unlink("/tmp/iov1-contract");
    printf("IOV1 functional served_delta=%ld PASS\n", after - before);
}
static void heap_case(int stale)
{
    int p[2]; assert(!pipe(p));
    char *obj = malloc(256); assert(obj); memset(obj, 0x44, 256);
    struct request r = {.fd = p[0], .buffer = obj, .bytes = 256};
    pthread_t thread;
    if (!stale) { assert(!pthread_create(&thread, 0, reader, &r)); wait_started(&r); }
    free(obj);
    char *fresh = malloc(256); assert(fresh); memset(fresh, 0xAA, 256);
    printf("IOV1 heap stale=%d revoked=1 old=%lx new=%lx EXPECT_CAUSE_25\n", stale, cursor(obj), cursor(fresh));
    fflush(stdout);
    char data[256]; memset(data, 0x5c, sizeof data);
    assert(write(p[1], data, sizeof data) == sizeof data);
    if (stale) reader(&r); else pthread_join(thread, 0);
    puts("IOV1 heap FAIL: revoked copy-back returned");
    exit(1);
}
static void recycle(void)
{
    int p[2]; assert(!pipe(p));
    char *x = mapping(4096), data[256]; memset(x, 0x11, 4096);
    memset(data, 0x5c, sizeof data);
    struct request r = {.fd = p[0], .meta = mapping(CAPSTONE_DELEGATE_META_BYTES),
        .exchange = x, .exchange_bytes = 4096, .offset = 64, .bytes = sizeof data};
    memset(r.meta, 0, CAPSTONE_DELEGATE_META_BYTES);
    int hold = counter("inline_hold"); assert(hold >= 0 && counter("inline_test_transport") == 1);
    pthread_t thread; assert(!pthread_create(&thread, 0, reader, &r)); wait_started(&r);
    int unmap = munmap(x, 4096), error = errno;
    char *y = mapping(4096); memset(y, 0xAA, 4096);
    assert(write(p[1], data, sizeof data) == sizeof data);
    assert(!pthread_join(thread, 0));
    assert(r.result == 256);
    int changed = 0; for (int i = 0; i < 4096; ++i) changed += (unsigned char)y[i] != 0xAA;
    if (hold) {
        assert(unmap == -1 && error == EBUSY && cursor(x) != cursor(y) && !changed);
        assert(!memcmp(x + 64, data, 256) && !munmap(x, 4096));
    } else {
        assert(!unmap && cursor(x) == cursor(y) && changed == 256 && !memcmp(y + 64, data, 256));
    }
    printf("IOV1 recycle hold=%d unmap=%d errno=%d reused=%d changed=%d %s\n",
           hold, unmap, error, cursor(x) == cursor(y), changed, hold ? "PASS" : "LANDED");
    munmap(y, 4096); munmap(r.meta, CAPSTONE_DELEGATE_META_BYTES); close(p[0]); close(p[1]);
}
static void rejected(const char *mode, int shared_input)
{
    /* Invalid declared exchanges fall back and are refused by the launcher's
     * transport validator. The read must never execute or return to this code. */
    int p[2] = {0, -1};
    if (!shared_input) { assert(!pipe(p)); assert(write(p[1], "r", 1) == 1); }
    struct request r = {.fd = p[0], .meta = mapping(CAPSTONE_DELEGATE_META_BYTES),
        .exchange = mapping(4096), .exchange_bytes = 4096, .bytes = 1};
    memset(r.meta, 0, CAPSTONE_DELEGATE_META_BYTES);
    if (!strcmp(mode, "span")) {
        int found = 0;
        for (int i = 0; i < 6 && !found; ++i) {
            char *x = mapping(8192), *y = mapping(4096);
            if (cursor(y) == cursor(x) + 8192) {
                r.exchange = x; r.exchange_bytes = 12288; r.offset = 8192 - 100; r.bytes = 200;
                found = 1;
            }
        }
        assert(found);
        puts("IOV1 span adjacent=1 crossing=100+100");
    }
    else if (!strcmp(mode, "outside")) r.exchange = (void *)0x10000UL;
    else if (!strcmp(mode, "text")) r.exchange = (void *)&rejected;
    else if (!strcmp(mode, "retired")) assert(!munmap(r.exchange, 4096));
    else if (!strcmp(mode, "forged")) { /* valid but unregistered, production build */ }
    printf("IOV1 %s EXPECT_TRANSPORT_REFUSAL\n", mode); fflush(stdout);
    raw(&r);
    printf("IOV1 %s FAIL: unregistered transport returned\n", mode); exit(1);
}
int main(int argc, char **argv)
{
    assert(argc == 2 || argc == 3);
    if (!strcmp(argv[1], "functional")) functional();
    else if (!strcmp(argv[1], "signal")) interrupted(0);
    else if (!strcmp(argv[1], "restart")) interrupted(1);
    else if (!strcmp(argv[1], "sigpipe")) sigpipe_case();
    else if (!strcmp(argv[1], "partial")) partial_case();
    else if (!strcmp(argv[1], "heap-libc")) heap_case(0);
    else if (!strcmp(argv[1], "stale-libc")) heap_case(1);
    else if (!strcmp(argv[1], "recycle")) recycle();
    else rejected(argv[1], argc == 3 && !strcmp(argv[2], "stdin"));
    return 0;
}
