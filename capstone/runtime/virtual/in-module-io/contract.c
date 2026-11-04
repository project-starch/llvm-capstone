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
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <capstone/delegate.h>

extern void __capstone_vm_delegate_transport(void *, void *, size_t);
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
static void handler(int signo) { if (signo == SIGUSR1) ++signals_seen; }
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
    assert(!sigaction(SIGUSR1, &sa, 0) && !pipe(p));
    long before = counter("inline_interrupted");
    pthread_t thread;
    assert(!pthread_create(&thread, 0, signal_sender, &p[1]));
    long result = read(p[0], &buf, 1); int error = errno;
    assert(!pthread_join(thread, 0));
    long after = counter("inline_interrupted");
    if (before >= 0) assert(after > before);
    assert(signals_seen == 1);
    if (restart) assert(result == 1 && buf == 's');
    else {
        assert(result == -1 && error == EINTR && buf == 0);
        assert(read(p[0], &buf, 1) == 1 && buf == 's');
    }
    printf("IOV1 signal restart=%d signals=%d read=%ld errno=%d interrupted_delta=%ld PASS\n",
           restart, signals_seen, result, error, after - before);
    close(p[0]); close(p[1]);
}
static void functional(void)
{
    char buf[8] = {0}; int p[2];
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
    long after = counter("inline_served");
    if (before >= 0 && counter("in_module_io")) assert(after >= before + 7);
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
    else if (!strcmp(argv[1], "heap-libc")) heap_case(0);
    else if (!strcmp(argv[1], "stale-libc")) heap_case(1);
    else if (!strcmp(argv[1], "recycle")) recycle();
    else rejected(argv[1], argc == 3 && !strcmp(argv[2], "stdin"));
    return 0;
}
