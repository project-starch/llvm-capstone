#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <capstone/virtual.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/epoll.h>
#include <stdint.h>

extern int __clone(int (*)(void *), void *, int, void *, ...);

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static unsigned arrived;
static __thread unsigned local = 41;
static int pipefd[2];

static void *worker(void *arg)
{
    assert(local == 41);
    local = *(unsigned *)arg;
    assert(!pthread_mutex_lock(&mutex));
    ++arrived;
    assert(!pthread_cond_broadcast(&cond));
    while (arrived < 2) assert(!pthread_cond_wait(&cond, &mutex));
    assert(!pthread_mutex_unlock(&mutex));
    assert(local == *(unsigned *)arg);
    return arg;
}
static void *reader(void *arg)
{
    char byte;
    assert(read(pipefd[0], &byte, 1) == 1 && byte == 'x');
    return arg;
}
static void *writer(void *arg)
{
    usleep(10000);
    assert(write(pipefd[1], "x", 1) == 1);
    return arg;
}
static void *identity(void *arg) { assert(local == 41); return arg; }
static void *epoll_worker(void *arg)
{
    int p[2], ep = epoll_create1(EPOLL_CLOEXEC);
    assert(ep >= 0 && !pipe(p));
    struct epoll_event in = {.events = EPOLLIN};
    in.data.u64 = *(unsigned *)arg;
    assert(!epoll_ctl(ep, EPOLL_CTL_ADD, p[0], &in));
    assert(write(p[1], "x", 1) == 1);
    for (unsigned i = 0; i < 256; ++i) {
        struct epoll_event out;
        assert(epoll_wait(ep, &out, 1, 1000) == 1);
        assert(out.events & EPOLLIN);
        assert(out.data.u64 == in.data.u64);
    }
    close(ep); close(p[0]); close(p[1]);
    return arg;
}
struct shared_writer { int fd; char *bytes; };
static void *shared_write(void *opaque)
{
    struct shared_writer *w = opaque;
    for (unsigned i = 0; i < 4; ++i) assert(write(w->fd, w->bytes, 65536) == 65536);
    capstone_virtual_thread_exit(NULL);
}
static void check_file(int fd, char *buffer, char value)
{
    assert(lseek(fd, 0, SEEK_SET) == 0);
    for (unsigned i = 0; i < 4; ++i) {
        assert(read(fd, buffer, 65536) == 65536);
        for (unsigned j = 0; j < 65536; ++j) assert(buffer[j] == value);
    }
    assert(read(fd, buffer, 1) == 0);
}

/* A worker's exit word in a page of the MAIN stack that nothing has touched (exit_word_worker):
 * the launcher stores 0 into it through Linux when the worker exits, and the main thread then
 * stores a pointer -- a capability, with its tag -- next to it. The module must admit the page
 * before the launcher's store (CV_TOUCH), or the tag lives in a frame nothing pins and nothing
 * sweeps at exit. The page must stay untouched by the domain until the launcher's store: the
 * page is inside an uninitialised stack array (a pointer derived from the array has the whole
 * array's bounds; one derived from the frame address has the frame's and faults, cause 24), the
 * worker is a raw virtual thread the main thread joins -- the join returns once the worker's
 * exit protocol ran -- and a short delegated sleep follows it, since the exit service marks the
 * thread done before it stores the word. A violated order shows as a missing `touch` line, a
 * FAIL, never as a pass. The runner checks the module's `touch` line for this page. */
extern _Noreturn void __capstone_vm_thread_exit_clear(uintptr_t clear, uintptr_t stack,
                                                      size_t bytes, long status);
static int *exit_word;
static void *exit_word_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear((uintptr_t)exit_word, 0, 0, 0);
}
/* A worker's transport cannot be retired under it (transport_lifetime): worker A names its
 * exchange mapping and blocks in a read; the main thread tries to unmap that mapping, which must
 * be refused with EBUSY -- the launcher will write A's read result into it, through Linux, and a
 * fresh mapping at the same address would be backing the module never admitted. Then the pipe
 * is fed, A's read completes into its own exchange, and A itself tries to unmap it while still
 * attached: refused too, since an unmap and a fresh mapping at the same address between two of
 * its services would leave the launcher writing to that address unadmitted. A then exits
 * normally (the SDK detaches before it unmaps). On a launcher without the hold the main
 * thread's unmap succeeds; on one that exempts the owner, A's does; either way no OK line. */
extern void *__capstone_virtual_exchange(void);
static void *transport_exchange;
static volatile int transport_blocked;
static int transport_pipe[2];
static int transport_own_unmap;
static void *transport_worker(void *arg)
{
    char byte;
    transport_exchange = __capstone_virtual_exchange();
    assert(transport_exchange);
    transport_blocked = 1;
    assert(read(transport_pipe[0], &byte, 1) == 1 && byte == 't');
    transport_own_unmap = munmap(transport_exchange, 1UL << 20) ? errno : 0;
    return arg;
}
/* A hold is released only for the mapping it was taken on (transport_identity): worker A attaches
 * (a delegated write), gives its transport back (detach, then munmap of both blocks) and stays
 * alive without one; worker B is created next and its exchange takes the address A's had (the
 * launcher reuses the most recently retired range of the same size); B attaches and waits. Then
 * A exits, and the main thread's munmap of B's exchange must still be refused with EBUSY. A
 * launcher that releases a hold by the exiting worker's old address (the nearest mapping at or
 * below it) drops B's hold at A's exit, and the munmap succeeds under B's attachment. The
 * workers wait by spinning on a flag with no call at all (the quantum preempts them): B must make
 * none between the main thread's munmap and its report. Runs before transport_lifetime, which a launcher that exempts the owner does not survive. */
extern void __capstone_virtual_transport_release(void);
static int identity_pipe[2];
static volatile int identity_state;
static void *volatile identity_exchange[2];
static void *identity_first(void *arg)
{
    assert(write(identity_pipe[1], "a", 1) == 1);   /* attached */
    identity_exchange[0] = __capstone_virtual_exchange();
    __capstone_virtual_transport_release();
    identity_state = 1;
    while (identity_state < 3) { }
    return arg;
}
static void *identity_second(void *arg)
{
    assert(write(identity_pipe[1], "b", 1) == 1);   /* attached */
    identity_exchange[1] = __capstone_virtual_exchange();
    identity_state = 2;
    while (identity_state < 4) { }
    return arg;
}
static void *allocator_worker(void *arg)
{
    for (unsigned i=0; i<8; ++i) {
        unsigned char *p = malloc(200003);
        assert(p);
        *(void **)p = arg; p[199999] = 91;
        p = realloc(p, 400000);
        assert(p && *(void **)p == arg && p[199999] == 91);
        p = realloc(p, 200003);
        assert(p && *(void **)p == arg && p[199999] == 91);
        free(p);
    }
    return arg;
}

int main(void)
{
    pthread_t a, b;
    unsigned values[2] = {51, 61};
    void *result;
    int tid;
    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
                CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID;
    assert(__clone(NULL, NULL, flags, NULL, &tid, NULL, &tid) == -ENOSYS);
    puts("VIRTUAL_PTHREAD_OK unsupported_clone_refused");
    assert(!pthread_create(&a, NULL, worker, &values[0]));
    assert(!pthread_create(&b, NULL, worker, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    assert(local == 41);
    puts("VIRTUAL_PTHREAD_OK tls mutex cond tagged_join");

    struct timespec deadline;
    assert(!clock_gettime(CLOCK_REALTIME, &deadline));
    deadline.tv_nsec += 1000000;
    if (deadline.tv_nsec >= 1000000000) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000; }
    assert(!pthread_mutex_lock(&mutex));
    assert(pthread_cond_timedwait(&cond, &mutex, &deadline) == ETIMEDOUT);
    assert(!pthread_mutex_unlock(&mutex));
    puts("VIRTUAL_PTHREAD_OK timed_wait");

    assert(!pipe(pipefd));
    assert(!pthread_create(&a, NULL, reader, &values[0]));
    assert(!pthread_create(&b, NULL, writer, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    close(pipefd[0]); close(pipefd[1]);
    puts("VIRTUAL_PTHREAD_OK independent_blocking_io");

    assert(!pthread_create(&a, NULL, epoll_worker, &values[0]));
    assert(!pthread_create(&b, NULL, epoll_worker, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    puts("VIRTUAL_PTHREAD_OK private_epoll_events");

    char *left = malloc(65536), *right = malloc(65536);
    assert(left && right);
    memset(left, 'a', 65536); memset(right, 'b', 65536);
    int fa = open("/tmp/virtual-shared-a", O_RDWR | O_CREAT | O_TRUNC, 0600);
    int fb = open("/tmp/virtual-shared-b", O_RDWR | O_CREAT | O_TRUNC, 0600);
    assert(fa >= 0 && fb >= 0);
    char *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    struct shared_writer w = {fb, right};
    long shared = capstone_virtual_thread_create(shared_write, &w, stack + 65536, tp);
    assert(shared > 0);
    for (unsigned i = 0; i < 4; ++i) assert(write(fa, left, 65536) == 65536);
    assert(!capstone_virtual_thread_join(shared));
    check_file(fa, left, 'a'); check_file(fb, right, 'b');
    close(fa); close(fb); unlink("/tmp/virtual-shared-a"); unlink("/tmp/virtual-shared-b");
    assert(!munmap(stack, 65536)); free(left); free(right);
    puts("VIRTUAL_PTHREAD_OK shared_tls_transport");

    for (unsigned i = 0; i < 64; ++i) {
        assert(!pthread_create(&a, NULL, identity, &values[0]));
        assert(!pthread_join(a, &result) && result == &values[0]);
    }
    puts("VIRTUAL_PTHREAD_OK 64_joined_lifetimes");
    assert(!pthread_create(&a, NULL, allocator_worker, &values[0]));
    assert(!pthread_create(&b, NULL, allocator_worker, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    puts("VIRTUAL_PTHREAD_OK concurrent_allocator_remap");

    {
        assert(!pipe(identity_pipe));
        assert(!pthread_create(&a, NULL, identity_first, &values[0]));
        while (identity_state < 1) sched_yield();
        assert(!pthread_create(&b, NULL, identity_second, &values[1]));
        while (identity_state < 2) sched_yield();
        identity_state = 3;
        assert(!pthread_join(a, &result) && result == &values[0]);
        int unmapped = munmap((void *)(uintptr_t)identity_exchange[1], 1UL << 20);
        int reason = errno;
        /* Reported before B goes on: without the hold, B's own exit would fault in the unmapped
         * exchange first. */
        if (identity_exchange[0] != identity_exchange[1])
            printf("VIRTUAL_PTHREAD_IDENTITY_NOT_REUSED:0x%lx:0x%lx\n",
                   (unsigned long)(uintptr_t)identity_exchange[0],
                   (unsigned long)(uintptr_t)identity_exchange[1]);
        else if (unmapped == 0) puts("VIRTUAL_PTHREAD_IDENTITY_UNMAPPED");
        else if (reason != EBUSY) printf("VIRTUAL_PTHREAD_IDENTITY_ERRNO:%d\n", reason);
        else puts("VIRTUAL_PTHREAD_OK transport_identity");
        fflush(stdout);
        identity_state = 4;
        assert(!pthread_join(b, &result) && result == &values[1]);
        close(identity_pipe[0]); close(identity_pipe[1]);
    }

    {
        assert(!pipe(transport_pipe));
        assert(!pthread_create(&a, NULL, transport_worker, &values[0]));
        while (!transport_blocked) sched_yield();
        struct timespec settle = {0, 100000000};
        nanosleep(&settle, NULL);   /* A is in its read by now; a race here reads as a FAIL */
        /* munmap needs only the cursor: an integer-derived pointer names the mapping. */
        uintptr_t base = (uintptr_t)transport_exchange;
        int unmapped = munmap((void *)base, 1UL << 20);
        int reason = errno;
        assert(write(transport_pipe[1], "t", 1) == 1);
        assert(!pthread_join(a, &result) && result == &values[0]);
        close(transport_pipe[0]); close(transport_pipe[1]);
        if (unmapped == 0) puts("VIRTUAL_PTHREAD_TRANSPORT_UNMAPPED");
        else if (reason != EBUSY) printf("VIRTUAL_PTHREAD_TRANSPORT_ERRNO:%d\n", reason);
        else if (transport_own_unmap == 0) puts("VIRTUAL_PTHREAD_TRANSPORT_OWNER_UNMAPPED");
        else if (transport_own_unmap != EBUSY)
            printf("VIRTUAL_PTHREAD_TRANSPORT_OWNER_ERRNO:%d\n", transport_own_unmap);
        else puts("VIRTUAL_PTHREAD_OK transport_lifetime");
    }

    {
        volatile char slab[64 * 4096];   /* 64 pages of the main stack nothing writes */
        /* Pointer arithmetic, never an integer cast back to a pointer: the word's pointer must
         * carry the slab's capability, or the main thread's own read of it is refused (24). */
        size_t align = (size_t)(-(uintptr_t)slab) & 4095;
        exit_word = (int *)((char *)slab + align + 32 * 4096);
        printf("VIRTUAL_PTHREAD_WORD:0x%lx\n", (unsigned long)(uintptr_t)exit_word);
        fflush(stdout);
        char *wstack = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(wstack != MAP_FAILED);
        void *wtp;
        __asm__ volatile("movc %0, tp" : "=r"(wtp));
        long wid = capstone_virtual_thread_create(exit_word_worker, NULL, wstack + 65536, wtp);
        assert(wid > 0);
        assert(!capstone_virtual_thread_join(wid));
        struct timespec pause = {0, 50000000};
        nanosleep(&pause, NULL);
        assert(*exit_word == 0);
        /* The capability store: a pointer into the same page, 16 bytes past the word. */
        *(void *volatile *)((char *)exit_word + 16) = &values[0];
        puts("VIRTUAL_PTHREAD_OK exit_word_untouched_stack");
    }
    return 0;
}
