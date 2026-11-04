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
#include <signal.h>
#include <setjmp.h>
#include <sys/syscall.h>

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

/* The transport lock's context identity (delegate.c dl_wire_identity): the cursor of the
 * context's own gp, as a slot offset from gp's base. 0 in the first context; nonzero, below
 * the fallback slot (64 * 16) and distinct in every live further context, whether it has its
 * own TLS or shares its creator's; unchanged in a handler on an alternate signal stack; and
 * given back at exit, so a process that creates more contexts than there are slots never
 * reaches the fallback. Read without a service exit, which the runner checks too. */
static unsigned long gp_slot(void)
{
    void *gp;
    __asm__ volatile("movc %0, gp" : "=r"(gp));
    return __builtin_capstone_cap_get_cursor(gp) - __builtin_capstone_cap_get_base(gp);
}
static pthread_barrier_t slot_barrier;
static volatile unsigned long slot_seen[2], raw_slot, handler_slot = ~0UL;
static volatile long handler_wrote;
static int slot_pipe[2];
static void *slot_worker(void *arg)
{
    slot_seen[*(unsigned *)arg] = gp_slot();
    pthread_barrier_wait(&slot_barrier);   /* both alive when each reads its slot */
    return arg;
}
static void *raw_slot_worker(void *arg)
{
    (void)arg;
    raw_slot = gp_slot();
    capstone_virtual_thread_exit(NULL);
}
static void *slot_once(void *arg) { *(unsigned long *)arg = gp_slot(); return arg; }
static void on_slot_signal(int sig)
{
    (void)sig;
    handler_slot = gp_slot();
    /* A delegated call nested in the delivery, which runs under the interrupted call's lock. */
    handler_wrote = write(slot_pipe[1], "s", 1);
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
    /* The raw exit service bypasses the runtime's exits, so this context keeps its identity slot
     * (delegate.c): one slot of 63, after context_identity has run. Giving it back would need
     * __capstone_context_identity_release, which dev's SDK (the control) does not define, and a
     * weak reference to it does not read as null here: the gp-free codegen forms the address
     * pc-relative, so the undefined symbol is the image's base (control run: cause 12 at it). */
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

/* Raw exits deliberately keep their slots. Exhaust all 63, then four raw
 * contexts sharing the main TLS and default transport write checked records.
 * Join before reading: all 4096 bytes fit even the minimum Linux pipe size,
 * so a reader cannot block holding the same wire the writers need. */
#define FALLBACK_THREADS 4
#define FALLBACK_RECORDS 32
extern _Noreturn void __capstone_vm_thread_exit(void);
static volatile unsigned fallback_arrived, fallback_go;
static unsigned long fallback_slots[FALLBACK_THREADS];
static unsigned fallback_ids[FALLBACK_THREADS];
static void *keep_slot(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit();
}
static void *fallback_writer(void *arg)
{
    unsigned t = *(unsigned *)arg;
    fallback_slots[t] = gp_slot();
    __atomic_add_fetch(&fallback_arrived, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&fallback_go, __ATOMIC_ACQUIRE)) { }
    for (unsigned i = 0; i < FALLBACK_RECORDS; ++i) {
        unsigned record[8];
        for (unsigned j = 0; j < 8; ++j) record[j] = (t * FALLBACK_RECORDS + i) * 8 + j;
        assert(write(pipefd[1], record, sizeof record) == sizeof record);
    }
    return arg;
}
static unsigned fallback_check_records(void)
{
    unsigned char seen[FALLBACK_THREADS * FALLBACK_RECORDS] = {0};
    unsigned bad = 0;
    for (unsigned n = 0; n < FALLBACK_THREADS * FALLBACK_RECORDS; ++n) {
        unsigned record[8];
        size_t got = 0;
        while (got < sizeof record) {
            ssize_t r = read(pipefd[0], (char *)record + got, sizeof record - got);
            assert(r > 0); got += r;
        }
        unsigned id = record[0] / 8;
        if (id >= sizeof seen || seen[id]++) ++bad;
        for (unsigned j = 0; j < 8; ++j) if (record[j] != id * 8 + j) ++bad;
    }
    return bad;
}
static int fallback_contract(void)
{
    char *stacks = mmap(NULL, FALLBACK_THREADS * 65536, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stacks != MAP_FAILED);
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    for (unsigned i = 0; i < 63; ++i) {
        long id = capstone_virtual_thread_create(keep_slot, NULL, stacks + 65536, tp);
        assert(id > 0 && !capstone_virtual_thread_join(id));
    }
    long threads[FALLBACK_THREADS];
    assert(!pipe(pipefd));
    for (unsigned t = 0; t < FALLBACK_THREADS; ++t) {
        fallback_ids[t] = t;
        threads[t] = capstone_virtual_thread_create(fallback_writer, &fallback_ids[t],
                                                    stacks + (t + 1) * 65536, tp);
        assert(threads[t] > 0);
    }
    while (__atomic_load_n(&fallback_arrived, __ATOMIC_ACQUIRE) != FALLBACK_THREADS) { }
    __atomic_store_n(&fallback_go, 1, __ATOMIC_RELEASE);
    for (unsigned t = 0; t < FALLBACK_THREADS; ++t)
        assert(!capstone_virtual_thread_join(threads[t]));
    unsigned bad = fallback_check_records();
    unsigned fallback = 0;
    for (unsigned t = 0; t < FALLBACK_THREADS; ++t) {
        fallback += fallback_slots[t] == 64 * 16;
    }
    /* Exercise __clone, private TLS/transport installation and normal pthread
     * teardown after exhaustion too. Returning through the pthread trampoline
     * must release the fallback transport without recycling a real gp slot. */
    pthread_t pthreads[FALLBACK_THREADS];
    fallback_arrived = fallback_go = 0;
    for (unsigned t = 0; t < FALLBACK_THREADS; ++t) {
        fallback_slots[t] = 0;
        assert(!pthread_create(&pthreads[t], NULL, fallback_writer, &fallback_ids[t]));
    }
    while (__atomic_load_n(&fallback_arrived, __ATOMIC_ACQUIRE) != FALLBACK_THREADS) { }
    __atomic_store_n(&fallback_go, 1, __ATOMIC_RELEASE);
    unsigned pthread_slots = 0;
    for (unsigned t = 0; t < FALLBACK_THREADS; ++t) {
        void *result;
        assert(!pthread_join(pthreads[t], &result) && result == &fallback_ids[t]);
        pthread_slots += fallback_slots[t] == 64 * 16;
    }
    unsigned pthread_bad = fallback_check_records();
    printf("VIRTUAL_FALLBACK_PTHREAD:slots=%u records=%u bad=%u\n",
           pthread_slots, FALLBACK_THREADS * FALLBACK_RECORDS, pthread_bad);
    if (pthread_slots == FALLBACK_THREADS && !pthread_bad)
        puts("VIRTUAL_PTHREAD_OK fallback_pthread");
    close(pipefd[0]); close(pipefd[1]);
    assert(!munmap(stacks, FALLBACK_THREADS * 65536));
    printf("VIRTUAL_FALLBACK:slots=%u records=%u bad=%u main=%lu\n", fallback,
           FALLBACK_THREADS * FALLBACK_RECORDS, bad, gp_slot());
    if (fallback == FALLBACK_THREADS && !bad && gp_slot() == 0)
        puts("VIRTUAL_PTHREAD_OK fallback_identity");
    return !(fallback == FALLBACK_THREADS && !bad && gp_slot() == 0 &&
             pthread_slots == FALLBACK_THREADS && !pthread_bad);
}

/* Abandon a default-transport round from its signal handler. The observer
 * has a private transport so it can report without touching the stranded
 * wire. A replacement raw context must not inherit the abandoned owner's
 * gp slot. Both direct handler exit and longjmp followed by exit are covered. */
static void *abandoned_tp;
static unsigned long abandoned_slot, replacement_slot;
static int abandon_jump;
static jmp_buf abandon_target;
static void abandon_handler(int signo)
{
    (void)signo;
    abandoned_slot = gp_slot();
    if (abandon_jump) longjmp(abandon_target, 1);
    capstone_virtual_thread_exit(NULL);
}
static void *abandon_owner(void *arg)
{
    (void)arg;
    sigset_t mask;
    assert(!sigemptyset(&mask) && !sigaddset(&mask, SIGUSR1));
    assert(!sigprocmask(SIG_UNBLOCK, &mask, NULL));
    if (!setjmp(abandon_target)) {
        /* This is the only native worker with SIGUSR1 unblocked. Raw
         * shared-TLS contexts do not have an independent musl tid for
         * raise(), so target the process with kill instead. */
        assert(!kill(getpid(), SIGUSR1));
        assert(!"handler returned");
    }
    capstone_virtual_thread_exit(NULL);
}
static void *replacement_owner(void *arg)
{
    (void)arg;
    replacement_slot = gp_slot();
    capstone_virtual_thread_exit(NULL);
}
static void *observe_abandoned(void *arg)
{
    (void)arg;
    char *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    long id = capstone_virtual_thread_create(abandon_owner, NULL, stack + 65536, abandoned_tp);
    assert(id > 0 && !capstone_virtual_thread_join(id));
    id = capstone_virtual_thread_create(replacement_owner, NULL, stack + 65536, abandoned_tp);
    assert(id > 0 && !capstone_virtual_thread_join(id));
    printf("VIRTUAL_ABANDONED:old=%lu new=%lu jump=%d\n",
           abandoned_slot, replacement_slot, abandon_jump);
    int ok = abandoned_slot && replacement_slot && abandoned_slot != replacement_slot;
    if (ok) puts("VIRTUAL_PTHREAD_OK abandoned_identity");
    fflush(stdout);
    _Exit(ok ? 0 : 1);
}
static int abandoned_contract(int jump)
{
    abandon_jump = jump;
    __asm__ volatile("movc %0, tp" : "=r"(abandoned_tp));
    struct sigaction sa = {.sa_handler = abandon_handler};
    assert(!sigemptyset(&sa.sa_mask) && !sigaction(SIGUSR1, &sa, NULL));
    sigset_t mask;
    assert(!sigemptyset(&mask) && !sigaddset(&mask, SIGUSR1));
    assert(!sigprocmask(SIG_BLOCK, &mask, NULL));
    pthread_t observer;
    assert(!pthread_create(&observer, NULL, observe_abandoned, NULL));
    for (;;) __asm__ volatile("" ::: "memory");
}

static void *private_tls_child(void *arg)
{
    (void)arg;
    /* The control accepts creation. Exit before its known fatal transport
     * attachment, so the oracle reports the missing refusal as a test
     * failure instead of conflating it with a launcher setup failure. */
    capstone_virtual_thread_exit(NULL);
}
static void *private_tls_target;
static volatile int private_tls_ready, private_tls_done;
static void private_tls_refuse(void *tp)
{
    char *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    long id = capstone_virtual_thread_create(private_tls_child, NULL, stack + 65536, tp);
    int reason = errno;
    if (id > 0) capstone_virtual_thread_join(id);
    assert(id == -1 && reason == ENOTSUP);
    assert(!munmap(stack, 65536));
}
static void *private_tls_creator(void *arg)
{
    (void)arg;
    assert(write(1, "PRIVATE_TLS_BEGIN\n", 18) == 18);
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    private_tls_refuse(tp);
    private_tls_target = tp;
    __atomic_store_n(&private_tls_ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&private_tls_done, __ATOMIC_ACQUIRE)) { }
    puts("VIRTUAL_PTHREAD_OK private_tls_refused");
    return NULL;
}
static void *private_tls_other(void *arg)
{
    private_tls_refuse(arg);
    return arg;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "fallback")) return fallback_contract();
    if (argc == 2 && !strcmp(argv[1], "identity-exit")) return abandoned_contract(0);
    if (argc == 2 && !strcmp(argv[1], "identity-jump")) return abandoned_contract(1);
    if (argc == 2 && !strcmp(argv[1], "private-tls")) {
        pthread_t creator;
        assert(!pthread_create(&creator, NULL, private_tls_creator, NULL));
        while (!__atomic_load_n(&private_tls_ready, __ATOMIC_ACQUIRE)) { }
        private_tls_refuse(private_tls_target);  /* main -> another pthread */
        pthread_t other;
        assert(!pthread_create(&other, NULL, private_tls_other, private_tls_target));
        assert(!pthread_join(other, NULL));     /* pthread -> another pthread */
        __atomic_store_n(&private_tls_done, 1, __ATOMIC_RELEASE);
        assert(!pthread_join(creator, NULL));
        puts("VIRTUAL_PTHREAD_OK private_tls_other_refused");
        return 0;
    }
    pthread_t a, b;
    unsigned values[2] = {51, 61};
    void *result;
    int tid;
    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
                CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID;
    assert(__clone(NULL, NULL, flags, NULL, &tid, NULL, &tid) == -ENOSYS);
    puts("VIRTUAL_PTHREAD_OK unsupported_clone_refused");
    /* A new process cannot inherit this mm's lifetime namespace. Keep the
     * libc entry and the raw syscall route fail-closed; pthreads below must
     * still work after the refusal. No child may execute this image. */
    errno = 0;
    assert(fork() == -1 && errno == ENOSYS);
    errno = 0;
    assert(syscall(SYS_clone, SIGCHLD, 0, 0, 0, 0) == -1 && errno == ENOSYS);
    puts("VIRTUAL_PTHREAD_OK protected_fork_refused");
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

    /* Before the exit-word case, whose check reads its page's touch line from the kernel log:
     * the threads below admit ~70 MiB of transports, one diag_admit line per page, and would roll
     * that line out of the ring buffer. */
    {
        unsigned which[2] = {0, 1};
        pthread_t ta, tb;
        unsigned long main_slot = gp_slot();
        assert(!pthread_barrier_init(&slot_barrier, NULL, 3));
        assert(!pthread_create(&ta, NULL, slot_worker, &which[0]));
        assert(!pthread_create(&tb, NULL, slot_worker, &which[1]));
        pthread_barrier_wait(&slot_barrier);
        assert(!pthread_join(ta, NULL) && !pthread_join(tb, NULL));
        assert(!pthread_barrier_destroy(&slot_barrier));
        char *rstack = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(rstack != MAP_FAILED);
        void *rtp;
        __asm__ volatile("movc %0, tp" : "=r"(rtp));
        long rid = capstone_virtual_thread_create(raw_slot_worker, NULL, rstack + 65536, rtp);
        assert(rid > 0 && !capstone_virtual_thread_join(rid));
        assert(!munmap(rstack, 65536));
        unsigned long reused_max = 0;
        for (unsigned i = 0; i < 70; ++i) {
            unsigned long got = ~0UL;
            pthread_t t;
            assert(!pthread_create(&t, NULL, slot_once, &got) && !pthread_join(t, NULL));
            if (got > reused_max) reused_max = got;
        }
        stack_t alt = {.ss_sp = malloc(65536), .ss_size = 65536};
        struct sigaction sa = {.sa_handler = on_slot_signal, .sa_flags = SA_ONSTACK};
        assert(alt.ss_sp && !pipe(slot_pipe) && !sigaltstack(&alt, NULL));
        assert(!sigemptyset(&sa.sa_mask) && !sigaction(SIGUSR1, &sa, NULL));
        assert(!raise(SIGUSR1));
        char byte = 0;
        assert(read(slot_pipe[0], &byte, 1) == 1 && byte == 's');
        signal(SIGUSR1, SIG_DFL);
        stack_t off = {.ss_flags = SS_DISABLE};
        assert(!sigaltstack(&off, NULL));
        close(slot_pipe[0]); close(slot_pipe[1]); free(alt.ss_sp);
        printf("VIRTUAL_PTHREAD_SLOTS:main=%lu a=%lu b=%lu raw=%lu reused_max=%lu handler=%lu\n",
               main_slot, slot_seen[0], slot_seen[1], raw_slot, reused_max, handler_slot);
        if (main_slot == 0 && slot_seen[0] && slot_seen[1] && slot_seen[0] != slot_seen[1] && raw_slot &&
            !(slot_seen[0] % 16) && !(slot_seen[1] % 16) && !(raw_slot % 16) &&
            slot_seen[0] < 64 * 16 && slot_seen[1] < 64 * 16 && raw_slot < 64 * 16 &&
            reused_max && reused_max < 64 * 16 && handler_slot == main_slot && handler_wrote == 1)
            puts("VIRTUAL_PTHREAD_OK context_identity");
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
