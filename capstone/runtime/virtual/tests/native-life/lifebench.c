/* SPDX-License-Identifier: MIT */
/* lifebench: attempts per lifecycle event. Built with the plain rv64 compiler and
 * run in a plain guest under the capprof plugin, one window per invocation, as
 * lqmicro does (the START/STOP markers are the plugin's). Two families:
 *   plain*  real Linux events (fork, exit, signal, ptrace, futex, thread)
 *   lib*    the lifecycle library on a synthetic namespace (software model)
 * The lib* numbers are the instruction cost of the library alone, NOT a kernel
 * path. A native kernel path would add the matching plain event's Linux work
 * (the mm copy, the signal frame write, the wakeup) to the library cost.
 *
 *   plainfork K R          R forks of a process padded to ~K resident pages; the child sleeps
 *   plainforkexit K R      R x (fork, child exits, parent waits)
 *   plainsig R             R x (raise a caught signal and return from its handler)
 *   plainptrace R          R x PTRACE_PEEKDATA of a stopped child
 *   plainfutexwake R       R x FUTEX_WAKE with no waiter
 *   plainfutexwait R       R x FUTEX_WAIT with a mismatched value
 *   plainthread R          R x (pthread_create, join)
 *   libfork K NODES R      R x nl_task_fork of a namespace with K pages and NODES live ids
 *   libexit K ARENAS R     R x teardown of a namespace with K pages and ARENAS arenas
 *   libsig R               R x (nl_sig_deliver, nl_sig_return)
 *   libptrace R            R x nl_ptrace_peek
 *   libfutexwake R         R x nl_futex_wake with no waiter
 *   libfutexwait R         R x nl_futex_wait with a mismatched value
 *   libthread R            R x (nl_thread_create, nl_task_exit)
 *   noop | close           exit at once | the stop marker alone */
#define _GNU_SOURCE
#include <capstone/native-life.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifdef __riscv
#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#else /* host sanity builds only */
#define START() asm volatile("" ::: "memory")
#define STOP()  asm volatile("" ::: "memory")
#endif
#define PG 4096UL

static void die(const char *what) { perror(what); exit(2); }
static long resident_pages(void)
{
    long size = 0, res = 0;
    FILE *f = fopen("/proc/self/statm", "r");
    if (f) { if (fscanf(f, "%ld %ld", &size, &res) != 2) res = 0; fclose(f); }
    return res;
}
/* pad the resident set to about K pages with touched anonymous memory */
static void pad_to(long k)
{
    long have = resident_pages();
    if (have >= k) return;
    size_t bytes = (size_t)(k - have) * PG;
    volatile char *p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap");
    for (size_t i = 0; i < bytes; i += PG) p[i] = 1;
}
static void report(const char *mode, long a, long b, long r, long extra)
{
    printf("LQ plain life %s a=%ld b=%ld rounds=%ld extra=%ld\n", mode, a, b, r, extra);
}

/* ---- plain Linux events ---- */
static int plainfork(long k, long rounds, int exits)
{
    pid_t kids[64];
    if (rounds > 64) rounds = 64;
    pad_to(k);
    long res = resident_pages();
    START();
    for (long r = 0; r < rounds; r++) {
        pid_t pid = fork();
        if (pid < 0) die("fork");
        if (pid == 0) {
            if (exits) _exit(0);
            struct timespec t = {3, 0};
            nanosleep(&t, 0);
            _exit(0);
        }
        kids[r] = pid;
        if (exits) { int st; waitpid(pid, &st, 0); }
    }
    STOP();
    for (long r = 0; r < rounds; r++) if (!exits) { kill(kids[r], SIGKILL); waitpid(kids[r], 0, 0); }
    report(exits ? "plainforkexit" : "plainfork", k, 0, rounds, res);
    return 0;
}
static volatile sig_atomic_t hits;
static void handler(int s) { (void)s; ++hits; }
static int plainsig(long rounds)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    sigaction(SIGUSR1, &sa, 0);
    START();
    for (long r = 0; r < rounds; r++) raise(SIGUSR1);
    STOP();
    report("plainsig", 0, 0, rounds, hits);
    return 0;
}
static int plainptrace(long rounds)
{
    static long word = 0x1234;
    pid_t pid = fork();
    long sum = 0;
    if (pid < 0) die("fork");
    if (pid == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        raise(SIGSTOP);
        _exit(0);
    }
    int st;
    waitpid(pid, &st, 0);
    START();
    for (long r = 0; r < rounds; r++) sum += ptrace(PTRACE_PEEKDATA, pid, &word, 0);
    STOP();
    kill(pid, SIGKILL); waitpid(pid, &st, 0);
    report("plainptrace", 0, 0, rounds, sum != 0);
    return 0;
}
static int plainfutex(long rounds, int wake)
{
    static volatile int word = 1;
    long ok = 0;
    START();
    for (long r = 0; r < rounds; r++)
        ok += wake ? syscall(SYS_futex, &word, 1 /*WAKE|PRIVATE=129*/ | 128, 1, 0, 0, 0)
                   : syscall(SYS_futex, &word, 0 /*WAIT*/ | 128, 0, 0, 0, 0);
    STOP();
    report(wake ? "plainfutexwake" : "plainfutexwait", 0, 0, rounds, ok);
    return 0;
}
static void *thread_main(void *p) { return p; }
static int plainthread(long rounds)
{
    START();
    for (long r = 0; r < rounds; r++) {
        pthread_t t;
        if (pthread_create(&t, 0, thread_main, 0)) die("pthread_create");
        pthread_join(t, 0);
    }
    STOP();
    report("plainthread", 0, 0, rounds, 0);
    return 0;
}

/* ---- library events ---- */
static struct nl_world *world(void)
{
    struct nl_world *w = nl_world_new();
    if (!w || cn_engine_create(CN_BACKEND_IOCTL, &nl_cn_ops, w, &w->engine)) die("world");
    return w;
}
static cn_u64 va(const struct nl_ns *ns, unsigned page, unsigned off) { return ns->base_va + (cn_u64)page * PG + off; }
static struct nl_cap mint_cap(struct nl_ns *ns, unsigned page, unsigned pages, unsigned rights)
{
    struct nl_cap c;
    if (nl_mint(ns, va(ns, page, 0), (cn_u64)pages * PG, rights, &c)) die("mint");
    return c;
}
static struct nl_task *mk(struct nl_world *w, struct nl_ns *ns, cn_u64 tid, struct cn_frame *f)
{
    struct nl_task *t = nl_task_new(w, ns, tid, 1000, f);
    if (!t) die("task");
    return t;
}
static int libfork(long k, long nodes, long rounds)
{
    struct nl_world *w = world();
    unsigned cap = nodes * 2 + 64 < 4096 ? 4096 : nodes * 2 + 64;
    struct nl_ns *A = nl_ns_new(w, 1, cap, k);
    struct nl_cap c0, *all = calloc(nodes + 1, sizeof(*all));
    struct cn_frame f;
    struct nl_task *P, *C;
    if (!A || !all) die("ns");
    c0 = mint_cap(A, 0, 1, 7);
    for (long i = 0; i < nodes; i++) all[i] = mint_cap(A, (unsigned)(i % k), 1, 7);
    for (long i = 0; i < nodes; i += 7) if (nl_retire(A, &all[i])) die("retire");   /* a non-empty free list */
    nl_frame_init(&f, A, 0x8000); nl_reg_from_cap(&f.x[10], &c0);
    P = mk(w, A, 10, &f);
    START();
    for (long r = 0; r < rounds; r++)
        if (nl_task_fork(P, 11 + r, 2 + r, NULL, &C)) die("fork");
    STOP();
    report("libfork", k, nodes, rounds, A->table->next_unused);
    return 0;
}
static int libexit(long k, long arenas, long rounds)
{
    struct nl_world *w = world();
    struct nl_ns *ns[40];
    struct nl_cap c;
    if (rounds > 40) rounds = 40;
    for (long r = 0; r < rounds; r++) {
        ns[r] = nl_ns_new(w, 1 + r, 4096, k);
        if (!ns[r]) die("ns");
        c = mint_cap(ns[r], 0, 1, 7);
        for (long i = 0; i < arenas; i++) nl_arena_add(ns[r], va(ns[r], 0, 0), 1, &c);
        if (nl_store_cap(ns[r], va(ns[r], k > 1 ? 1 : 0, 0), &c)) die("store");
        ns[r]->refs = 1;
    }
    START();
    for (long r = 0; r < rounds; r++) if (nl_ns_put(ns[r])) die("teardown");
    STOP();
    report("libexit", k, arenas, rounds, ns[0]->stats.retire_calls);
    return 0;
}
static int libsig(long rounds)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_cap d0 = mint_cap(A, 0, 1, 7), code = mint_cap(A, 2, 1, NL_RIGHT_X | NL_RIGHT_R);
    struct cn_frame f;
    struct nl_task *T;
    struct nl_sigctx u;
    struct cn_signal_token tk;
    nl_frame_init(&f, A, 0x8000); nl_reg_from_cap(&f.x[10], &d0); nl_reg_from_cap(&f.x[12], &code);
    T = mk(w, A, 10, &f);
    if (nl_sigaction(T, 10, 12, 0)) die("sigaction");
    START();
    for (long r = 0; r < rounds; r++) {
        if (nl_sig_deliver(T, 10, &u, &tk)) die("deliver");
        if (nl_sig_return(T, &u)) die("return");
    }
    STOP();
    report("libsig", 0, 0, rounds, 0);
    return 0;
}
static int libptrace(long rounds)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_actor dbg = {1000, 0};
    struct nl_task *T = mk(w, A, 10, NULL);
    cn_u64 v = 0, sum = 0;
    T->stopped = 1;
    nl_store_u64(A, va(A, 1, 0), 0x1234);
    START();
    for (long r = 0; r < rounds; r++) { if (nl_ptrace_peek(&dbg, T, va(A, 1, 0), &v)) die("peek"); sum += v; }
    STOP();
    report("libptrace", 0, 0, rounds, sum != 0);
    return 0;
}
static int libfutex(long rounds, int wake)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_cap word = mint_cap(A, 0, 1, 7);
    struct cn_frame f;
    struct nl_task *T;
    int blocked = 0;
    long ok = 0;
    nl_frame_init(&f, A, 0x8000); nl_reg_from_cap(&f.x[10], &word);
    T = mk(w, A, 10, &f);
    nl_store_u64(A, va(A, 0, 0), 1);
    START();
    for (long r = 0; r < rounds; r++)
        ok += wake ? nl_futex_wake(T, 10, 0, 1) : nl_futex_wait(T, 10, 0, 0, &blocked);
    STOP();
    report(wake ? "libfutexwake" : "libfutexwait", 0, 0, rounds, ok);
    return 0;
}
static int libthread(long rounds)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_cap st = mint_cap(A, 0, 1, 7), tls = mint_cap(A, 1, 1, 7);
    struct cn_frame f;
    struct nl_task *P, *T;
    nl_frame_init(&f, A, 0x8000); nl_reg_from_cap(&f.x[10], &st); nl_reg_from_cap(&f.x[11], &tls);
    P = mk(w, A, 10, &f);
    START();
    for (long r = 0; r < rounds; r++) {
        if (nl_thread_create(P, NL_CLONE_SETTLS, 10, 11, 0, 0, &T)) die("thread");
        nl_task_exit(T);
    }
    STOP();
    report("libthread", 0, 0, rounds, 0);
    return 0;
}

int main(int argc, char **argv)
{
    const char *m = argc > 1 ? argv[1] : "";
    long a = argc > 2 ? atol(argv[2]) : 0, b = argc > 3 ? atol(argv[3]) : 0, c = argc > 4 ? atol(argv[4]) : 0;
    if (!strcmp(m, "noop")) return 0;
    if (!strcmp(m, "close")) { STOP(); return 0; }
    if (!strcmp(m, "plainfork")) return plainfork(a, b, 0);
    if (!strcmp(m, "plainforkexit")) return plainfork(a, b, 1);
    if (!strcmp(m, "plainsig")) return plainsig(a);
    if (!strcmp(m, "plainptrace")) return plainptrace(a);
    if (!strcmp(m, "plainfutexwake")) return plainfutex(a, 1);
    if (!strcmp(m, "plainfutexwait")) return plainfutex(a, 0);
    if (!strcmp(m, "plainthread")) return plainthread(a);
    if (!strcmp(m, "libfork")) return libfork(a, b, c);
    if (!strcmp(m, "libexit")) return libexit(a, b, c);
    if (!strcmp(m, "libsig")) return libsig(a);
    if (!strcmp(m, "libptrace")) return libptrace(a);
    if (!strcmp(m, "libfutexwake")) return libfutex(a, 1);
    if (!strcmp(m, "libfutexwait")) return libfutex(a, 0);
    if (!strcmp(m, "libthread")) return libthread(a);
    fprintf(stderr, "unknown mode\n");
    return 2;
}
