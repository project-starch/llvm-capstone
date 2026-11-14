/* gp-cursor directed test: linker gp relaxation (-capstone-mark-gp-relaxable, lld --relax-gp) rebases global
 * accesses on gp, so gp's cursor must be __global_pointer$ in every context, and the transport lock's identity no
 * longer lives in it (delegate.c, "THE OWNER'S IDENTITY"). Each check reads gp's cursor with movc and compares it
 * with __global_pointer$'s address (formed pc-relative, not from gp's cursor), and writes and reads small globals,
 * which the gp arm reaches through gp, against the same globals reached through their address. The checks:
 *   main_cursor, main_access          at main
 *   longjmp_cursor                    after a longjmp out of a nested call
 *   handler_cursor, nested_cursor     in a handler on the alternate signal stack, and in a handler nested in it
 *   handler_access                    small globals written in the handler, read in main
 *   handler_create_refused            (single mode) a context create inside a handler fails with EDEADLK, and
 *                                     pthread_create fails, leaving single-context mode (contract test 3)
 *   preempt_cursor                    across 4096 delegated calls, 256 fresh-page faults and a long spin
 *   single_mode                       no context created yet: identity mode 0 (contract test 1; the runner
 *                                     compares the service exits of this binary's N delegated calls with
 *                                     GP_MULTI's)
 * With -DGP_MULTI, a second context is created first, then:
 *   service_mode                      identity mode 1 after the first create
 *   child_cursor, child_access        in the second context
 *   contention                        two live contexts write 2048 records each, concurrently, through the
 *                                     transport; every record arrives whole and once (contract test 2)
 * Control: -DGP_MUTANT moves gp's cursor at main's start, where a skipped set point would leave it; every cursor
 * and access check must then fail. */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern char __global_pointer$[];
extern int __capstone_context_identity_mode(void);
extern long __capstone_virtual_context_create(void *(*)(void *), void *, void *, void *);

static unsigned long gp_cursor(void)
{
    void *gp;
    __asm__ volatile("movc %0, gp" : "=r"(gp));
    return __builtin_capstone_cap_get_cursor(gp);
}
static int checks, failures;
/* A setup call that fails is reported, never a silent exit: the run then reads as incomplete. */
static int setup_failed(const char *what)
{
    printf("GP_SETUP_FAIL:%s:errno=%d\n", what, errno);
    printf("GP_DONE:%d:%d\n", checks, failures + 1);
    return 2;
}
static void check(int ok, const char *name)
{
    ++checks;
    failures += !ok;
    printf("GP_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
}

/* Small data (8 bytes each): reached through gp in the gp arm. */
static volatile long sv_main, sv_handler, sv_child, sv_nested;
static volatile long *volatile addr_of[4];   /* the same globals, reached through their address */
/* Taken in a function of its own: a function that both takes a global's address and accesses it by name shares
   one gp-derived base between the two, and that base (bounded, stored) is not relaxable, so the by-name access
   would not be either. */
static void __attribute__((noinline)) take_addresses(void)
{
    addr_of[0] = &sv_main; addr_of[1] = &sv_handler; addr_of[2] = &sv_child; addr_of[3] = &sv_nested;
}
static void fence_rw(void) { __asm__ volatile("fence rw, rw" ::: "memory"); }
/* Write VAR by name (a folded, gp-relaxable access) and read it back through its address. */
#define ACCESS_OK(var, k, token) \
    ((var) = (token), fence_rw(), *addr_of[k] == (token) && (var) == (token))

static jmp_buf jb;
static void __attribute__((noinline)) jump_back(int depth)
{
    if (depth) jump_back(depth - 1);
    longjmp(jb, 1);
}

static volatile unsigned long handler_seen = 1, nested_seen = 1;
static volatile int handler_depth, refused_errno, pthread_refused, mode_in_handler = -1;
static void *never(void *a) { return a; }
static void on_usr1(int sig)
{
    (void)sig;
    if (handler_depth++ == 0) {
        handler_seen = gp_cursor();
        sv_handler = 0x4a11d1e5;
#ifndef GP_MULTI
        static char stack[16384] __attribute__((aligned(16)));
        void *tp;
        __asm__ volatile("movc %0, tp" : "=r"(tp));
        errno = 0;
        long id = __capstone_virtual_context_create(never, NULL, stack + sizeof stack, tp);
        refused_errno = id < 0 ? errno : -1;
        pthread_t t;
        pthread_refused = pthread_create(&t, NULL, never, NULL) != 0;
        mode_in_handler = __capstone_context_identity_mode();
#endif
        raise(SIGUSR1);   /* SA_NODEFER: a nested handler */
    } else {
        nested_seen = gp_cursor();
        sv_nested = 0x6e57ed;
    }
    --handler_depth;
}

#ifdef GP_MULTI
static volatile unsigned long child_seen = 1;
static volatile long child_access_ok;
static int pipefd[2];
static void *child(void *arg)
{
    child_seen = gp_cursor();
    child_access_ok = ACCESS_OK(sv_child, 2, 0xc41d);
    unsigned id = *(unsigned *)arg;
    for (unsigned i = 0; i < 2048; ++i) {
        unsigned long rec[4] = {id, i, id ^ i, 0x9e3779b97f4a7c15ul * (id * 4096 + i)};
        if (write(pipefd[1], rec, sizeof rec) != sizeof rec) return NULL;
    }
    return arg;
}
static void *drain(void *arg)
{
    static unsigned char seen[2][2048];
    long *bad = arg;
    for (unsigned n = 0; n < 2 * 2048; ++n) {
        unsigned long rec[4];
        size_t got = 0;
        while (got < sizeof rec) {
            ssize_t r = read(pipefd[0], (char *)rec + got, sizeof rec - got);
            if (r <= 0) { ++*bad; return arg; }
            got += r;
        }
        if (rec[0] > 1 || rec[1] >= 2048 || rec[2] != (rec[0] ^ rec[1]) ||
            rec[3] != 0x9e3779b97f4a7c15ul * (rec[0] * 4096 + rec[1]) || seen[rec[0]][rec[1]]++)
            ++*bad;
    }
    return arg;
}
#endif

int main(void)
{
    take_addresses();
    unsigned long G = __builtin_capstone_cap_get_cursor(__global_pointer$);
#ifdef GP_MUTANT
    {   /* where a context without its set point would have gp's cursor: the image start */
        void *gp;
        __asm__ volatile("movc %0, gp" : "=r"(gp));
        unsigned long base = __builtin_capstone_cap_get_base(gp);
        __asm__ volatile("scc gp, gp, %0" : : "r"(base) : "memory");
    }
#endif
    printf("GP_INFO:G=%#lx:cursor=%#lx\n", G, gp_cursor());
    check(gp_cursor() == G, "main_cursor");
    check(ACCESS_OK(sv_main, 0, 0x6a1), "main_access");

    if (!setjmp(jb)) jump_back(8);
    check(gp_cursor() == G, "longjmp_cursor");

#ifdef GP_MULTI
    pthread_t a, b, d;
    unsigned ids[2] = {0, 1};
    long bad = 0;
    if (pipe(pipefd)) return setup_failed("pipe");
    int pc = pthread_create(&a, NULL, child, &ids[0]);
    /* The first create switches the identity mode even when the context itself is then refused. */
    check(__capstone_context_identity_mode() == 1, "service_mode");
    if (pc == EAGAIN) {
        /* The native route refuses further contexts (the create service fails; musl reports EAGAIN), so no
           second context can exist there: the context checks do not apply. */
        printf("GP_NA:child_cursor,child_access,contention:further contexts refused (EAGAIN)\n");
    } else if (pc) {
        errno = pc; return setup_failed("pthread_create");
    } else {
        if ((pc = pthread_create(&b, NULL, child, &ids[1])) || (pc = pthread_create(&d, NULL, drain, &bad))) {
            errno = pc; return setup_failed("pthread_create 2");
        }
        pthread_join(a, NULL); pthread_join(b, NULL); pthread_join(d, NULL);
        check(child_seen == G, "child_cursor");
        check(child_access_ok == 1, "child_access");
        check(bad == 0, "contention");
    }
#endif

    static char alt[65536] __attribute__((aligned(16)));
    stack_t ss = {.ss_sp = alt, .ss_size = sizeof alt};
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_usr1;
    sa.sa_flags = SA_ONSTACK | SA_NODEFER;
    if (sigaltstack(&ss, NULL)) return setup_failed("sigaltstack");
    if (sigaction(SIGUSR1, &sa, NULL)) {
        /* The native route's kernel refuses caught handlers (capstone_native_signal_action: class 2 is
           EOPNOTSUPP), so no handler context can exist there: the handler checks do not apply. Any other
           failure is a failure. */
        if (errno != EOPNOTSUPP) return setup_failed("sigaction");
        printf("GP_NA:handler_cursor,nested_cursor,handler_access,handler_create_refused:caught handlers refused (EOPNOTSUPP)\n");
    } else {
        raise(SIGUSR1);
        check(handler_seen == G, "handler_cursor");
        check(nested_seen == G, "nested_cursor");
        check(*addr_of[1] == 0x4a11d1e5 && *addr_of[3] == 0x6e57ed, "handler_access");
#ifndef GP_MULTI
        printf("GP_INFO:refused_errno=%d:pthread_refused=%d:mode_in_handler=%d\n", refused_errno, pthread_refused,
               mode_in_handler);
        check(refused_errno == EDEADLK && pthread_refused && mode_in_handler == 0 &&
              __capstone_context_identity_mode() == 0, "handler_create_refused");
#endif
    }

    /* Escapes: delegated calls (a file in the run directory, which every route's grant covers; /dev/null needs
       the native route's dom_null), fresh-page faults, and a long spin across timer preemptions. The cursor is
       read after each; the environment's own failures are setup failures, not cursor results. */
    int fd = open("gp-preempt.tmp", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return setup_failed("open gp-preempt.tmp");
    int ok = 1;
    for (unsigned i = 0; i < 4096; ++i) {
        if (write(fd, "x", 1) != 1) { close(fd); return setup_failed("write"); }
        if (!(i & 255)) ok &= gp_cursor() == G;
    }
    close(fd); unlink("gp-preempt.tmp");
    unsigned char *pages = mmap(NULL, 256 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) return setup_failed("mmap");
    for (unsigned i = 0; i < 256; ++i) { pages[i * 4096] = (unsigned char)i; ok &= gp_cursor() == G; }
    volatile unsigned long spin = 0;
    for (unsigned long i = 0; i < 200000000ul; ++i) {
        spin += i;
        if (!(i & 0xffffff)) ok &= gp_cursor() == G;
    }
    check(ok && gp_cursor() == G && ACCESS_OK(sv_main, 0, 0x7e1), "preempt_cursor");
#ifndef GP_MULTI
    check(__capstone_context_identity_mode() == 0, "single_mode");
#endif
    printf("GP_DONE:%d:%d\n", checks, failures);
    return failures != 0;
}
