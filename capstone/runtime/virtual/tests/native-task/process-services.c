/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <capstone/delegate.h>

extern void __capstone_vm_delegate_transport(void *, void *, size_t);
#ifdef PROCESS_DIRECT
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static void *scalar(unsigned long n)
{
    void *r;
    __asm__ volatile("movc %0, zero\n\tmv %0, %1" : "=&r"(r) : "r"(n));
    return r;
}
static void *dirty(unsigned long n)
{
    _Alignas(16) uint64_t words[2] = {n, 0xabadcafe};
    void *r;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(r) : "r"(words) : "memory");
    return r;
}
#endif
static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "PROCESS_FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
static void exchange(char c)
{
    char reply = 0;
    CHECK(write(1, &c, 1) == 1);
    CHECK(read(0, &reply, 1) == 1 && reply == c);
}
static int make_pipe(int *p, unsigned flags)
{
#ifdef PROCESS_DIRECT
    long rc = __capstone_native_syscall(SYS_pipe2, p, scalar(flags), 0, 0, 0, 0);
    if (rc < 0) { errno = -rc; return -1; }
    return rc;
#else
    return pipe2(p, flags);
#endif
}
static long wire_pipe(void *meta, void *data, size_t offset, unsigned flags)
{
    struct capstone_delegate_entry *e = meta;
    memset(e, 0, sizeof *e);
    e->version = CAPSTONE_DELEGATE_VERSION;
    e->count = 1;
    e->nr = SYS_pipe2;
    e->args[0] = offset;
    e->args[1] = flags;
    e->flags = 1;
    __capstone_vm_delegate_transport(meta, data, 8192);
    return e->result;
}
static void close_pair(int *p)
{
    CHECK(close(p[0]) == 0);
    CHECK(close(p[1]) == 0);
}
static void handler(int sig) { (void)sig; _exit(98); }
static void signal_checks(void)
{
    struct sigaction a = {.sa_handler = SIG_IGN}, old;
    CHECK(sigemptyset(&a.sa_mask) == 0);
    CHECK(sigaction(SIGUSR1, &a, NULL) == 0);
    CHECK(sigaction(SIGUSR1, NULL, &old) == 0 && old.sa_handler == SIG_IGN);
    a.sa_handler = handler;
    errno = 0;
    CHECK(sigaction(SIGUSR1, &a, NULL) == -1 && errno == EOPNOTSUPP);
    CHECK(sigaction(SIGUSR1, NULL, &old) == 0 && old.sa_handler == SIG_IGN);
    a.sa_handler = SIG_IGN;
    CHECK(sigaction(SIGKILL, &a, NULL) == -1 && errno == EINVAL);
    CHECK(sigaction(SIGSTOP, &a, NULL) == -1 && errno == EINVAL);
    CHECK(sigaction(0, &a, NULL) == -1 && errno == EINVAL);
    exchange('I'); /* ordinary Linux sends SIGUSR1 before replying */
    CHECK(sigaction(SIGUSR1, NULL, &old) == 0 && old.sa_handler == SIG_IGN);
}
static void basic(void)
{
    int p[2], baseline[2];
    char data[8] = {0};
    CHECK(make_pipe(p, O_CLOEXEC | O_NONBLOCK) == 0);
    baseline[0] = p[0]; baseline[1] = p[1];
    CHECK(p[0] >= 3 && p[1] == p[0] + 1 && p[1] < 1024);
    CHECK(fcntl(p[0], F_GETFD) == FD_CLOEXEC && fcntl(p[1], F_GETFD) == FD_CLOEXEC);
    CHECK((fcntl(p[0], F_GETFL) & (O_ACCMODE | O_NONBLOCK)) == (O_RDONLY | O_NONBLOCK));
    CHECK((fcntl(p[1], F_GETFL) & (O_ACCMODE | O_NONBLOCK)) == (O_WRONLY | O_NONBLOCK));
    CHECK(write(p[1], "abc", 3) == 3);
    CHECK(read(p[0], data, sizeof data) == 3 && !memcmp(data, "abc", 3));
    CHECK(read(p[0], data, 1) == -1 && errno == EAGAIN);
    CHECK(close(p[1]) == 0 && read(p[0], data, 1) == 0 && close(p[0]) == 0);
    for (unsigned i = 0; i < 64; ++i) {
        p[0] = -901; p[1] = -902;
        CHECK(make_pipe(p, O_DIRECT) == -1 && errno == EINVAL);
        CHECK(p[0] == -901 && p[1] == -902);
    }
    unsigned long base = __builtin_capstone_cap_get_cursor(p);
    void *shortp = __builtin_capstone_cap_shrink(p, base, base + sizeof p - 1);
    void *readonly = __builtin_capstone_cap_tighten(p, 4);
    CHECK(make_pipe(shortp, 0) == -1 && errno == EFAULT);
    CHECK(make_pipe(readonly, 0) == -1 && errno == EFAULT);
#ifdef PROCESS_DIRECT
    CHECK(make_pipe(NULL, 0) == -1 && errno == EINVAL);
    CHECK(__capstone_native_syscall(SYS_pipe2, scalar(base), 0, 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(SYS_pipe2, p, p, 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(SYS_pipe2, p, dirty(0), 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(SYS_pipe2, p, 0, p, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(SYS_pipe2, p, scalar(1UL << 32), 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(0xc0de0005UL, scalar(SIGKILL), scalar(1), 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(0xc0de0005UL, scalar(SIGSTOP), scalar(1), 0, 0, 0, 0) == -EINVAL);
#else
    CHECK(make_pipe(NULL, 0) == -1 && errno == EFAULT);
#endif
    CHECK(make_pipe(p, 0) == 0 && p[0] == baseline[0] && p[1] == baseline[1]);
    CHECK(fcntl(p[0], F_GETFD) == 0 && fcntl(p[1], F_GETFD) == 0);
    CHECK(!(fcntl(p[0], F_GETFL) & O_NONBLOCK) && !(fcntl(p[1], F_GETFL) & O_NONBLOCK));
    struct sigaction ignore = {.sa_handler = SIG_IGN}, saved;
    CHECK(sigemptyset(&ignore.sa_mask) == 0 && sigaction(SIGPIPE, &ignore, &saved) == 0);
    CHECK(close(p[0]) == 0);
    CHECK(write(p[1], "x", 1) == -1 && errno == EPIPE);
    CHECK(close(p[1]) == 0 && sigaction(SIGPIPE, &saved, NULL) == 0);
    signal_checks();
}

static void rollback(void)
{
    char *meta = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *data = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(meta != MAP_FAILED && data != MAP_FAILED);
    if (meta == MAP_FAILED || data == MAP_FAILED) return;
    memset(meta, 0, 8192); memset(data, 0, 8192);
    int baseline[2];
    CHECK(make_pipe(baseline, 0) == 0);
    close_pair(baseline);
    int *partial = (int *)(data + 4092);
    for (int i = 0; i < 128; ++i) {
        partial[0] = -901; partial[1] = -902;
        CHECK(mprotect(data + 4096, 4096, PROT_NONE) == 0);
#ifdef PROCESS_DIRECT
        CHECK(make_pipe(partial, 0) == -1 && errno == EFAULT);
#else
        CHECK(wire_pipe(meta, data, 4092, 0) == -EFAULT);
#endif
        /* The first int proves reservation and partial result copying took
         * place. Its reservation must have been returned, without install. */
        CHECK(partial[0] == baseline[0]);
        CHECK(fcntl(baseline[0], F_GETFD) == -1 && errno == EBADF);
        CHECK(fcntl(baseline[1], F_GETFD) == -1 && errno == EBADF);
        CHECK(mprotect(data + 4096, 4096, PROT_READ | PROT_WRITE) == 0);
        CHECK(partial[1] == -902);
        int p[2];
        CHECK(make_pipe(p, 0) == 0 && p[0] == baseline[0] && p[1] == baseline[1]);
        close_pair(p);
    }
    CHECK(munmap(data, 8192) == 0 && munmap(meta, 8192) == 0);
}

static void exhaustion(int limit)
{
    int held[2048], count = 0, p[2];
    while (count + 2 <= (int)(sizeof held / sizeof held[0]) && make_pipe(p, 0) == 0) {
        held[count++] = p[0]; held[count++] = p[1];
    }
    CHECK(errno == EMFILE && count == ((limit - 3) / 2) * 2);
    if (count < 2) return;
    for (int i = 0; i < 32; ++i) {
        p[0] = -901; p[1] = -902;
        CHECK(make_pipe(p, 0) == -1 && errno == EMFILE);
        CHECK(p[0] == -901 && p[1] == -902);
    }
    /* Odd remaining capacity forces prepare's second reservation to fail.
     * The first reservation must be immediately reusable by fcntl. */
    int last = fcntl(held[0], F_DUPFD_CLOEXEC, limit - 1);
    CHECK(last == limit - 1 && close(last) == 0);
    int hole = held[0];
    CHECK(close(hole) == 0);
    held[0] = -1;
    CHECK(make_pipe(p, 0) == 0 && p[0] == hole && p[1] == limit - 1);
    close_pair(p);
    for (int i = 0; i < count; ++i) if (held[i] >= 0) CHECK(close(held[i]) == 0);
    CHECK(make_pipe(p, 0) == 0 && p[0] == 3 && p[1] == 4);
    close_pair(p);
}

static void bad_metadata(void)
{
    char *meta = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *data = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (meta == MAP_FAILED || data == MAP_FAILED) _exit(90);
    memset(meta, 0, 8192); memset(data, 0, 8192);
    if (mprotect(meta + 4096, 4096, PROT_READ)) _exit(91);
    /* Entry and result fit page zero, while the signal-count publication
     * faults in page one. The task must die before its next user instruction. */
    wire_pipe(meta, data, 8, 0);
    _exit(92);
}

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    if (!strcmp(argv[1], "basic")) basic();
    else if (!strcmp(argv[1], "rollback")) rollback();
    else if (!strcmp(argv[1], "limit") && argc == 3) exhaustion(atoi(argv[2]));
    else if (!strcmp(argv[1], "bad-meta")) bad_metadata();
    else if (!strcmp(argv[1], "default")) {
        struct sigaction a = {.sa_handler = SIG_DFL};
        CHECK(sigemptyset(&a.sa_mask) == 0 && sigaction(SIGUSR2, &a, NULL) == 0);
        exchange('D');
        CHECK(0); /* supervisor must terminate this task with SIGUSR2 */
    } else if (!strcmp(argv[1], "stop-kill")) {
        struct sigaction a = {.sa_handler = SIG_IGN};
        CHECK(sigemptyset(&a.sa_mask) == 0);
        CHECK(sigaction(SIGSTOP, &a, NULL) == -1 && errno == EINVAL);
        CHECK(sigaction(SIGKILL, &a, NULL) == -1 && errno == EINVAL);
        exchange('S'); exchange('K');
        CHECK(0);
    } else return 2;
    fprintf(stderr, "NATIVE_PROCESS mode=%s checks=%d failures=%d\n", argv[1], checks, failures);
    return failures != 0;
}
