/* Native binfmt contract; uses the existing SDK without ABI changes. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <signal.h>
#include <capstone/delegate.h>
#include <unistd.h>

/* Capture the actual initial transport at its assembly service boundary so
 * malformed-wire tests reach the module without the libc prevalidator. */
static struct capstone_delegate_entry *wire_meta;
static char *wire_exchange;
static size_t wire_bytes;
extern void __real___capstone_vm_delegate_transport(void *, void *, size_t);
void __wrap___capstone_vm_delegate_transport(void *meta, void *exchange, size_t bytes)
{
    wire_meta = meta; wire_exchange = exchange; wire_bytes = bytes;
    __real___capstone_vm_delegate_transport(meta, exchange, bytes);
}
static long raw(struct capstone_delegate_entry e)
{
    struct capstone_delegate_entry saved = *wire_meta;
    *wire_meta = e;
    __real___capstone_vm_delegate_transport(wire_meta, wire_exchange, wire_bytes);
    long result = wire_meta->result;
    *wire_meta = saved;
    return result;
}
static int wire_checks(void)
{
    if (!wire_meta || wire_bytes < 256) return 30;
    struct capstone_delegate_entry e = {.version = 2, .count = 1,
        .nr = CAPSTONE_SYS_newfstatat, .flags = 6,
        .args = {(uint64_t)-100, 16, 16, 0}};
    strcpy(wire_exchange + 16, "contract.dom");
    if (raw(e)) return 31;
    struct stat st;
    memcpy(&st, wire_exchange + 16, sizeof st);
    if (!S_ISREG(st.st_mode) || !st.st_size) return 32;
    e.nr = CAPSTONE_SYS_openat; e.flags = 0; e.args[2] = O_RDONLY;
    if (raw(e) != -EINVAL) return 33;
    e.nr = CAPSTONE_SYS_read; e.flags = 2; e.args[0] = 0;
    e.args[1] = wire_bytes - 1; e.args[2] = UINT64_MAX;
    if (raw(e) != -EFAULT) return 34;
    e.nr = CAPSTONE_SYS_ioctl; e.flags = 0; e.args[1] = 0x5413;
    if (raw(e) != -ENOSYS) return 35;
    e.nr = CAPSTONE_SYS_clone;
    if (raw(e) != -ENOSYS) return 36;
    e.nr = CAPSTONE_SYS_rt_sigaction;
    if (raw(e) != -ENOSYS) return 37;
    e.nr = CAPSTONE_SYS_exit_group; e.count = 0;
    if (raw(e) != -EINVAL) return 38;
    e.count = 1; e.flags = 64;
    return raw(e) != -EINVAL ? 39 : 0;
}

static int output(const char *p, size_t n)
{
    while (n) {
        ssize_t k = write(1, p, n);
        if (k <= 0) return 1;
        p += k; n -= k;
    }
    return 0;
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "hello")) return output("native hello\n", 13);
    if (!strcmp(argv[1], "exit")) return 37;
    if (!strcmp(argv[1], "loop")) {
        volatile unsigned long sum = 0;
        for (unsigned long i = 0; i < 100000; ++i) sum += i;
        return sum != 4999950000UL;
    }
    if (!strcmp(argv[1], "malloc")) {
        char *p = malloc(1U << 20);
        if (!p) return 3;
        for (unsigned i = 0; i < (1U << 20); ++i) p[i] = 'a' + i % 26;
        int rc = output(p, 1U << 20);
        free(p);
        return rc;
    }
    if (!strcmp(argv[1], "mmap")) {
        char *p = mmap(NULL, 1U << 20, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return 4;
        for (unsigned i = 0; i < (1U << 20); i += 4096) p[i] = i / 4096;
        for (unsigned i = 0; i < (1U << 20); i += 4096)
            if ((unsigned char)p[i] != i / 4096) return 5;
        return munmap(p, 1U << 20) ? 6 : 0;
    }
    if (!strcmp(argv[1], "read")) {
        char b[4];
        ssize_t n = read(0, b, sizeof b);
        return n != 4 || memcmp(b, "test", 4) || output(b, 4);
    }
    if (!strcmp(argv[1], "pipe")) {
        union { void *cap; int fd[2]; } ends;
        char buffer[64];
        ends.cap = buffer;
        if (pipe2(ends.fd, O_CLOEXEC | O_NONBLOCK)) return 40;
        if (__builtin_capstone_cap_get_tag(ends.cap)) return 41;
        if (!(fcntl(ends.fd[0], F_GETFD) & FD_CLOEXEC)) return 42;
        if (write(ends.fd[1], "pipe", 4) != 4 ||
            read(ends.fd[0], buffer, sizeof buffer) != 4 ||
            memcmp(buffer, "pipe", 4)) return 43;
        if (close(ends.fd[1]) || read(ends.fd[0], buffer, sizeof buffer) ||
            close(ends.fd[0])) return 44;
        errno = 0;
        if (read(ends.fd[0], buffer, 1) != -1 || errno != EBADF) return 45;
        errno = 0;
        return pipe2(ends.fd, -1) != -1 || errno != EINVAL;
    }
    if (!strcmp(argv[1], "protect-refuse")) {
        extern long __capstone_vm_protect(uintptr_t, size_t, int);
        char local;
        return __capstone_vm_protect(0, 4096, PROT_READ) != -EINVAL ||
               __capstone_vm_protect(UINTPTR_MAX - 4095, 8192, PROT_READ) != -EINVAL ||
               __capstone_vm_protect((uintptr_t)&local & ~(uintptr_t)4095, 4096, PROT_NONE) != -EINVAL;
    }
    if (!strcmp(argv[1], "bad-fd")) {
        char b;
        errno = 0;
        if (read(-1, &b, 1) != -1 || errno != EBADF) return 11;
        errno = 0;
        return write(-1, "x", 1) != -1 || errno != EBADF;
    }
    if (!strcmp(argv[1], "unsupported")) {
        errno = 0;
        pid_t pid = fork();
        return pid != -1 || errno != ENOSYS;
    }
    if (!strcmp(argv[1], "startup")) {
        char cwd[32];
        const char *v = getenv("DOM_TEST");
        if (!v || strcmp(v, "present")) return 9;
        if (getpid() < 1 || getppid() < 1) return 10;
        errno = 0;
        return !getcwd(cwd, sizeof cwd) || strcmp(cwd, "/tmp");
    }
    if (!strcmp(argv[1], "wire")) return wire_checks();
    if (!strcmp(argv[1], "files")) {
        int fd = open("/tmp/native-file", O_CREAT | O_TRUNC | O_RDWR, 0600);
        char b[12] = {0};
        struct stat st;
        struct iovec out[2] = {{"hello ", 6}, {"world", 5}};
        struct iovec in[2] = {{b, 3}, {b + 3, 8}};
        if (fd < 0 || writev(fd, out, 2) != 11 || fstat(fd, &st) || st.st_size != 11 ||
            lseek(fd, 0, SEEK_SET) || readv(fd, in, 2) != 11 || strcmp(b, "hello world")) return 20;
        if (pwrite(fd, "!", 1, 10) != 1 || pread(fd, b, 11, 0) != 11 ||
            strcmp(b, "hello worl!")) return 21;
        if (close(fd) || unlink("/tmp/native-file")) return 22;
        return 0;
    }
    if (!strcmp(argv[1], "private-fd")) {
        /* No extra descriptor has been granted to this domain. The private
         * context and any inherited descriptors must all remain inaccessible. */
        for (int fd = 3; fd < 32; ++fd) {
            errno = 0;
            if (fcntl(fd, F_GETFD) != -1 || errno != EBADF) return 23;
            errno = 0;
            if (close(fd) != -1 || errno != EBADF) return 24;
        }
        return 0;
    }
    if (!strcmp(argv[1], "signal-mask")) {
        sigset_t set, old;
        struct sigaction sa = {0};
        sigemptyset(&set); sigaddset(&set, SIGUSR2);
        if (sigprocmask(SIG_BLOCK, &set, &old) || sigprocmask(SIG_SETMASK, &old, &set) ||
            sigismember(&set, SIGUSR2) != 1) return 27;
        sa.sa_handler = SIG_IGN;
        if (sigaction(SIGUSR1, &sa, 0)) return 28;
        sa.sa_handler = SIG_DFL;
        return sigaction(SIGUSR1, &sa, 0) != 0;
    }
    if (!strcmp(argv[1], "spin")) {
        if (output("ready\n", 6)) return 7;
        for (;;) __asm__ volatile ("" ::: "memory");
    }
    return 8;
}
