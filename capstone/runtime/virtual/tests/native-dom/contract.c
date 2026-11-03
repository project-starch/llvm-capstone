/* Native binfmt day-one contract; uses the existing SDK without ABI changes. */
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <signal.h>
#include <unistd.h>

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
        char path[64], target[128];
        int found = -1;
        for (int fd = 0; fd < 32; ++fd) {
            /* Avoid stdio in this descriptor-policy probe. */
            strcpy(path, "/proc/self/fd/");
            unsigned n = strlen(path);
            if (fd >= 10) path[n++] = '0' + fd / 10;
            path[n++] = '0' + fd % 10; path[n] = 0;
            ssize_t len = readlink(path, target, sizeof target - 1);
            if (len < 0) continue;
            target[len] = 0;
            if (strstr(target, "capstone-dom")) { found = fd; break; }
        }
        if (found < 0) return 23;
        errno = 0;
        if (close(found) != -1 || errno != EBADF) return 24;
        errno = 0;
        if (dup(found) != -1 || errno != EBADF) return 25;
        errno = 0;
        if (dup2(0, found) != -1 || errno != EBADF) return 26;
        errno = 0;
        return fcntl(found, F_GETFD) != -1 || errno != EBADF;
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
