/* Native binfmt day-one contract; uses the existing SDK without ABI changes. */
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
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
        int fd = open("/not-supported", O_RDONLY);
        return fd != -1 || errno != ENOSYS;
    }
    if (!strcmp(argv[1], "startup")) {
        char cwd[32];
        const char *v = getenv("DOM_TEST");
        if (!v || strcmp(v, "present")) return 9;
        if (getpid() < 1 || getppid() < 1) return 10;
        errno = 0;
        return getcwd(cwd, sizeof cwd) != NULL || errno != ENOSYS;
    }
    if (!strcmp(argv[1], "spin")) {
        if (output("ready\n", 6)) return 7;
        for (;;) __asm__ volatile ("" ::: "memory");
    }
    return 8;
}
