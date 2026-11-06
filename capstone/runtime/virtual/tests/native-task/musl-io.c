/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <capstone/launch.h>
extern const struct capstone_launch_task *__capstone_launch_task(void);
extern long __syscall_cp(void *, void *, void *, void *, void *, void *, void *);
#ifdef MUSL_IO_DIRECT_TEST
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
#endif
static int failures, checks;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; printf("FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
/* LDC intentionally preserves a nonzero, untagged upper payload. */
static void *dirty(unsigned long low) {
    _Alignas(16) uint64_t words[2] = {low, 0x6961662031353d73ULL};
    void *value;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(value) : "r"(words) : "memory");
    return value;
}
static void scalar_regression(void) {
    char data[] = "computed length with dirty scalar cargo", copy[64] = {0};
    size_t n = strlen(data);
    int fd = open("scalar-data", O_CREAT | O_TRUNC | O_RDWR, 0600);
    CHECK(fd >= 0);
    CHECK(__syscall_cp((void *)SYS_write, dirty(fd), data, dirty(n), 0, 0, 0) == (long)n);
    CHECK(pread(fd, copy, n, 0) == (long)n && !memcmp(data, copy, n));
    CHECK(__syscall_cp((void *)SYS_pwrite64, dirty(fd), data, dirty(n), dirty(1), 0, 0) == (long)n);
    CHECK(lseek(fd, 0, SEEK_CUR) == (off_t)n);
#ifdef MUSL_IO_DIRECT_TEST
    if (__capstone_launch_task()->native.features == CAPSTONE_LAUNCH_NATIVE_IO) {
        CHECK(__capstone_native_syscall(SYS_write, dirty(fd), data, dirty(n), 0, 0, 0) == -EINVAL);
        CHECK(__builtin_capstone_cap_get_tag(data));
        CHECK(__syscall_cp((void *)SYS_write, data, data, dirty(n), 0, 0, 0) == -EINVAL);
        CHECK(__syscall_cp((void *)SYS_pwrite64, dirty(fd), data, dirty(n), data, 0, 0) == -EINVAL);
        CHECK(lseek(fd, 0, SEEK_CUR) == (off_t)n);
        CHECK(lseek(fd, 0, SEEK_END) == (off_t)n + 1);
    }
#endif
    CHECK(close(fd) == 0);
}
static unsigned char pattern(size_t i) { return (i * 17 + 23) % 251; }
int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "pipe")) {
        char data[64]; memset(data, 'S', sizeof data);
        CHECK(read(0, data, sizeof data) == 3);
        CHECK(!memcmp(data, "ABC", 3) && data[3] == 'S');
        errno = 0; CHECK(read(0, data, sizeof data) == -1 && errno == EAGAIN);
    } else {
        scalar_regression();
        size_t sizes[] = {0, 1, 4095, 4096, 65535, 65536, 65537, 262144, 1048577, 1300000};
        unsigned char *data = malloc(1300000), *copy = malloc(1300000);
        if (!data || !copy) return 2;
        for (size_t i = 0; i < 1300000; ++i) data[i] = pattern(i);
        for (unsigned s = 0; s < sizeof sizes/sizeof sizes[0]; ++s) {
            size_t n = sizes[s];
            int fd = open("io-data", O_CREAT | O_TRUNC | O_RDWR, 0600);
            if (fd < 0) return 3;
            printf("TRANSFER_BEGIN:%zu\n", n); fflush(stdout);
            CHECK(write(fd, data, n) == (ssize_t)n);
            CHECK(lseek(fd, 0, SEEK_CUR) == (off_t)n);
            CHECK(lseek(fd, 0, SEEK_SET) == 0);
            memset(copy, 'S', n);
            CHECK(read(fd, copy, n) == (ssize_t)n);
            CHECK(!memcmp(data, copy, n));
            printf("TRANSFER_END:%zu\n", n); fflush(stdout);
            CHECK(read(fd, copy, 1) == 0);
            if (n >= 4096) {
                CHECK(pread(fd, copy, 4096, 0) == 4096);
                CHECK(!memcmp(data, copy, 4096));
                CHECK(lseek(fd, 0, SEEK_CUR) == (off_t)n);
                CHECK(pwrite(fd, data, 4096, 0) == 4096);
                CHECK(lseek(fd, 0, SEEK_CUR) == (off_t)n);
                CHECK(pread(fd, copy, 64, n - 3) == 3);
                CHECK(!memcmp(copy, data + n - 3, 3));
                CHECK(lseek(fd, 0, SEEK_SET) == 0);
                unsigned long base = __builtin_capstone_cap_get_cursor(copy);
                void *small = __builtin_capstone_cap_shrink(copy, base, base + 32);
                errno = 0; CHECK(read(fd, small, 33) == -1 && errno == EFAULT);
                CHECK(lseek(fd, 0, SEEK_CUR) == 0);
                errno = 0; CHECK(write(fd, small, 33) == -1 && errno == EFAULT);
                CHECK(lseek(fd, 0, SEEK_CUR) == 0);
                errno = 0; CHECK(pread(fd, copy, 1, -1) == -1 && errno == EINVAL);
            }
            CHECK(close(fd) == 0);
        }
        errno = 0; CHECK(read(-1, copy, 1) == -1 && errno == EBADF);
        errno = 0; CHECK(syscall(SYS_write, -1, data, 1) == -1 && errno == EBADF);
        free(data); free(copy);
    }
    CHECK(getpid() > 0 && getpid() == getpid());
    printf("MUSL_IO checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
