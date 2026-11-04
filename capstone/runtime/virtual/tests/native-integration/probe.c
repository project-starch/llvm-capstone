/* SPDX-License-Identifier: MIT */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <capstone/capability.h>

extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static long call(long nr, long fd, void *buf, long bytes)
{
    return __capstone_native_syscall(nr, (void *)(__uintcap_t)(unsigned long)fd, buf, (void *)(__uintcap_t)(unsigned long)bytes, 0, 0, 0);
}
static int checks, failures;
static void line(const char *text)
{
    write(1, text, strlen(text));
    write(1, "\n", 1);
}
#define CHECK(label, expr) do { ++checks; if (!(expr)) { ++failures; line("FAIL " label); } else line("PASS " label); } while (0)
int main(void)
{
    char buffer[64] = {0};
    struct timespec ts;
    struct { char data[32]; char sentinel[32]; } region;
    unsigned long base = __builtin_capstone_cap_get_cursor(region.data);
    char *bounded = __builtin_capstone_cap_shrink(region.data, base, base + 32);
    long pid = call(172, 0, 0, 0);
    CHECK("direct scalar", pid > 0 && pid == getpid());
    CHECK("direct input", call(63, 0, buffer, 1) == 1 && buffer[0] == 'Q');
    memset(buffer, 'S', sizeof(buffer));
    CHECK("forged scalar buffer", call(63, 0, (void *)0x1234, 1) == -EFAULT);
    CHECK("zero forged scalar", call(63, 0, (void *)0x1234, 0) == -EFAULT);
    CHECK("zero null", call(63, 0, 0, 0) == 0);
    CHECK("oversized bounds", call(63, 0, buffer, 1048577) == -E2BIG);
    CHECK("failed input sentinel", buffer[0] == 'S' && buffer[63] == 'S');
    memset(&region, 'S', sizeof(region));
    CHECK("tight bound refused", call(63, 0, bounded, 33) == -EFAULT);
    CHECK("bound sentinel intact", region.sentinel[0] == 'S');
    {
        union { void *cap; unsigned char bytes[16]; } tagged;
        tagged.cap = buffer;
        CHECK("tag before kernel write", __builtin_capstone_cap_get_tag(tagged.cap));
        CHECK("kernel byte write", call(63, 0, tagged.bytes, 1) == 1);
        CHECK("kernel write clears tag", !__builtin_capstone_cap_get_tag(tagged.cap));
    }
    {
        char *allocation = malloc(4096);
        CHECK("object allocation", allocation != NULL);
        if (allocation) {
            char *parent = capstone_cap_derive(allocation, 0, 4096);
            char *stale = capstone_cap_derive(parent, 0, 64);
            capstone_cap_revoke_child(parent, stale);
            char *replacement = capstone_cap_derive(parent, 0, 64);
            memset(replacement, 'S', 64);
            CHECK("revoked input refused", call(63, 0, stale, 1) == -EFAULT);
            CHECK("replacement sentinel intact", replacement[0] == 'S');
            free(allocation);
        }
    }
    CHECK("private descriptor", call(63, 3, buffer, 1) == -EBADF);
    CHECK("unsupported syscall", call(9999, 0, 0, 0) == -ENOSYS);
    CHECK("unsupported path", __capstone_native_syscall(56, (void *)-100L,
        (void *)"/proc/self/mem", 0, 0, 0, 0) == -ENOSYS);
    CHECK("clock output", call(113, CLOCK_MONOTONIC, &ts, 0) == 0 && ts.tv_sec >= 0);
    {
        char *large = malloc(65536);
        CHECK("large allocation", large != NULL);
        if (large) {
            CHECK("page chunked input", call(63, 0, large, 65536) == 65536);
            CHECK("page chunked contents", large[0] == 'Q' && large[65535] == '\n');
            free(large);
        }
    }
    CHECK("legacy checked write", write(1, "legacy\n", 7) == 7);
    CHECK("direct checked write", call(64, 1, "direct\n", 7) == 7);
    char summary[96];
    snprintf(summary, sizeof(summary), "native integration checks=%d failures=%d", checks, failures);
    line(summary);
    return failures != 0;
}
