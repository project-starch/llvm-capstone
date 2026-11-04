/* SPDX-License-Identifier: MIT */
/* A fully prefetched pipe is supplied by pipe-exec. Each process opens one
 * system-wide capprof window; its setup and diagnostic output are excluded. */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifdef NATIVE_DIRECT
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static long input(void *p, unsigned long n)
{ return __capstone_native_syscall(63, 0, p, (void *)(__uintcap_t)n, 0, 0, 0); }
static long pid(void)
{ return __capstone_native_syscall(172, 0, 0, 0, 0, 0, 0); }
#else
static long input(void *p, unsigned long n) { return read(0, p, n); }
static long pid(void) { return getpid(); }
#endif
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    unsigned bytes = atoi(argv[1]), calls = bytes == 65536 ? 1 : 100;
    char *buffer = malloc(65536);
    if (!buffer) return 3;
    /* Make output pages resident outside the window. Volatile prevents the
     * compiler deleting these stores before a read that overwrites them. */
    for (unsigned i = 0; i < 65536; i += 4096) ((volatile char *)buffer)[i] = 0;
    ((volatile char *)buffer)[65535] = 0;
    long expected = bytes ? bytes : pid();
    int failures = 0;
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    for (unsigned i = 0; i < calls; ++i)
        failures += (bytes ? input(buffer, bytes) : pid()) != expected;
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    char text[128];
    int n = snprintf(text, sizeof(text), "BUDGET bytes=%u calls=%u failures=%d\n", bytes, calls, failures);
    write(1, text, n);
    free(buffer);
    return failures != 0;
}
