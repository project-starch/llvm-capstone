/* SPDX-License-Identifier: MIT */
/* A fully prefetched pipe is supplied by pipe-exec. Each process opens one
 * system-wide capprof window; its setup and diagnostic output are excluded. */
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef NATIVE_DIRECT
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static long input(void *p, unsigned long n)
{ return __capstone_native_syscall(63, 0, p, (void *)(__uintcap_t)n, 0, 0, 0); }
static long output(void *p, unsigned long n)
{ return __capstone_native_syscall(64, (void *)1, p, (void *)(__uintcap_t)n, 0, 0, 0); }
static long pid(void)
{ return __capstone_native_syscall(172, 0, 0, 0, 0, 0, 0); }
#else
static long input(void *p, unsigned long n) { return read(0, p, n); }
static long output(void *p, unsigned long n) { return write(1, p, n); }
#ifdef FORCE_EXCHANGE_PID
/* getpid() is normally answered from the SDK launch metadata. Force a real
 * exchange request only for the scalar transport microbenchmark. */
extern long __capstone_delegate_ints(uint64_t, uint64_t, uint64_t, uint64_t);
static long pid(void) { return __capstone_delegate_ints(172, 0, 0, 0); }
#else
static long pid(void) { return getpid(); }
#endif
#endif
int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) return 2;
    int writing = argc == 3 && !strcmp(argv[2], "write");
    unsigned bytes = atoi(argv[1]), calls = bytes == 65536 ? 1 : 100;
    char *buffer = malloc(65536);
    if (!buffer) return 3;
    if (writing) memset(buffer, 'X', 65536);
    /* Make output pages resident outside the window. Volatile prevents the
     * compiler deleting these stores before a read that overwrites them. */
    for (unsigned i = 0; i < 65536; i += 4096) ((volatile char *)buffer)[i] = writing ? 'X' : 0;
    ((volatile char *)buffer)[65535] = writing ? 'X' : 0;
    long expected = bytes ? bytes : pid();
    int failures = 0;
    if (bytes == 4097) {
        /* One page mapped and unmapped, never touched: the map service pair. */
        __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
        for (unsigned i = 0; i < calls; ++i) {
            void *p = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            failures += p == MAP_FAILED;
            if (p != MAP_FAILED) failures += munmap(p, 4096) != 0;
        }
        __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
        char text[128];
        int n = snprintf(text, sizeof(text), "BUDGET bytes=%u calls=%u failures=%d\n", bytes, calls, failures);
        write(1, text, n);
        return failures != 0;
    }
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    for (unsigned i = 0; i < calls; ++i)
        failures += (bytes ? (writing ? output(buffer, bytes) : input(buffer, bytes)) : pid()) != expected;
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    char text[128];
    int n = snprintf(text, sizeof(text), "BUDGET bytes=%u calls=%u failures=%d\n", bytes, calls, failures);
    write(1, text, n);
    free(buffer);
    return failures != 0;
}
