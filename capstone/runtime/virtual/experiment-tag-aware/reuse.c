/* EXPERIMENT: whole-mapping retirement and untagged reuse on the raw-copy kernel. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void *volatile stale;
static unsigned long type(void *p)
{
    unsigned long t;
    __asm__ volatile(".insn r 0x5b, 1, 4, %0, %1, x1" : "=r"(t) : "r"(p));
    return t;
}

int main(int argc, char **argv)
{
    size_t bytes = 4096 * 64;
    void **p = mmap(0, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 2; }
    for (size_t i = 0; i < bytes / sizeof(void *); ++i) p[i] = p;
    stale = p;
    unsigned long old = __builtin_capstone_cap_get_cursor(p);
    if (munmap(p, bytes)) { perror("munmap"); return 2; }
    void **q = mmap(0, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q == MAP_FAILED) { perror("mmap reuse"); return 2; }
    unsigned long now = __builtin_capstone_cap_get_cursor(q);
    unsigned long bad = 0;
    for (size_t i = 0; i < bytes / sizeof(void *); ++i)
        if (type(q[i]) != 7) ++bad;
    printf("REUSE old=%lx new=%lx same_va=%d bad_tags=%lu\n", old, now, old == now, bad);
    fflush(stdout);
    if (bad) return 1;
    if (argc > 1 && !strcmp(argv[1], "stale")) {
        puts("STALE_AFTER_MUNMAP about to dereference");
        fflush(stdout);
        volatile uint64_t value = *(volatile uint64_t *)stale;
        printf("FAIL stale read returned %llu\n", (unsigned long long)value);
        return 1;
    }
    puts("REUSE PASS (physical-frame reuse requires external evidence)");
    return munmap(q, bytes) != 0;
}
