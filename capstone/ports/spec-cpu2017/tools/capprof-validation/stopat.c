/* capprof work-bounded window test (stopat-run.sh, stopatcheck.py). stopat_f is called 1200 times,
 * 1000 of them inside two windows; its body and the loop around the call carry instructions with
 * unique immediates, so the window files show them by their bytes alone:
 *   stopat_f   addi t3,t3,0x3d1 (its first instruction: the stop site) / addi t3,t3,0x3d2
 *   run loop   addi t3,t3,0x3e2 before the call / addi t3,t3,0x3e3 after it
 *   main       0x3e1 / 0x3e4 around window 0's loop, 0x3e5 / 0x3e6 around window 1's
 * main: run(100) START run(300) STOP run(100) START run(700) STOP. Entries inside windows are
 * 1..300 in window 0 and 301..1000 in window 1; the 100 calls before each window count nothing.
 *
 * `stopat fault`: START, 64 calls of stopat_g(page i of a fresh mapping) with addi t3 0x3f2 before
 * and 0x3f3 after each call, STOP. stopat_g's first instruction is a store that faults on its
 * fresh page and is retried, and it is the last instruction of its page, so its block ends at it
 * and the instruction after it (ret, the next page) is in another block: an arrival that faults
 * must count as one entry with its retry.
 *
 * `stopat inter N`: a 2 s sleep, then N calls of stopat_f with a 1 ms sleep after each, no markers (a
 * window opened around two such processes holds both; two domains run one after the other on a
 * one-slot platform, see stopat-run.sh). Built with -DSTOPAT_BIG it carries a 3 MiB .bss,
 * so its domain grant (4 MiB, not 512 KiB), and with it the launcher's base for it, differs while
 * its site's and anchor's offsets do not (stopatcheck.py checks both).
 *
 * `stopat handler` (plain builds, -DSTOPAT_PLAIN): START, 10 calls of stopat_h(page) with addi t3
 * 0x3f4 before and 0x3f5 after each, STOP. stopat_h's first instruction stores to a PROT_NONE page;
 * the SIGSEGV handler first runs stopat_h's second instruction (addi t3 0x3f6, then ret) by a call
 * of its own, then opens the page, and the store is retried: site + len is reached once by another
 * path and once straight from the site, and only the second is an entry. */
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef STOPAT_PLAIN
#include <signal.h>
#include <sys/mman.h>
#endif
#ifdef STOPAT_BIG
char stopat_big[3 << 20];
#endif
#ifdef STOPAT_ENTRY
/* -Wl,-e,stopat_entry: the program's entry (the anchor), a site in the anchor's own first block,
 * its successor, then the usual start. That block is translated before the anchor has run. */
__asm__(".text\n.globl stopat_entry\n.type stopat_entry,@function\nstopat_entry:\n addi t3, t3, 0x3f7\n"
        ".globl stopat_e\n.type stopat_e,@function\nstopat_e:\n addi t3, t3, 0x3f8\n addi t3, t3, 0x3f9\n"
        " j _start\n");
#endif

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define MARK(imm) asm volatile("addi t3, t3, " #imm ::: "t3", "memory")

__attribute__((noinline)) long stopat_f(long x) {
    MARK(0x3d1); MARK(0x3d2);
    return x * 3 + 1;
}

__asm__(".text\n.balign 4096\n.skip 4092\n.globl stopat_g\n.type stopat_g,@function\n"
        "stopat_g:\n sd zero, 0(a0)\n ret\n.size stopat_g, .-stopat_g\n");
void stopat_g(char *p);

#ifdef STOPAT_PLAIN
__asm__(".text\n.balign 64\n.globl stopat_h\n.type stopat_h,@function\n"
        "stopat_h:\n sd zero, 0(a0)\n addi t3, t3, 0x3f6\n ret\n.size stopat_h, .-stopat_h\n");
void stopat_h(char *p);
static char *hpage;
static void on_segv(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)si; (void)uc;
    ((void (*)(void))((char *)stopat_h + 4))();   /* site + len, by a path that is not the site */
    mprotect(hpage, 4096, PROT_READ | PROT_WRITE);
}
#endif

__attribute__((noinline)) long run(long n, long acc) {
    for (long i = 0; i < n; i++) {
        MARK(0x3e2);
        acc = stopat_f(acc);
        MARK(0x3e3);
    }
    return acc;
}

int main(int argc, char **argv) {
    if (argc > 2 && argv[1][0] == 'i') {
        long n = atol(argv[2]), acc = 0;
        sleep(2);   /* both processes have started (and anchored) before either loops */
        for (long i = 0; i < n; i++) {
            acc = stopat_f(acc);
            usleep(1000);   /* the other process runs while this one sleeps: arrivals alternate */
        }
#ifdef STOPAT_BIG
        acc += ((volatile char *)stopat_big)[n];   /* keeps the array, and with it the grant size */
#endif
        printf("%ld\n", acc);
        return 0;
    }
#ifdef STOPAT_PLAIN
    if (argc > 1 && argv[1][0] == 'h') {
        hpage = mmap(0, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        struct sigaction sa = { 0 };
        sa.sa_sigaction = on_segv;
        sa.sa_flags = SA_SIGINFO | SA_NODEFER;
        if (hpage == MAP_FAILED || sigaction(SIGSEGV, &sa, 0)) return 1;
        START();
        for (int i = 0; i < 10; i++) {
            mprotect(hpage, 4096, PROT_NONE);
            MARK(0x3f4);
            stopat_h(hpage);
            MARK(0x3f5);
        }
        STOP();
        return 0;
    }
#endif
    if (argc > 1) {
        long pages = 64;
        char *p = malloc(pages * 4096 + 4096);
        if (!p) return 1;
        char *q = p + ((4096 - ((unsigned long)p & 4095)) & 4095);  /* keeps p's capability */
        START();
        for (long i = 0; i < pages; i++) {
            MARK(0x3f2);
            stopat_g(q + i * 4096);
            MARK(0x3f3);
        }
        STOP();
        printf("%d\n", q[8]);
        return 0;
    }
    long acc = run(100, 0);
    START(); MARK(0x3e1); acc = run(300, acc); MARK(0x3e4); STOP();
    acc = run(100, acc);
    START(); MARK(0x3e5); acc = run(700, acc); MARK(0x3e6); STOP();
    printf("%ld\n", acc);
    return 0;
}
