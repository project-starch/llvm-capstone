/* Negative tests of the child-frame protection (S-21), modes:
 *   multipage  a frame inside a 2-page heap arena            -> create must fail -EINVAL (find_arena exact match)
 *   twice      the same frame created twice                  -> the 2nd create's retire fails -> -EACCES + must-fire line
 *   copy       a well-formed frame, then read our own copy   -> create succeeds, the read faults cause 25
 *   write1024  a well-formed frame, then WRITE our own copy at frame+1024.. (the S caller's snapshot that
 *              every escape restores on the RTL, S-21) before the child runs: with fix (a) the write
 *              faults cause 25; without it the write lands and the child still reaches its exit ECALL
 *              (QEMU keeps the caller internally, so the overwrite itself has no effect there)
 *              MEASURED 2026-10-26: does NOT separate the modules. Its child exits at once, the launcher then
 *              unmaps the frame (exec.c, after CV_THREAD_EXIT), and that unmap's retire kills the copy on a
 *              module without fix (a) too. `copy` races the same exit.
 *   prewrite   the control: write our copy at frame+1024 with no create -> must land
 *   write-live a child that never exits, then write our copy at frame+1024 while it lives: with fix (a) the
 *              write faults cause 25; without it the write lands (the S-21 window, create to exit)
 *   copy-exited  `copy` after the child has EXITED (flag seen, then a 0.5 s sleep and a bounded spin): the
 *              launcher's exit unmap retires the frame on any module, so the read faults cause 25 with or
 *              without fix (a). A SURVIVED read on a module without the retire means the exit unmap did not
 *              retire the frame (or the wait was too short: dev's no-retire kit is the control that it is not)
 * The frame is filled exactly as runtime/virtual/thread.c fills it. Prints FRAME_TEST lines; a survived read exits 1. */
#include <capstone/capability.h>
#include <capstone/virtual.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "vm.h"

extern long __capstone_vm_thread_create_frame(void *frame);

static void *child(void *unused) { (void)unused; capstone_virtual_thread_exit(NULL); }
static volatile int child_ran;
static void *child_flag(void *unused) { (void)unused; child_ran = 1; capstone_virtual_thread_exit(NULL); }
static void *child_live(void *unused) { (void)unused; child_ran = 1; for (;;) __asm__ volatile("" ::: "memory"); }
static unsigned char stack_area[16384] __attribute__((aligned(16)));

static unsigned char *acquire(size_t bytes)
{
    sublet_cap slot;
    if (cap_vm_acquire(&slot, bytes, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) return NULL;
    return cap_vm_copyable(&slot);
}
static void fill_entry(unsigned char *frame, void *(*entry)(void *));
static void fill(unsigned char *frame) { fill_entry(frame, child); }
static void fill_entry(unsigned char *frame, void *(*entry)(void *))
{
    memset(frame, 0, 4096);
    ((uint64_t *)frame)[7] = 3;
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16), (void *)child);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack_area + sizeof stack_area);
    __asm__ volatile("stc gp, 0(%0)" : : "r"(frame + 64 + 3 * 16) : "memory");
    capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)entry);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "multipage")) {
        unsigned char *f = acquire(8192);
        if (!f) { puts("FRAME_TEST:multipage acquire-failed"); return 3; }
        fill(f);
        printf("FRAME_TEST:multipage rc=%ld (expect %d)\n", __capstone_vm_thread_create_frame(f), -EINVAL);
        return 0;
    }
    if (!strcmp(argv[1], "twice")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:twice acquire-failed"); return 3; }
        fill_entry(f, child_flag);
        long a = __capstone_vm_thread_create_frame(f);
        long b = __capstone_vm_thread_create_frame(f);
        printf("FRAME_TEST:twice first=%ld second=%ld (expect >0, then -EBUSY %d)\n", a, b, -EBUSY);
        fflush(stdout);
        /* the first child must still run: it sets child_ran before it exits */
        for (unsigned long i = 0; i < 400000000UL && !child_ran; ++i) __asm__ volatile("" ::: "memory");
        printf("FRAME_TEST:twice child_ran=%d (expect 1: the first child unharmed)\n", child_ran);
        return 0;
    }
    if (!strcmp(argv[1], "copy")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:copy acquire-failed"); return 3; }
        fill(f);
        long a = __capstone_vm_thread_create_frame(f);
        printf("FRAME_TEST:copy created=%ld (expect >0); reading our copy of the frame next\n", a);
        fflush(stdout);
        volatile uint64_t v = ((volatile uint64_t *)f)[7];
        printf("FRAME_TEST:copy SURVIVED read=%lu\n", (unsigned long)v);
        return 1;
    }
    if (!strcmp(argv[1], "copy-exited")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:copy-exited acquire-failed"); return 3; }
        fill_entry(f, child_flag);
        long a = __capstone_vm_thread_create_frame(f);
        for (unsigned long i = 0; i < 400000000UL && !child_ran; ++i) __asm__ volatile("" ::: "memory");
        usleep(500000);
        for (unsigned long i = 0; i < 400000000UL; ++i) __asm__ volatile("" ::: "memory");
        printf("FRAME_TEST:copy-exited created=%ld child_ran=%d (the child has exited); reading our copy\n", a, child_ran);
        fflush(stdout);
        volatile uint64_t v = ((volatile uint64_t *)f)[7];
        printf("FRAME_TEST:copy-exited SURVIVED read=%lu\n", (unsigned long)v);
        return 1;
    }
    if (!strcmp(argv[1], "write1024")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:write1024 acquire-failed"); return 3; }
        fill_entry(f, child_flag);
        long a = __capstone_vm_thread_create_frame(f);
        printf("FRAME_TEST:write1024 created=%ld (expect >0); writing our copy at frame+1024 next\n", a);
        fflush(stdout);
        /* +1024 is the snapshot's base; +1120 and +1408 are the two readings of its sp slot (ISSUES S-21). */
        ((volatile uint64_t *)(f + 1024))[0] = 0x5321000000001024UL;
        ((volatile uint64_t *)(f + 1120))[0] = 0x5321000000001120UL;
        ((volatile uint64_t *)(f + 1408))[0] = 0x5321000000001408UL;
        printf("FRAME_TEST:write1024 WROTE frame+1024, +1120, +1408\n");
        fflush(stdout);
        for (unsigned long i = 0; i < 400000000UL && !child_ran; ++i) __asm__ volatile("" ::: "memory");
        printf("FRAME_TEST:write1024 child_ran=%d (the child ran to its exit ECALL)\n", child_ran);
        return 0;
    }
    if (!strcmp(argv[1], "prewrite")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:prewrite acquire-failed"); return 3; }
        fill_entry(f, child_flag);
        ((volatile uint64_t *)(f + 1024))[0] = 0x5321000000000001UL;
        printf("FRAME_TEST:prewrite WROTE frame+1024 with no create (expect: lands)\n");
        return 0;
    }
    if (!strcmp(argv[1], "write-live")) {
        unsigned char *f = acquire(4096);
        if (!f) { puts("FRAME_TEST:write-live acquire-failed"); return 3; }
        fill_entry(f, child_live);
        long a = __capstone_vm_thread_create_frame(f);
        for (unsigned long i = 0; i < 400000000UL && !child_ran; ++i) __asm__ volatile("" ::: "memory");
        printf("FRAME_TEST:write-live created=%ld child_ran=%d (the child lives); writing our copy at frame+1024\n",
               a, child_ran);
        fflush(stdout);
        ((volatile uint64_t *)(f + 1024))[0] = 0x5321000000001024UL;
        ((volatile uint64_t *)(f + 1120))[0] = 0x5321000000001120UL;
        ((volatile uint64_t *)(f + 1408))[0] = 0x5321000000001408UL;
        printf("FRAME_TEST:write-live WROTE frame+1024, +1120, +1408 while the child lives\n");
        return 0;
    }
    return 2;
}
