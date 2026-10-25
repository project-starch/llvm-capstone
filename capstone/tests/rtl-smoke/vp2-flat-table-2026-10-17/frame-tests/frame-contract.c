/* Negative tests of the child-frame protection (S-21), modes:
 *   multipage  a frame inside a 2-page heap arena            -> create must fail -EINVAL (find_arena exact match)
 *   twice      the same frame created twice                  -> the 2nd create's retire fails -> -EACCES + must-fire line
 *   copy       a well-formed frame, then read our own copy   -> create succeeds, the read faults cause 25
 * The frame is filled exactly as runtime/virtual/thread.c fills it. Prints FRAME_TEST lines; a survived read exits 1. */
#include <capstone/capability.h>
#include <capstone/virtual.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vm.h"

extern long __capstone_vm_thread_create_frame(void *frame);

static void *child(void *unused) { (void)unused; capstone_virtual_thread_exit(NULL); }
static volatile int child_ran;
static void *child_flag(void *unused) { (void)unused; child_ran = 1; capstone_virtual_thread_exit(NULL); }
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
    return 2;
}
