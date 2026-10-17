/* S-21 positive control: the domain tries to write its CHILD's frame after creating the thread.
 *
 * On the RTL every escape restores the S caller's snapshot from frame+1024, so a domain that can
 * still write its child's frame can overwrite the kernel's saved pc/sp/satp/stvec and have the next
 * escape load them. The fix retires the frame arena's ancestor at CV_THREAD_CREATE, which makes the
 * domain's own copy stale.
 *
 * Reading, and the ORDER of the markers is what makes it legible:
 *   S21_CREATE=<id>        the create was admitted (needs supervisor_slots > 1 on BOTH the module
 *                          and the machine; at one slot the create is refused and the test says
 *                          nothing about S-21)
 *   S21_BEFORE_WRITE       printed and flushed before the attempt, so a fault at the write is
 *                          distinguishable from a fault anywhere earlier
 *   S21_WRITE_SUCCEEDED    the escalation worked -- expected ONLY on a pre-S-21 module
 *
 * So: BEFORE_WRITE present and WRITE_SUCCEEDED absent = the hole is closed. Both present = open.
 * BEFORE_WRITE absent = the run says nothing, whatever else it printed.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <capstone/capability.h>
#include <capstone/virtual.h>
#include "vm.h"

extern long __capstone_vm_thread_create_frame(void *frame);
extern void __capstone_vm_thread_exit(void);

/* MUST NOT RETURN. The first version returned, and with x1 unset the child jumped to ra = 0 --
 * cause 12 at pc 0, which looked like a module fault and was my missing return trampoline.
 * thread.c installs thread_exit_entry in x1 for exactly this; exiting directly is simpler here. */
/* The child must be LIVE while the parent writes its frame, or the ordinary teardown denies the
 * write and the control cannot tell the modules apart. First version exited immediately: both arms
 * then read cause 25 at frame+1408 -- the right address, the wrong reason. It spins on a shared
 * flag the parent sets AFTER the attempt, so the frame is a live child's frame at that moment. */
static volatile int child_may_exit;
static volatile int child_is_running;

static void *child_entry(void *unused)
{
    (void)unused;
    child_is_running = 1;
    while (!child_may_exit) { }
    __capstone_vm_thread_exit();
    for (;;) { }
}

int main(void)
{
    sublet_cap slot;
    unsigned char *frame;
    long id;

    /* A marker BEFORE the first operation. Without it the first run printed nothing at all and
     * died at cause 12 pc 0, which says only "something went wrong somewhere" -- a void arm, not a
     * reading. Every stage below is bracketed the same way, so whatever fails is named. */
    puts("S21_START"); fflush(stdout);
    if (cap_vm_acquire(&slot, 4096, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) {
        puts("S21_NO_FRAME"); fflush(stdout); return 1;
    }
    puts("S21_ACQUIRED"); fflush(stdout);
    frame = cap_vm_copyable(&slot);
    if (!frame) { puts("S21_NO_COPYABLE"); fflush(stdout); return 1; }
    puts("S21_COPYABLE"); fflush(stdout);
    memset(frame, 0, 4096);
    ((uint64_t *)frame)[7] = 3;                      /* service and page-fault events */
    puts("S21_FRAME_ZEROED"); fflush(stdout);
    /* The child's stack must be SEPARATE memory. The first version pointed it at frame + 2048 --
     * inside the very page the module pins and, with S-21, retires -- so the child ran on a stack
     * it could not reach and faulted at pc 0. That was my bug, not a reading about S-21. */
    {
        void *stk = mmap(NULL, 65536, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (stk == MAP_FAILED) { puts("S21_NO_STACK"); fflush(stdout); return 1; }
        capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)child_entry);
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16),
                           (void *)((unsigned char *)stk + 65536));
        /* gp is process authority, inherited by every same-mm child; thread.c does this too. */
        __asm__ volatile("stc gp, 0(%0)" : : "r"(frame + 64 + 3 * 16) : "memory");
    }
    puts("S21_SLOTS_SET"); fflush(stdout);

    id = __capstone_vm_thread_create_frame(frame);
    printf("S21_CREATE=%ld\n", id); fflush(stdout);
    if (id < 0) { puts("S21_CREATE_REFUSED"); fflush(stdout); return 2; }

    /* Wait for the child to be demonstrably running, so the write targets a LIVE child's frame.
     * Bounded: if it never runs, say so instead of spinning -- an unbounded wait here would be a
     * hang reported as nothing. */
    {
        unsigned long spins = 0;
        while (!child_is_running && spins < 200000000UL) ++spins;
        if (!child_is_running) { puts("S21_CHILD_NEVER_RAN"); fflush(stdout); return 3; }
    }
    puts("S21_CHILD_LIVE"); fflush(stdout);

    puts("S21_BEFORE_WRITE"); fflush(stdout);
    /* The escalation attempt: frame+1408 is inside the S caller's snapshot region on silicon. */
    *(volatile uint64_t *)(frame + 1408) = 0xdeadbeefUL;
    puts("S21_WRITE_SUCCEEDED"); fflush(stdout);
    child_may_exit = 1;
    return 0;
}
