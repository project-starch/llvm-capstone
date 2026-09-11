/* Does releasing a region give its mmap-offset window back?
 *
 * WHY THIS EXISTS. `ioctl_release_region` used to decrement `region_n` and free the pages while
 * leaving `pre_mmap_offset` where the create had pushed it, so every create/release cycle consumed
 * offset space that nothing ever handed back. Nothing observed it because the offsets are internal
 * and the library caches them without invalidating (`libcapstone.c:562-571`), which after a
 * release+recreate makes the stale cache accidentally CORRECT rather than wrong -- the leak is
 * invisible from the mapping side and shows only in the offset itself.
 *
 * THE CHECK FIRES IN BOTH DIRECTIONS, which is the point of writing it rather than asserting the
 * fix is obvious. Three identical create/query/release cycles at the same size:
 *
 *     with the fix     off1 == off2 == off3            -- the window is handed back
 *     without it       off2 == off1 + size,  off3 == off1 + 2*size
 *
 * So a build carrying the old module FAILS this program. A test that could only pass would say
 * nothing about the module and something about the test.
 *
 * IT CANNOT RUN YET. The first release aborts QEMU on
 *     helper_csrevoke: Assertion `rs1_v->val.cap.type == CAP_TYPE_REV' failed
 * because the monitor revokes a region whose handle is still LINEAR when nothing ever shared it.
 * That is M-6 in the registry, and it IS NOW FIXED: the abort is gone and cycle 0 completes --
 * create, query, release, `rc=0`. What this program still cannot do is finish cycle 1, because the
 * first CREATE after a release faults. That is M-7, and it is NOT in the release path: a one-cycle
 * build of this same source passes end to end. "Once the guard exists, this runs unchanged" is what
 * an earlier version of this comment predicted, and it was wrong twice over -- a path nothing has
 * ever executed carried two defects, and the second one is a cycle further along than the first
 * reading of it said.
 *
 * SO ITS FAIL PATH IS PROVEN AND ITS PASS PATH IS NOT. Every run so far has aborted at cycle 0, which
 * exercises the create/query half and none of the comparison. When the guard lands, read the three
 * printed mmap_offset values by eye on the first successful run rather than trusting
 * __OFFSETCYCLE_PASSED__: a program that has only ever failed has an unproven idea of what passing
 * looks like.
 *
 * WHAT WOULD MAKE IT VOID, and is therefore checked rather than assumed: a release that the
 * monitor refuses (retval 1: the slot is kept because something sits above it) frees nothing and
 * restores nothing, so a run in which any release returns non-zero measures the refusal and not
 * the offset bookkeeping. That case exits non-zero with its own marker instead of reporting a pass.
 *
 * The query goes over a SECOND fd opened here. The module's region table and `pre_mmap_offset` are
 * file-scope statics shared by every opener, so this reads the same state the library mutated; it
 * avoids reaching into libcapstone's static `dev_fd`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "capstone.h"

/* Overridable so the run can be bisected: -DCYCLES=1 does one create/query/release and stops, which
   separates "the release itself faults" from "the NEXT create after a release faults". Without that
   separation the whole thing is one indivisible failure. */
#ifndef CYCLES
#define CYCLES 3
#endif
/* CYCLES=1 makes the comparison off[0] == off[0], which cannot fail and reports PASS having tested
   nothing about offsets. That is the single most expensive mistake class on this project, and a
   bisection arm walked straight into it -- the one-cycle run printed __OFFSETCYCLE_PASSED__ while
   the two-cycle run of the same build faulted. Refuse it at compile time rather than trusting
   whoever sets the define to remember. Use CYCLES=1 only to answer "does the release complete",
   and read that answer from the `released ... rc=` line, not from the verdict. */
#if CYCLES < 2
#error "offsetcycle needs CYCLES >= 2: with one cycle the offset comparison cannot fail"
#endif

int main(int argc, char **argv) {
    unsigned long size = 0;
    for (int a = 1; a < argc; a++) {           /* argv[1] may be a domain path on the board */
        unsigned long v = strtoul(argv[a], NULL, 0);
        if (v > 0) { size = v; break; }
    }
    if (!size) size = 4UL << 20;               /* 4 MiB: at the buddy ceiling, so no CMA needed */

    /* An optional SECOND size, used from cycle 1 onward. It separates two failures that look
       identical when every cycle asks for the same bytes: if the fault address follows the FIRST
       size the access is to the region that was freed, and if it follows the second it is the newly
       created one faulting on its own setup. With one size the kernel hands back the same pages and
       the two addresses coincide, so the question cannot be asked. */
    unsigned long size2 = 0;
    {
        int seen = 0;
        for (int a2 = 1; a2 < argc; a2++) {
            unsigned long v = strtoul(argv[a2], NULL, 0);
            if (v > 0) { if (seen) { size2 = v; break; } seen = 1; }
        }
    }
    if (!size2) size2 = size;

    if (capstone_init()) { printf("OFFSETCYCLE ERROR cannot initialise Capstone\n"); return 1; }
    int fd = open(CAPSTONE_DEV_PATH, O_RDWR);
    if (fd < 0) { printf("OFFSETCYCLE ERROR cannot open %s\n", CAPSTONE_DEV_PATH); return 1; }

    size_t off[CYCLES];
    printf("OFFSETCYCLE size=%lu size2=%lu cycles=%d\n", size, size2, CYCLES);

    for (int i = 0; i < CYCLES; i++) {
        unsigned long want = (i == 0) ? size : size2;
        region_id_t rid = create_region(want);
        if ((long)rid < 0) {
            printf("OFFSETCYCLE CREATE FAILED cycle=%d\n", i);
            printf("BASELINE-PROBE offsetcycle_leak = -1\n");
            printf("__OFFSETCYCLE_CREATE_FAILED__\n");
            return 1;
        }
        struct ioctl_region_query_args q;
        memset(&q, 0, sizeof q);
        q.region_id = rid;
        if (ioctl(fd, IOCTL_REGION_QUERY, (unsigned long)&q)) {
            printf("OFFSETCYCLE QUERY FAILED cycle=%d id=%lu\n", i, (unsigned long)rid);
            printf("BASELINE-PROBE offsetcycle_leak = -1\n");
            printf("__OFFSETCYCLE_QUERY_FAILED__\n");
            return 1;
        }
        off[i] = q.mmap_offset;
        printf("OFFSETCYCLE cycle=%d id=%lu len=%lu mmap_offset=%lu\n",
               i, (unsigned long)rid, (unsigned long)q.len, (unsigned long)off[i]);
        if (q.len != want) {                   /* the query landed on some other region */
            printf("OFFSETCYCLE ERROR queried len %lu != requested %lu\n",
                   (unsigned long)q.len, want);
            printf("BASELINE-PROBE offsetcycle_leak = -1\n");
            printf("__OFFSETCYCLE_WRONG_REGION__\n");
            return 1;
        }
        /* Markers either side of the release, because the monitor faults somewhere in here and a
           program that dies between two prints tells you which side of them it died on. Without
           these the failure is "somewhere after the query", which is one bit. */
        printf("OFFSETCYCLE cycle=%d releasing id=%lu\n", i, (unsigned long)rid);
        fflush(stdout);
        int rel = release_region(rid);
        printf("OFFSETCYCLE cycle=%d released id=%lu rc=%d\n", i, (unsigned long)rid, rel);
        fflush(stdout);
        if (rel != 0) {                        /* refused or kept: this run measures nothing */
            printf("OFFSETCYCLE RELEASE NOT FREED cycle=%d rc=%d -- run is VOID\n", i, rel);
            printf("BASELINE-PROBE offsetcycle_leak = -1\n");
            printf("__OFFSETCYCLE_RELEASE_REFUSED__\n");
            return 1;
        }
    }

    unsigned long leak = (unsigned long)(off[CYCLES - 1] - off[0]);
    int ok = 1;
    for (int i = 1; i < CYCLES; i++) if (off[i] != off[0]) ok = 0;
    printf("OFFSETCYCLE first=%lu last=%lu leak_bytes=%lu\n",
           (unsigned long)off[0], (unsigned long)off[CYCLES - 1], leak);
    printf("BASELINE-PROBE offsetcycle_leak = %lu\n", leak);
    fflush(stdout);
    printf(ok ? "__OFFSETCYCLE_PASSED__\n" : "__OFFSETCYCLE_LEAKED__\n");
    return ok ? 0 : 1;
}
