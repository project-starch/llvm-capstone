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
 * That is M-6 in the registry, found by this program. Once the guard exists, this runs unchanged.
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

#define CYCLES 3

int main(int argc, char **argv) {
    unsigned long size = 0;
    for (int a = 1; a < argc; a++) {           /* argv[1] may be a domain path on the board */
        unsigned long v = strtoul(argv[a], NULL, 0);
        if (v > 0) { size = v; break; }
    }
    if (!size) size = 4UL << 20;               /* 4 MiB: at the buddy ceiling, so no CMA needed */

    if (capstone_init()) { printf("OFFSETCYCLE ERROR cannot initialise Capstone\n"); return 1; }
    int fd = open(CAPSTONE_DEV_PATH, O_RDWR);
    if (fd < 0) { printf("OFFSETCYCLE ERROR cannot open %s\n", CAPSTONE_DEV_PATH); return 1; }

    size_t off[CYCLES];
    printf("OFFSETCYCLE size=%lu cycles=%d\n", size, CYCLES);

    for (int i = 0; i < CYCLES; i++) {
        region_id_t rid = create_region(size);
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
        if (q.len != size) {                   /* the query landed on some other region */
            printf("OFFSETCYCLE ERROR queried len %lu != requested %lu\n",
                   (unsigned long)q.len, size);
            printf("BASELINE-PROBE offsetcycle_leak = -1\n");
            printf("__OFFSETCYCLE_WRONG_REGION__\n");
            return 1;
        }
        int rel = release_region(rid);
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
