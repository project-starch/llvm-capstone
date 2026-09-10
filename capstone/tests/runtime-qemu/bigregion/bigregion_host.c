/* Can a capability region larger than the buddy allocator's largest block exist?
 *
 * WHY THIS EXISTS. modcapstone now allocates every region with `dma_alloc_pages`, which draws
 * from CMA when a CMA area exists and from the buddy allocator when it does not. The buddy
 * allocator's ceiling is MAX_ORDER 10 x 4 KiB = 4 MiB (include/linux/mmzone.h), and NO REGION
 * ABOVE 4 MiB HAS EVER BEEN CREATED ON ANY TARGET -- board or QEMU. This program is the first
 * thing that tries.
 *
 * IT IS BUILT AS A MATCHED PAIR, not a single arm. Run it at 4 MiB and at 8 MiB with no CMA area:
 * the first must succeed and the second must fail, which is what proves the ceiling is real and
 * where it is. Then run 8 MiB again WITH `cma=`: it must succeed. A single passing arm at 8 MiB
 * proves nothing on its own -- it is equally consistent with the size argument never reaching the
 * allocator.
 *
 * TWO CHECKS THAT LOOK PEDANTIC AND ARE NOT, both of which have already cost this project a
 * misread measurement:
 *
 *   - `create_region` returns (region_id_t)-1 on failure, and region_id_t is UNSIGNED. Testing it
 *     as `if (rid)` or forgetting to test it at all is how a create failure came to be reported as
 *     "map_region failed" in three committed documents. The test is `(long)rid < 0`.
 *   - `map_region` returns mmap()'s value RAW, so a rejected mapping is MAP_FAILED ((void*)-1),
 *     not NULL. `if (p)` accepts MAP_FAILED and the program then faults on first use.
 *
 * The write-back test touches the FIRST and LAST byte. The last byte is the one that matters: an
 * allocation that succeeded but is shorter than asked for -- the exact failure mode a rounding bug
 * produces -- passes a first-byte test and fails this one.
 *
 * Kernel Oops/BUG/WARNING lines are counted before and after, because a large CMA allocation
 * migrates pages and is a plausible way to upset the kernel rather than merely fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "capstone.h"

static unsigned long dmesg_count(const char *needle) {
    char cmd[256];
    snprintf(cmd, sizeof cmd, "dmesg | grep -c '%s'", needle);
    FILE *f = popen(cmd, "r");
    if (!f) return (unsigned long)-1;
    unsigned long n = 0;
    if (fscanf(f, "%lu", &n) != 1) n = 0;
    pclose(f);
    return n;
}

int main(int argc, char **argv) {
    /* The size in BYTES is the first numeric argument. The board driver invokes host programs as
       `host <domain-path> <args>`, so argv[1] may be a path -- skipped, not parsed as 0. */
    unsigned long size = 0;
    for (int a = 1; a < argc; a++) {
        unsigned long v = strtoul(argv[a], NULL, 0);
        if (v > 0) { size = v; break; }
    }
    if (!size) { printf("BIGREGION ERROR no size given\n"); return 1; }

    if (capstone_init()) { printf("BIGREGION ERROR cannot initialise Capstone\n"); return 1; }

    unsigned long oops0 = dmesg_count("Oops"), bug0 = dmesg_count("BUG:"), warn0 = dmesg_count("WARNING:");
    printf("BIGREGION request bytes=%lu (%lu MiB)\n", size, size >> 20);

    region_id_t rid = create_region(size);
    if ((long)rid < 0) {
        /* Distinguishable from a map failure, which is the whole point. */
        printf("BIGREGION CREATE FAILED bytes=%lu -- no CMA area, or above it\n", size);
        printf("BASELINE-PROBE bigregion_bytes = 0\n");
        printf("__BIGREGION_CREATE_FAILED__\n");
        return 1;
    }
    printf("BIGREGION created id=%lu\n", (unsigned long)rid);

    unsigned char *p = (unsigned char *)map_region(rid, size);
    if (!p || p == (unsigned char *)-1) {
        printf("BIGREGION MAP FAILED id=%lu ptr=%p\n", (unsigned long)rid, (void *)p);
        printf("BASELINE-PROBE bigregion_bytes = 0\n");
        printf("__BIGREGION_MAP_FAILED__\n");
        return 1;
    }
    printf("BIGREGION mapped id=%lu at %p\n", (unsigned long)rid, (void *)p);

    /* First and last byte. The LAST one is the real test: a short allocation passes the first. */
    p[0] = 0x5a;
    p[size - 1] = 0xa5;
    int ok = (p[0] == 0x5a && p[size - 1] == 0xa5);
    printf("BIGREGION writeback first=%d last=%d\n", p[0] == 0x5a, p[size - 1] == 0xa5);

    unsigned long oops = dmesg_count("Oops") - oops0,
                  bug  = dmesg_count("BUG:") - bug0,
                  warn = dmesg_count("WARNING:") - warn0;
    printf("BIGREGION dmesg oops=%lu bug=%lu warn=%lu\n", oops, bug, warn);
    if (oops || bug || warn) ok = 0;

    /* ALSO report in the BASELINE-PROBE form, because the BOARD driver's staged-marker guard
       hard-stops on any arm that emits neither `SQ: obs=` nor `ladder-perf: RESULT ... retval=`
       nor `BASELINE-PROBE <what> = <value>`. That guard cost boot sw52 its last arm when a
       host-binary probe reported in a format it did not know. This probe is a host binary
       reporting a value, so the form is honest rather than a workaround -- and emitting it here
       is cheaper than teaching the guard a fourth format. */
    printf("BASELINE-PROBE bigregion_bytes = %lu\n", ok ? size : 0UL);
    fflush(stdout);
    printf(ok ? "__BIGREGION_PASSED__\n" : "__BIGREGION_FAILED__\n");
    return ok ? 0 : 1;
}
