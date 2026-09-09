/* M-2 positive control (ISSUES.md M-2; docs/plans/monitor-unification.md Phase B item 9).
 *
 * The kernel module mirrors the monitor's region table into its own `regions[]` array in
 * probe_regions(), copying up to the monitor's REGION_COUNT with NO bound against its own array
 * size. Before the fix the array held 64 entries while the monitor (Phase B item 3, both targets)
 * has 96 slots, so the 65th region overran the array into the statics that follow it. Nothing
 * reached that on the board because the Q-03 wedge stopped every run first; the hole fix removed
 * that brake.
 *
 * This program creates regions (up to N, default 72) with NO domain at all -- domain slots are
 * never reused, so a domain per region would hit the 32-domain budget first -- until the monitor
 * refuses (its pre-carve check, print 0x1237: every carve also leaves a fragment slot, so ~52 host
 * regions fill 96 monitor slots and the highest ID handed out is ~94). Then, for every ID from 64
 * to that highest ID -- the zone the module mirrored past its old array -- it queries the module's
 * cached length and maps, writes and reads the region, and counts kernel Oops/BUG/WARNING lines.
 * Expected on a fixed module:
 *   M2 created=52 max_id=94 first_failed_id=52
 *   M2 region id=64 qlen=4096 mmap=ok write=1        ... through id=94 (holes: qlen=0 is a hole,
 *                                                        reported and skipped, not a failure)
 *   M2 dmesg oops=0 bug=0 warn=0
 *   __M2_REGION_OVERFLOW_PASSED__
 * On the unfixed module the same run prints wrong lengths / NULL maps for ids >= 64 or dies
 * in the kernel (no PASSED line either way). That is the control: it must fail before the fix. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "capstone.h"

#define OLD_MODULE_MAX 64

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
    /* N = the first numeric argument; the board driver invokes host programs as
       `host <domain-path> <args>`, so argv[1] may be a path -- skipped, not parsed as 0. */
    int n = 72;
    for (int a = 1; a < argc; a++) { int v = atoi(argv[a]); if (v > 0) { n = v; break; } }
    if (capstone_init()) {
        fprintf(stderr, "M2 ERROR cannot initialise Capstone\n");
        return 1;
    }
    int fd = open(CAPSTONE_DEV_PATH, O_RDWR);
    if (fd < 0) {
        printf("M2 ERROR cannot open %s\n", CAPSTONE_DEV_PATH);
        return 1;
    }
    unsigned long oops0 = dmesg_count("Oops"), bug0 = dmesg_count("BUG:"), warn0 = dmesg_count("WARNING:");

    /* Create until the monitor refuses (pre-carve check, print 0x1237) or N is reached. Every
       carve from the pool also leaves a fragment slot, so ~52 host regions fill 96 monitor slots;
       what matters is the highest region ID the monitor handed out, which is what the module
       mirrored -- past 64 that mirroring overran its array before the fix. */
    region_id_t ids[256];
    int created = 0; long first_failed = -1, max_id = -1;
    for (int i = 0; i < n && i < 256; i++) {
        region_id_t rid = create_region(4096);
        if ((long)rid < 0) { first_failed = i; break; }
        ids[created++] = rid;
        if ((long)rid > max_id) max_id = (long)rid;
    }
    printf("M2 created=%d max_id=%ld first_failed_id=%ld\n", created, max_id, first_failed);

    /* Only the regions THIS program created are checked (4096 bytes each, mappable); the other IDs
       in the zone are pool fragments (other lengths) or holes (length 0) and are not the subject.
       The control is vacuous unless at least one created region has an ID past the old array. */
    int checked = 0, ok = 1;
    for (int k = 0; k < created; k++) {
        long id = (long)ids[k];
        if (id < OLD_MODULE_MAX) continue;
        struct ioctl_region_query_args q;
        memset(&q, 0, sizeof q);
        q.region_id = ids[k];
        ioctl(fd, IOCTL_REGION_QUERY, &q);
        unsigned char *p = map_region(ids[k], 4096);
        int wr = 0;
        if (p) { p[8] = (unsigned char)(0x40 + (id & 0x3f)); p[4095] = 0xa5; wr = (p[8] == (unsigned char)(0x40 + (id & 0x3f)) && p[4095] == 0xa5); }
        printf("M2 region id=%ld qlen=%lu mmap=%s write=%d\n", id, (unsigned long)q.len, p ? "ok" : "NULL", wr);
        if (q.len != 4096 || !p || !wr) ok = 0;
        checked++;
    }
    printf("M2 checked_past_%d=%d\n", OLD_MODULE_MAX, checked);
    if (checked == 0) { printf("M2 VACUOUS: no created region has an ID >= %d\n", OLD_MODULE_MAX); ok = 0; }
    unsigned long oops = dmesg_count("Oops") - oops0, bug = dmesg_count("BUG:") - bug0, warn = dmesg_count("WARNING:") - warn0;
    printf("M2 dmesg oops=%lu bug=%lu warn=%lu\n", oops, bug, warn);
    if (oops || bug || warn) ok = 0;
    fflush(stdout);
    printf(ok ? "__M2_REGION_OVERFLOW_PASSED__\n" : "__M2_REGION_OVERFLOW_FAILED__\n");
    capstone_cleanup();
    return ok ? 0 : 1;
}
