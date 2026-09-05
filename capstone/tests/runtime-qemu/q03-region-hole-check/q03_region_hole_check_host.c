/* Q-03 module-consistency loader.
 *
 * A drop-in replacement for capstone-test.user as the batch runner's --loader: it creates and
 * calls the domain exactly as that loader does (same two printf lines, so the runner's retval
 * parse is unchanged), then checks that the kernel module's region table still agrees with the
 * monitor's after whatever the item did to the pool:
 *
 *   Q03CHK count=N                       the module's count after a probe (== monitor REGION_COUNT,
 *                                        holes included), found by querying ids until len == 0
 *   Q03CHK region id= qlen= mmap= write= a fresh region: the module's cached length must be the
 *                                        one requested, mmap must resolve, the page must be writable
 *   Q03CHK oob_share id= retval=         REGION_SHARE of an out-of-range id: the ioctl plumbing must
 *                                        deliver the monitor's rejection (nonzero), or the next line
 *                                        cannot be read. On 2026-09-05 this line found a regression
 *                                        the replays could not: the first hole build read
 *                                        region_live[id] for an out-of-range id (Capstone-C does not
 *                                        short-circuit ||) and faulted in M-mode.
 *   Q03CHK hole_share id= retval=        REGION_SHARE of the id given as argv[2] -- a HOLE left by an
 *                                        exact fit -- must be rejected (nonzero) before any domain
 *                                        call. On a live id the same ioctl would hand a pool fragment
 *                                        to the domain, so it is only issued when an id is given.
 *
 * All checks run only for items whose name contains "chk"; every other item gets exactly the
 * capstone-test.user ioctl sequence, so the pool history that produces the exact fit is unchanged.
 *   Q03CHK dmesg reuse= fetchfail=       the module's two alerts, counted from the ring buffer (the
 *                                        runner sets the console level so they never print live)
 *
 * The share ioctl is issued directly, not through libcapstone, whose share_region() is void and
 * discards the monitor's return value. See docs/plans/q03-region-hole-sentinel.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
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
    if (argc < 2) {
        fprintf(stderr, "usage: %s <domain.dom> [hole-id]\n", argv[0]);
        return 1;
    }
    if (capstone_init()) {
        fprintf(stderr, "Failed to initialise Capstone\n");
        return 1;
    }
    dom_id_t dom_id = create_dom(argv[1], NULL);
    printf("Created domain ID = %lu\n", (unsigned long)dom_id);
    unsigned long dom_retval = call_dom(dom_id);
    printf("Called dom (1-th time) retval = %lu\n", dom_retval);

    /* Items whose name does not contain "chk" get EXACTLY capstone-test.user's ioctl sequence and
       nothing else, so the pool history that produces the exact fit at item 8 is unchanged. */
    if (!strstr(argv[1], "chk")) {
        capstone_cleanup();
        return 0;
    }

    int fd = open(CAPSTONE_DEV_PATH, O_RDWR);
    if (fd < 0) {
        printf("Q03CHK ERROR cannot open %s\n", CAPSTONE_DEV_PATH);
        return 2;
    }
    probe_regions();
    /* the module's count after the probe (== the monitor's REGION_COUNT, holes included): the
       module answers len = 0 for any id at or beyond its count. libcapstone's region_count() is
       its own lazily-filled cache and reads 0 here, so it is not used. */
    int count = 0;
    for (;;) {
        struct ioctl_region_query_args cq;
        memset(&cq, 0, sizeof cq);
        cq.region_id = (region_id_t)count;
        ioctl(fd, IOCTL_REGION_QUERY, &cq);
        if (cq.len == 0 || count > 200) break;
        count++;
    }
    printf("Q03CHK count=%d\n", count);

    region_id_t rid = create_region(4096);
    struct ioctl_region_query_args q;
    memset(&q, 0, sizeof q);
    q.region_id = rid;
    ioctl(fd, IOCTL_REGION_QUERY, &q);
    unsigned char *p = map_region(rid, 4096);
    int wr_ok = 0;
    if (p) {
        p[0] = 0x5a;
        p[4095] = 0xa5;
        wr_ok = (p[0] == 0x5a && p[4095] == 0xa5);
    }
    printf("Q03CHK region id=%lu qlen=%lu mmap=%s write=%d\n",
           (unsigned long)rid, (unsigned long)q.len, p ? "ok" : "NULL", wr_ok);

    struct ioctl_region_share_args oob;
    memset(&oob, 0, sizeof oob);
    oob.dom_id = dom_id;
    oob.region_id = (region_id_t)count + 100;
    ioctl(fd, IOCTL_REGION_SHARE, &oob);
    printf("Q03CHK oob_share id=%lu retval=%u\n", (unsigned long)oob.region_id, oob.retval);

    if (argc >= 3 && strstr(argv[1], "chk")) {
        struct ioctl_region_share_args hs;
        memset(&hs, 0, sizeof hs);
        hs.dom_id = dom_id;
        hs.region_id = (region_id_t)strtoul(argv[2], NULL, 0);
        ioctl(fd, IOCTL_REGION_SHARE, &hs);
        printf("Q03CHK hole_share id=%lu retval=%u\n", (unsigned long)hs.region_id, hs.retval);
    }
    printf("Q03CHK dmesg reuse=%lu fetchfail=%lu\n",
           dmesg_count("Region ID reuse detected"), dmesg_count("Failed to fetch information"));
    close(fd);
    capstone_cleanup();
    return 0;
}
