/* Fourth negative test of the S-21 frame protection, from the TRUSTED side (a domain cannot ask for a grant at a chosen
 * address): after a well-formed child frame P1 is created, add() must refuse a grant [P0, P0+8192) that overlaps the frame
 * page but starts before it, and must accept the adjacent single pages P2 (above) and P0 (below) as the control that the
 * check is not refusing everything. Prints OVERLAP_TEST lines with each call's result and errno. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#include "wire.h"

static int dev;
/* Each grant into its own EMPTY register slot: before the context starts add() refuses a grant into a
 * non-empty slot with -EFAULT, which would mask the adjacency the controls test. */
static int add(char *base, unsigned long bytes, unsigned reg, const char *what)
{
    struct cv_map r = {0, (unsigned long)base, bytes, 6, reg, (unsigned long)base, 0, 0};
    int rc = ioctl(dev, CV_ADD, &r), e = rc ? errno : 0;
    printf("OVERLAP_TEST:%s base=%p bytes=%lu rc=%d errno=%d (%s)\n", what, base, bytes, rc, e, e ? strerror(e) : "ok");
    return rc ? -e : 0;
}
int main(void)
{
    dev = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    if (dev < 0) { printf("OVERLAP_TEST:open-failed errno=%d\n", errno); return 3; }
    char *p = mmap(NULL, 4 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { puts("OVERLAP_TEST:mmap-failed"); return 3; }
    if (madvise(p, 4 * 4096, MADV_NOHUGEPAGE) && errno != EINVAL) { puts("OVERLAP_TEST:madvise-failed"); return 3; }
    memset(p, 0, 4 * 4096);
    char *P0 = p, *P1 = p + 4096, *P2 = p + 8192;
    int frame = add(P1, 4096, 10, "frame-grant(expect ok)");
    struct cv_thread_create tc = {(unsigned long)P1, 0};
    int crc = ioctl(dev, CV_THREAD_CREATE, &tc), ce = crc ? errno : 0;
    printf("OVERLAP_TEST:thread-create frame=%p rc=%d errno=%d thread=%llu (expect ok, thread > 0)\n",
           P1, crc, ce, (unsigned long long)tc.thread);
    int over = add(P0, 8192, 11, "overlap-starting-before(expect EINVAL)");
    int above = add(P2, 4096, 11, "adjacent-above(expect ok)");
    int below = add(P0, 4096, 12, "adjacent-below(expect ok)");
    int pass = !frame && !crc && tc.thread > 0 && over == -EINVAL && !above && !below;
    printf("OVERLAP_TEST:%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
