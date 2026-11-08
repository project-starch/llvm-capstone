/* kfast-probe: forged delegated rounds for the kfast measurement prototype (perf/kernel-fastpath).
 *
 * Every line it prints is the RESULT of a delegated round, so a run with the module's kfast=0 and one with kfast=1
 * must print the same text; the module's kfast_* counters (sysfs) say which path each round took.
 *
 * __capstone_delegate_ints4 is the libc's own round for integer-only requests: it packs the request by the shape
 * table, so an OFFSET argument (read's buffer, fstat's stat) is flagged and carries whatever integer it is given.
 * That forges exactly the request the launcher's validator exists to refuse -- a buffer outside the exchange region
 * the domain was granted -- without touching any capability: the wire carries offsets, never capabilities
 * (capstone/delegate.h), and an out-of-range buffer is the only "bad buffer" the transport can express.
 *   bad-*   must be refused with -EFAULT (14) on both settings: the module must fall back (kfast_fb_invalid), the
 *           launcher must refuse.
 *   priv-*  fstat on fds 3..9 with a valid buffer: a launcher-private descriptor answers -EBADF (9) on both settings,
 *           the module falling back (kfast_fb_private).
 *   ok-*    valid requests the module serves when kfast=1 (kfast_served_*): the control that the fast path fires.
 *   clock-* CLOCK_PROCESS_CPUTIME_ID is not served (kfast_fb_clock); its result is 0 either way. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

long __capstone_delegate_ints4(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d);
enum { NR_LSEEK = 62, NR_READ = 63, NR_WRITE = 64, NR_FSTAT = 80, NR_CLOCK_GETTIME = 113 };

static void line(const char *what, long r) { printf("%s %ld\n", what, r); }

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/etc/hostname";
    int fd = open(path, O_RDONLY);
    char buf[64];
    struct timespec ts;
    if (fd < 0) { printf("open %s failed %d\n", path, errno); return 2; }
    /* ok: the libc's own read, lseek, fstat and clock_gettime, served by the module when kfast=1 */
    line("ok-read", read(fd, buf, sizeof buf));
    line("ok-lseek", lseek(fd, 0, SEEK_SET));
    line("ok-clock", clock_gettime(CLOCK_MONOTONIC, &ts));
    /* ok: forged but valid -- offset 256, 16 bytes, inside any exchange region */
    line("ok-forged-read", __capstone_delegate_ints4(NR_READ, fd, 256, 16, 0));
    line("ok-forged-fstat", __capstone_delegate_ints4(NR_FSTAT, fd, 512, 0, 0) == 0);
    /* bad: offsets and lengths outside the exchange region */
    line("bad-read-offset", __capstone_delegate_ints4(NR_READ, fd, UINT64_C(1) << 40, 16, 0));
    line("bad-read-length", __capstone_delegate_ints4(NR_READ, fd, 256, UINT64_C(1) << 40, 0));
    line("bad-read-wrap", __capstone_delegate_ints4(NR_READ, fd, 256, UINT64_MAX - 128, 0));
    line("bad-write-offset", __capstone_delegate_ints4(NR_WRITE, 1, UINT64_C(1) << 40, 4, 0));
    line("bad-fstat-offset", __capstone_delegate_ints4(NR_FSTAT, fd, UINT64_MAX, 0, 0));
    line("bad-clock-offset", __capstone_delegate_ints4(NR_CLOCK_GETTIME, CLOCK_MONOTONIC, UINT64_C(1) << 40, 0, 0));
    /* priv: the launcher's own descriptors are never the domain's */
    for (int i = 3; i < 10; ++i) {
        char what[32];
        snprintf(what, sizeof what, "priv-fstat-%d", i);
        long r = __capstone_delegate_ints4(NR_FSTAT, i, 512, 0, 0);
        line(what, r < 0 ? r : 0);
    }
    line("clock-cputime", clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts));
    /* ok: write(2) itself (stdio's printf goes out as writev), to a regular file */
    int out = open("/tmp/kf-probe-out.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    line("ok-write", write(out, "kfast probe\n", 12));
    line("bad-write-length", __capstone_delegate_ints4(NR_WRITE, out, 256, UINT64_C(1) << 40, 0));
    line("ok-close-out", close(out));
    line("ok-close", close(fd));
    /* O_DIRECT on ext2/3/4: the alignment check reads the buffer address, so the module must leave it to the
     * launcher (kfast_fb_file). The file is written by an ordinary descriptor first; every read result must be the
     * same on both settings, whichever it is. */
    const char *direct = argc > 2 ? argv[2] : "/tmp/kf-direct.dat";   /* must be on ext2/3/4: tmpfs refuses O_DIRECT */
    int w = open(direct, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    static char block[8192];
    memset(block, 'd', sizeof block);
    line("direct-fill", write(w, block, sizeof block));
    close(w);
    int d = open(direct, O_RDONLY | O_DIRECT);
    line("direct-open", d >= 0);
    static const uint64_t offs[] = {0, 16, 496, 512, 4096};
    for (unsigned i = 0; d >= 0 && i < sizeof offs / sizeof offs[0]; ++i) {
        char what[32];
        lseek(d, 0, SEEK_SET);
        snprintf(what, sizeof what, "direct-read-at-%u", (unsigned)offs[i]);
        line(what, __capstone_delegate_ints4(NR_READ, d, 4096 + offs[i], 512, 0));
    }
    if (d >= 0) close(d);
    return 0;
}
