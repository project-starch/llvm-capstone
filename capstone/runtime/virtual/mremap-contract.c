/* mremap probe for the runtime's map-copy-unmap mremap (runtime/app-mremap). Modes:
 *   ok     grow 3 -> 5 pages (bytes and a stored capability survive), shrink 5 -> 2, same-page
 *          resize returns the same pointer, no-MAYMOVE stays ENOSYS, a non-mapping fails cleanly;
 *   stale  grow, then read through the OLD pointer: must fault (revoked);
 *   fresh  grow, then read through the NEW pointer: the control for stale. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define P 4096UL
static void require(int ok, const char *m) { if (!ok) { printf("MREMAP_FAIL:%s errno=%d\n", m, errno); fflush(stdout); exit(1); } }
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    int *obj = malloc(sizeof *obj); require(obj != NULL, "malloc"); *obj = 4242;
    unsigned char *p = mmap(NULL, 3 * P, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(p != MAP_FAILED, "mmap");
    for (unsigned long i = 0; i < 3 * P; ++i) p[i] = (unsigned char)(i * 7);
    *(int **)(p + 64) = obj;
    unsigned char *q = mremap(p, 3 * P, 5 * P, MREMAP_MAYMOVE);
    require(q != MAP_FAILED, "grow");
    if (!strcmp(mode, "stale")) { printf("MREMAP_STALE_READ\n"); fflush(stdout); volatile unsigned char x = p[100]; (void)x; require(0, "stale read survived"); }
    if (!strcmp(mode, "fresh")) { printf("MREMAP_FRESH_READ %u\n", q[100]); return 0; }
    for (unsigned long i = 0; i < 3 * P; ++i) if (i < 64 || i >= 80) require(q[i] == (unsigned char)(i * 7), "grow bytes");
    require(**(int **)(q + 64) == 4242, "capability survived the move");
    q[5 * P - 1] = 9; require(q[3 * P] == 0, "new pages zero");
    unsigned char *r = mremap(q, 5 * P, 2 * P, MREMAP_MAYMOVE);
    require(r != MAP_FAILED && r[100] == (unsigned char)700 && **(int **)(r + 64) == 4242, "shrink");
    require(mremap(r, 2 * P, 2 * P - 100, MREMAP_MAYMOVE) == r, "same pages returns same pointer");
    errno = 0; require(mremap(r, 2 * P, 4 * P, 0) == MAP_FAILED && errno == ENOSYS, "no MAYMOVE stays ENOSYS");
    char *heap = malloc(8192); errno = 0;
    void *bad = mremap(heap, 8192, 3 * P, MREMAP_MAYMOVE);
    require(bad == MAP_FAILED && errno != 0, "non-mapping refused");
    require(r[100] == (unsigned char)700, "r untouched after the refused call");
    printf("MREMAP_OK bad_errno=%d\n", errno);
    return 0;
}
