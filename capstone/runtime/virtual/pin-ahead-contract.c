/* The adapter's pin-ahead window (module parameter pin_ahead_pages), driven
 * by run-pin-ahead.py. Build with the virtual SDK. Every mode maps the same
 * 32-page private region; the runner compares each mode's CAPSTONE_VM_STATS
 * with mode "base", which maps it and touches nothing:
 *   seq     write page 0: the window pins forward, clipped to the arena;
 *   stop    write page 4, then page 0: the second window stops at page 4;
 *   vma     pages 8..31 read-only; write page 0 (the window stops at the
 *           VMA end, page 8), then read page 8 (pinned with FOLL_FORCE, and
 *           its window too);
 *   vmax    pages 8..31 read-write-execute, a VMA FOLL_WRITE can also pin:
 *           a write to page 0 pins up to the VMA end (8) all the same;
 *   rowrite as vma, then write page 9: still a fault (the domain dies), so a
 *           FOLL_FORCE pin never makes read-only memory writable;
 *   retire  write page 0, then munmap: every pin of the window is released.
 * run-step-hooks.py also uses:
 *   walk     write every page in order (one fault per page with window 1);
 *   stream   the same over a region of STREAM_PAGES pages instead (the adaptive window's
 *            growth and its cap, capstone_vm.c fault_ahead_max);
 *   clip/clipx walk across a VMA boundary at page 8 (read-only / executable);
 *   clipwrite as clip, then attempt a store into the read-only prefetched pages;
 *   nonlin   walk an image NONLIN grant, with an already-pinned page after the region;
 *   syscalls 64 delegated umask() calls, each checking the previous reply. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define PAGE 4096
#define PAGES 32
#define STREAM_PAGES 512

/* Separate the payload from startup globals. The final page stops speculation beyond it. */
static volatile char nonlin_region[(PAGES + 1) * PAGE] __attribute__((aligned(1048576)));

static void require(int ok, const char *message)
{
    if (!ok) { fprintf(stderr, "PIN_FAIL:%s\n", message); exit(1); }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "base";
    if (!strcmp(mode, "syscalls")) {
        mode_t previous = umask(022);
        for (int i = 0; i < 64; ++i) {
            mode_t next = i & 1 ? 022 : 077;
            require(umask(next) == (i & 1 ? 077 : 022), "umask reply");
        }
        umask(previous);
        printf("PIN_OK:%s\n", mode);
        return 0;
    }
    size_t pages = !strcmp(mode, "stream") ? STREAM_PAGES :
                   !strcmp(mode, "interleave") ? 2 * PAGES : PAGES;
    int nonlin = !strcmp(mode, "nonlin");
    int clipped = !strcmp(mode, "clip") || !strcmp(mode, "clipx") || !strcmp(mode, "clipwrite");
    volatile char *a = nonlin ? nonlin_region : mmap(NULL, pages * PAGE, PROT_READ | PROT_WRITE,
                                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(a != MAP_FAILED, "mmap");
    /* The region's address, so the runner counts the pins of THIS arena (the module's diag_admit
     * lines name arenas by address) rather than the process's pinned-page total, which also
     * holds the program's own lazily admitted pages. */
    printf("PIN_REGION:%p\n", (void *)a);
    if (nonlin) printf("PIN_SPAN:%u\n", PAGES);
    fflush(stdout);
    if (!strcmp(mode, "interleave")) {
        /* Two streams in one arena, with a VMA boundary preventing the
         * first stream's last window from speculating into the second. */
        require(!mprotect((void *)(a + PAGES * PAGE), PAGES * PAGE,
                          PROT_READ | PROT_WRITE | PROT_EXEC), "interleave mprotect");
        for (size_t i = 0; i < PAGES; ++i) {
            a[i * PAGE] = (char)i;
            a[(i + PAGES) * PAGE] = (char)(i + 1);
        }
        for (size_t i = 0; i < PAGES; ++i)
            require(a[i * PAGE] == (char)i && a[(i + PAGES) * PAGE] == (char)(i + 1),
                    "interleave readback");
    } else if (clipped) {
        int prot = !strcmp(mode, "clipx") ? PROT_READ | PROT_WRITE | PROT_EXEC : PROT_READ;
        require(!mprotect((void *)(a + 8 * PAGE), (PAGES - 8) * PAGE, prot), "clip mprotect");
        for (size_t i = 0; i < 8; ++i) a[i * PAGE] = (char)(i + 1);
        for (size_t i = 8; i < PAGES; ++i) require(a[i * PAGE] == 0, "clip readback");
        for (size_t i = 0; i < 8; ++i) require(a[i * PAGE] == (char)(i + 1), "clip kept stores");
        if (!strcmp(mode, "clipwrite")) {
            puts("PIN_ROWRITE_ATTEMPT"); fflush(stdout);
            a[9 * PAGE] = 1;
            require(0, "adaptive store to read-only page succeeded");
        }
    } else if (!strcmp(mode, "seq")) {
        a[0] = 1;
    } else if (!strcmp(mode, "stop")) {
        a[4 * PAGE] = 1;
        a[0] = 1;
    } else if (!strcmp(mode, "vma") || !strcmp(mode, "rowrite")) {
        require(!mprotect((void *)(a + 8 * PAGE), (PAGES - 8) * PAGE, PROT_READ), "mprotect");
        a[0] = 1;
        require(a[8 * PAGE] == 0, "read-only page reads zero");
        require(a[0] == 1, "first page kept its store");
        if (!strcmp(mode, "rowrite")) {
            printf("PIN_ROWRITE_ATTEMPT\n");
            fflush(stdout);
            a[9 * PAGE] = 1;
            require(0, "store to a read-only page succeeded");
        }
    } else if (!strcmp(mode, "vmax")) {
        require(!mprotect((void *)(a + 8 * PAGE), (PAGES - 8) * PAGE,
                          PROT_READ | PROT_WRITE | PROT_EXEC), "mprotect");
        a[0] = 1;
    } else if (!strcmp(mode, "walk") || !strcmp(mode, "stream") || !strcmp(mode, "gaps") || nonlin) {
        if (nonlin) a[PAGES * PAGE] = 1;
        /* Cross an existing pin both before training and within a window.
         * Earlier faults also leave older predictor slots at the same next
         * page: the active walk must win when those streams converge. */
        if (!strcmp(mode, "gaps")) a[PAGE] = a[7 * PAGE] = 1;
        for (size_t i = 0; i < pages; ++i)
            a[i * PAGE] = (char)i;
        for (size_t i = 0; i < pages; ++i)
            require(a[i * PAGE] == (char)i, "walk readback");
    } else if (!strcmp(mode, "retire")) {
        a[0] = 1;
        require(!munmap((void *)a, PAGES * PAGE), "munmap");
    } else {
        require(!strcmp(mode, "base"), "mode");
    }
    printf("PIN_OK:%s\n", mode);
    return 0;
}
