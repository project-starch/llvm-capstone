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
 *   syscalls 64 delegated umask() calls, each checking the previous reply. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define PAGE 4096
#define PAGES 32

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
    volatile char *a = mmap(NULL, PAGES * PAGE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(a != MAP_FAILED, "mmap");
    if (!strcmp(mode, "seq")) {
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
    } else if (!strcmp(mode, "walk")) {
        for (int i = 0; i < PAGES; ++i)
            a[i * PAGE] = (char)i;
        for (int i = 0; i < PAGES; ++i)
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
