/* The tracer's positive control: a known sequence of heap events; the expected counts go to stdout as
 * "EXPECT <field> <value>" lines (the parent's) and "EXPECT_CHILD <field> <value>" (the fork child's), written with
 * write(2) so that stdio allocates nothing. check-control.py compares them with slotcensus's lines.
 * Build it with -fno-builtin: at -O2 GCC deletes a malloc whose result is unused, and a calloc, aligned_alloc or
 * realloc(NULL) freed at once, so those events never reach the tracer (measured: 3025 of them). */
#define _GNU_SOURCE
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum { HOLD = 10000, MOVES = 100, INPLACE = 50, EACH = 10, NULLREALLOC = 5, ZEROREALLOC = 5, CHILD = 3000 };
static void *held[HOLD], *tmp[MOVES], *blk[MOVES], *kid[CHILD];

static void say(const char *who, const char *field, long long v)
{
    char line[96];
    int n = snprintf(line, sizeof line, "%s %s %lld\n", who, field, v);
    if (write(1, line, (size_t)n) < 0) _exit(3);
}

int main(void)
{
    long long allocs = 0, frees = 0;
    /* 1. the peak: HOLD objects live at once, then half freed */
    for (int i = 0; i < HOLD; ++i) held[i] = malloc(32), ++allocs;
    long long peak = HOLD;
    for (int i = 0; i < HOLD / 2; ++i) free(held[i]), ++frees;
    /* 2. realloc that must move: a blocker after each object, then growth to 64 MiB (mmap) */
    for (int i = 0; i < MOVES; ++i) {
        tmp[i] = malloc(16); blk[i] = malloc(16); allocs += 2;
        void *p = realloc(tmp[i], 64 << 20);
        if (!p || p == tmp[i]) _exit(4);       /* the control itself checks the move */
        tmp[i] = p; ++allocs; ++frees;         /* a move is one free and one allocation */
    }
    for (int i = 0; i < MOVES; ++i) free(tmp[i]), free(blk[i]), frees += 2;
    /* 3. realloc in place: to the same usable size */
    for (int i = 0; i < INPLACE; ++i) {
        void *p = malloc(100); ++allocs;
        void *q = realloc(p, malloc_usable_size(p));
        if (q != p) _exit(5);
        free(q); ++frees;
    }
    /* 4. the other entry points, each EACH times */
    for (int i = 0; i < EACH; ++i) {
        void *a = calloc(4, 8), *b = NULL, *c = aligned_alloc(64, 64), *d = memalign(128, 40);
        if (posix_memalign(&b, 32, 24)) _exit(6);
        allocs += 4;
        free(a); free(b); free(c); free(d); frees += 4;
    }
    /* 5. realloc(NULL, n) allocates; realloc(p, 0) frees (glibc returns NULL) */
    for (int i = 0; i < NULLREALLOC; ++i) free(realloc(NULL, 48)), ++allocs, ++frees;
    for (int i = 0; i < ZEROREALLOC; ++i) {
        void *p = malloc(48); ++allocs;
        if (realloc(p, 0) != NULL) _exit(7);
        ++frees;
    }
    /* 6. a fork child inherits HOLD/2 live objects and allocates CHILD more */
    long long live_at_fork = HOLD - HOLD / 2;
    pid_t pid = fork();
    if (pid == 0) {
        for (int i = 0; i < CHILD; ++i) kid[i] = malloc(64);
        say("EXPECT_CHILD", "peak_live", live_at_fork + CHILD);
        say("EXPECT_CHILD", "allocs", CHILD);
        say("EXPECT_CHILD", "forked", 1);
        clearenv();                             /* as nginx does to its workers: the tracer must still report */
        exit(0);                                /* exit, not _exit: the tracer reports from a destructor */
    }
    int st = 0;
    waitpid(pid, &st, 0);
    say("EXPECT", "peak_live", peak);
    say("EXPECT", "allocs", allocs);
    say("EXPECT", "frees", frees);
    say("EXPECT", "end_live", live_at_fork);
    say("EXPECT", "realloc_moved", MOVES);
    say("EXPECT", "realloc_inplace", INPLACE);
    say("EXPECT", "forked", 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : 8;
}
