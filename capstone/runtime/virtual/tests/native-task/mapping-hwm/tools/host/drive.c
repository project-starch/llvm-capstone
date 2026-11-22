/* drive.c <stream>: replays a logged @@HM request stream through the real plain_reuse.c. Lines "A <bytes>",
 * "R <bytes>" (release of a live mapping of that size, the newest), "X <new> <old>" (in-place growth of the newest
 * live mapping of <old>). The code under test prints its own @@HM lines on fd 2. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "glue.h"
struct m { void *p; size_t len; int live; } *L; size_t n, cap;
static size_t rnd(size_t x) { return (x + 4095) & -4096UL; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r"); char op; size_t b, o;
    while (fscanf(f, " %c %zu", &op, &b) == 2) {
        if (op == 'X') { if (fscanf(f, " %zu", &o) != 1) return 3; }
        if (op == 'A') {
            void *p = __plain_map(b); if (p == MAP_FAILED) return 4;
            if (n == cap) L = realloc(L, (cap = cap ? 2 * cap : 4096) * sizeof *L);
            L[n++] = (struct m){p, rnd(b), 1};
        } else {
            size_t want = op == 'X' ? o : rnd(b); long k;
            for (k = n - 1; k >= 0 && !(L[k].live && L[k].len == want); --k) ;
            if (k < 0) { fprintf(stderr, "no live mapping of %zu\n", want); return 5; }
            if (op == 'R') { L[k].live = 0; __plain_unmap(L[k].p, b); }
            else {
                __plain_remap_fit(o, b);
                void *q = mremap(L[k].p, o, b, MREMAP_MAYMOVE); if (q == MAP_FAILED) return 6;
                __plain_remapped(o, b); L[k].p = q; L[k].len = b;
            }
        }
    }
    return 0;
}
