/* rgrow MODE: directed test of mallocng patch 0011 (large reallocs grown in place) on the native route.
 * MODE ok: grows a large chunk twice (a move into a mapping with room, then in place), shrinks it, and checks every
 *   word and every stored pointer (capability tags survive the copy and the in-place reissue) at each step; prints
 *   "RGROW moved=<0|1> inplace=<0|1> OK" and exits 0, or "RGROW FAIL <step> <index>" and exits 1.
 * MODE stale: grows in place, then reads through the capability realloc replaced; must not return (the old child is
 *   ended); prints "RGROW STALE-READ-RETURNED" and exits 3 if it does. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct cell { long v; long *ptr; };
static long target[64];

static int check(struct cell *c, size_t n, const char *step)
{
    for (size_t i = 0; i < n; ++i)
        if (c[i].v != (long)(i * 2654435761u) || c[i].ptr != &target[i % 64] || *c[i].ptr != (long)(i % 64)) {
            printf("RGROW FAIL %s %zu\n", step, i);
            return 1;
        }
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    for (int i = 0; i < 64; ++i) target[i] = i;
    size_t n0 = 12000, n1 = 60000, n2 = 110000, n3 = 20000;   /* x 32 B: 384 KB, 1.9 MB, 3.5 MB, 640 KB */
    struct cell *a = malloc(n0 * sizeof *a);
    if (!a) { printf("RGROW FAIL malloc 0\n"); return 1; }
    for (size_t i = 0; i < n0; ++i) { a[i].v = (long)(i * 2654435761u); a[i].ptr = &target[i % 64]; }
    if (check(a, n0, "initial")) return 1;
    uintptr_t at0 = (uintptr_t)a;
    struct cell *b = realloc(a, n1 * sizeof *b);                /* moves: the first mapping has no room */
    if (!b) { printf("RGROW FAIL realloc1 0\n"); return 1; }
    if (check(b, n0, "grow1")) return 1;
    for (size_t i = n0; i < n1; ++i) { b[i].v = (long)(i * 2654435761u); b[i].ptr = &target[i % 64]; }
    uintptr_t at1 = (uintptr_t)b;
    struct cell *c = realloc(b, n2 * sizeof *c);                /* in place: room for 2 x n1 */
    if (!c) { printf("RGROW FAIL realloc2 0\n"); return 1; }
    if (!strcmp(mode, "stale")) {
        volatile long v = b[0].v;                               /* b was ended by the in-place reissue */
        printf("RGROW STALE-READ-RETURNED %ld inplace=%d\n", (long)v, (uintptr_t)c == at1);
        return 3;
    }
    if (check(c, n1, "grow2")) return 1;
    for (size_t i = n1; i < n2; ++i) { c[i].v = (long)(i * 2654435761u); c[i].ptr = &target[i % 64]; }
    if (check(c, n2, "fill2")) return 1;
    struct cell *d = realloc(c, n3 * sizeof *d);                /* shrink */
    if (!d) { printf("RGROW FAIL realloc3 0\n"); return 1; }
    if (check(d, n3, "shrink")) return 1;
    free(d);
    printf("RGROW moved=%d inplace=%d OK\n", at1 != at0, (uintptr_t)c == at1);
    return 0;
}
