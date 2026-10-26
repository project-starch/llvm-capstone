/* The two image shapes the two-segment loader has to carry beyond an ordinary program, built from
 * this one source with -DSHAPE_TABLE (a constant table of capabilities) or without it (no global
 * that holds a capability). Each prints VIRTUAL_SHAPE_OK <name> once; run-shapes.py counts them.
 *
 * no_capability_globals: neither the program nor the libc it links emits a .gct record, so the
 * section holds only the runtime's start and end markers and has length zero (readelf). The
 * markers are declared allocated ("a"); with the flag reverted this build does not link:
 * ld.lld reports "text segment out of order: no allocated .gct input?" from virtual/link.ld,
 * which is what keeps an empty .gct inside the read+execute segment.
 *
 * constant_capability_table: a table of capabilities with constant initialisers. The initialiser
 * has to store tagged values, so the table cannot live in read+execute text; the compiler places
 * it in .data (the read+write segment). An image that put it in .rodata would fault at the first
 * initialiser store. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef SHAPE_TABLE
static int first = 7, second = 11;
static int plus_one(int x) { return x + 1; }
static int times_two(int x) { return x * 2; }
static int *const data_table[] = { &first, &second };
static int (*const function_table[])(int) = { plus_one, times_two };
#endif

int main(void)
{
    int failures = 0;
#ifdef SHAPE_TABLE
    volatile int i = 0;
    int a = *data_table[i], b = *data_table[i + 1];
    int c = function_table[i](a), d = function_table[i + 1](b);
    if (a == 7 && b == 11 && c == 8 && d == 22)
        printf("VIRTUAL_SHAPE_OK constant_capability_table\n");
    else { printf("VIRTUAL_SHAPE_FAIL constant_capability_table %d %d %d %d\n", a, b, c, d); ++failures; }
#else
    char buffer[64];
    volatile int n = 40;
    memset(buffer, 'x', (size_t)n);
    buffer[n] = 0;
    if (strlen(buffer) == 40)
        printf("VIRTUAL_SHAPE_OK no_capability_globals\n");
    else { printf("VIRTUAL_SHAPE_FAIL no_capability_globals\n"); ++failures; }
#endif
    printf("VIRTUAL_SHAPE_DONE failures=%d\n", failures);
    return failures != 0;
}
