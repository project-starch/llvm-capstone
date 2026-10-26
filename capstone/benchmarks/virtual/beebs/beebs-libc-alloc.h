/* The <name>-libc variant of a BEEBS program with the in-file bump allocator: every call of
 * malloc_beebs / free_beebs / realloc_beebs / calloc_beebs goes to the C library's allocator.
 * Forced into every translation unit of the variant; build.py renames the bump definitions in the
 * defining file (bench_unused_bump_*), so these function-like macros reach only calls and
 * prototypes. The upstream text is otherwise unchanged. */
#include <stdlib.h>
#define malloc_beebs(n) malloc(n)
#define free_beebs(p) free(p)
#define realloc_beebs(p, n) realloc(p, n)
#define calloc_beebs(m, n) calloc(m, n)
