/* ffcensus's positive control: hands replace_init a function table built from libc (as Firefox hands it mozjemalloc's)
 * and runs control.c's event sequence through the table, so check-control.py can compare the census with the counts
 * the sequence must produce. With an argument "arena" it runs the moz_arena_* control instead: ARENA objects held at
 * once through moz_arena_malloc/calloc/memalign, MOVES moving moz_arena_reallocs, then all freed. */
#define _GNU_SOURCE
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "replace_malloc.h"

static malloc_table_t T;

static void *arena_malloc(arena_id_t a, size_t n) { (void)a; return malloc(n); }
static void *arena_calloc(arena_id_t a, size_t k, size_t n) { (void)a; return calloc(k, n); }
static void *arena_realloc(arena_id_t a, void *p, size_t n) { (void)a; return realloc(p, n); }
static void arena_free(arena_id_t a, void *p) { (void)a; free(p); }
static void *arena_memalign(arena_id_t a, size_t al, size_t n) { (void)a; return memalign(al, n); }
static size_t usable(usable_ptr_t p) { return malloc_usable_size((void *)p); }
static arena_id_t arena_create(arena_params_t *params) { (void)params; return 1; }
static void arena_dispose(arena_id_t a) { (void)a; }

static int arena_control(void)
{
    enum { ARENA = 2000, MOVES = 100 };
    static void *held[ARENA], *blk[MOVES];
    for (int i = 0; i < ARENA; ++i)
        held[i] = i % 3 == 0 ? T.moz_arena_malloc(1, 48) : i % 3 == 1 ? T.moz_arena_calloc(1, 4, 12)
                                                                       : T.moz_arena_memalign(1, 64, 48);
    for (int i = 0; i < MOVES; ++i) {
        blk[i] = T.moz_arena_malloc(1, 16);   /* a blocker after held[i], then growth that must move */
        void *p = T.moz_arena_realloc(1, held[i], 64 << 20);
        if (!p || p == held[i]) _exit(4);
        held[i] = p;
    }
    for (int i = 0; i < ARENA; ++i) T.moz_arena_free(1, held[i]);
    for (int i = 0; i < MOVES; ++i) T.moz_arena_free(1, blk[i]);
    printf("EXPECT peak_live %d\nEXPECT allocs %d\nEXPECT frees %d\nEXPECT realloc_moved %d\nEXPECT forked 0\n",
           ARENA + MOVES, ARENA + 2 * MOVES, ARENA + 2 * MOVES, MOVES);
    return 0;
}

/* control.c, with every allocator call routed through the table */
#define malloc(n) T.malloc(n)
#define calloc(a, b) T.calloc(a, b)
#define realloc(p, n) T.realloc(p, n)
#define free(p) T.free(p)
#define memalign(a, n) T.memalign(a, n)
#define aligned_alloc(a, n) T.aligned_alloc(a, n)
#define posix_memalign(o, a, n) T.posix_memalign(o, a, n)
#define malloc_usable_size(p) T.malloc_usable_size(p)
#define main control_main
#include "control.c"
#undef main
#undef malloc
#undef calloc
#undef realloc
#undef free
#undef memalign
#undef aligned_alloc
#undef posix_memalign
#undef malloc_usable_size

int main(int argc, char **argv)
{
    T.malloc = malloc;
    T.calloc = calloc;
    T.realloc = realloc;
    T.free = free;
    T.memalign = memalign;
    T.posix_memalign = posix_memalign;
    T.aligned_alloc = aligned_alloc;
    T.valloc = valloc;
    T.malloc_usable_size = usable;
    T.moz_create_arena_with_params = arena_create;
    T.moz_dispose_arena = arena_dispose;
    T.moz_arena_malloc = arena_malloc;
    T.moz_arena_calloc = arena_calloc;
    T.moz_arena_realloc = arena_realloc;
    T.moz_arena_free = arena_free;
    T.moz_arena_memalign = arena_memalign;
    struct ReplaceMallocBridge *bridge = NULL;
    replace_init(&T, &bridge);
    fflush(stdout);
    return argc > 1 && strcmp(argv[1], "arena") == 0 ? arena_control() : control_main();
}
