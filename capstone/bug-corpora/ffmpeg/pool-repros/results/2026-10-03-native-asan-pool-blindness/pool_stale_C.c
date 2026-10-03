/* Positive-control experiment for "pooled memory is invisible to ASan at the pin".
 *
 * Arm A: linked against the pin's unpatched libavutil  -> PREDICT: ASan clean.
 * Arm B: same, with e6255fb822 applied to refstruct.c  -> PREDICT: use-after-poison.
 *
 * The read at STALE READ below is a use-after-return-to-pool: the entry has gone back
 * onto pool->available_entries and no free() ever happened, which is exactly why a
 * free-keyed tool cannot see it. The pool is created with flags 0, so it has no
 * free_entry_cb and therefore IS covered by e6255fb822's `if (!pool->free_entry_cb)`.
 */
#include <stdio.h>
#include <string.h>
#include "libavutil/refstruct.h"

#define TAB 32

/* Arm C: the ONLY difference from armA/armB -- this pool carries a free_entry_cb,
 * which is exactly what e6255fb822 gates its poisoning on. */
static void noop_free_entry(AVRefStructOpaque o, void *obj) { (void)o; (void)obj; }


int main(void)
{
    AVRefStructPool *pool = av_refstruct_pool_alloc_ext(TAB, 0, NULL, NULL, NULL, noop_free_entry, NULL);
    if (!pool) { puts("E1 pool"); return 1; }

    unsigned char *a = av_refstruct_pool_get(pool);
    if (!a) { puts("E2 get"); return 2; }
    memset(a, 0xA0, TAB);
    printf("live_read=0x%02X\n", a[0]);

    unsigned char *held = a;          /* the consumer keeps using this */
    av_refstruct_unref(&a);           /* back onto pool->available_entries; NO free() */
    printf("released=%d\n", a == NULL);

    /* STALE READ -- the entry is resting in the pool right now. */
    printf("stale_read=0x%02X\n", held[0]);

    /* And the pool hands the same storage straight back out. */
    unsigned char *b = av_refstruct_pool_get(pool);
    if (!b) { puts("E3 reget"); return 3; }
    printf("reuse_same_address=%d\n", b == held);
    memset(b, 0xCC, TAB);
    printf("after_new_owner_stale_read=0x%02X\n", held[0]);

    av_refstruct_unref(&b);
    av_refstruct_pool_uninit(&pool);
    puts("DONE");
    return 0;
}
