/* Stack use-after-return (CWE-562), the class of CRASH-100 in the blind set.
 *
 * This row is NOT about the nested allocator. It is here because it is the one
 * place CHERI is expected to do WORSE than ASAN, and a baseline that reports
 * only where CHERI matches is not a baseline. A dangling stack capability stays
 * tagged and in bounds: the frame is reused but never unmapped, and CheriBSD
 * does not revoke stack capabilities on return, so neither spatial bounds nor
 * heap revocation applies. ASAN catches this class with
 * detect_stack_use_after_return=1.
 *
 * THE ADDRESS IS SMUGGLED OUT THROUGH A GLOBAL, NOT RETURNED. `return buf;`
 * does not work: gcc hardens a returned local by substituting NULL, so the row
 * became a null-deref -- which CHERI *does* trap, on the tag. That would have
 * credited CHERI with catching a class it cannot catch. Writing &buf to a
 * volatile global keeps the real capability.
 *
 * Both allocator arms are identical here by construction, since the defect never
 * touches emalloc. That is the point: the arms are the allocator variable, and
 * this row sits outside it. */
#include "row.h"
#include <string.h>

static char *volatile escaped;

static void leak_frame(void)
{
    char buf[48];
    memset(buf, 'S', sizeof buf);
    buf[sizeof buf - 1] = 0;
    escaped = buf;                 /* capability to a frame about to die */
}

/* Called between the leak and the use so the frame is genuinely reused -- an
 * uninstrumented stack-UAR that still reads its own old bytes proves nothing. */
static void clobber_frame(void)
{
    volatile char other[48];
    int i;
    for (i = 0; i < 48; i++) { other[i] = (char) ('0' + (i & 7)); }
    row_sink += (unsigned long) other[0];
}

ROW_MAIN("c1_stack_after_return", {
    leak_frame();
    char *dangling = escaped;
    printf("DANGLING-NONNULL=%d\n", dangling != (char *) 0);
    clobber_frame();
    printf("STACK-AFTER-RETURN byte0=%02x (expect reuse, not 'S'=53)\n",
           (unsigned char) dangling[0]);
    row_consume(dangling, 48);
})
