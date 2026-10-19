/* What bound does CheriBSD's malloc actually hand out, and how far can a
 * pointer leave it before the tag goes?
 *
 * Two corpus-wide claims rest on this and neither can be reasoned from source:
 *
 *   1. ROUNDING. malloc(n) returns a capability longer than n, so an overread
 *      of a few bytes can be invisible to the bounds check. A cell that faults
 *      si_code=2 on a defect whose access lies inside that slack did NOT have
 *      its access rejected -- the access succeeded and something downstream
 *      faulted on what it returned.
 *   2. REPRESENTABILITY. A cursor moved far outside the bounds stops being
 *      representable in a compressed 128-bit capability, and the arithmetic
 *      itself clears the tag. An access that far out therefore reports
 *      si_code=2 (PROT_CHERI_TAG) even though the bounds rule is what stopped
 *      it, so a review that classifies cells by fault code alone will withdraw
 *      a real detection.
 *
 * Build with build-helpers.sh, push, run. It takes no arguments: the sizes are
 * the ones the three corpora's cases actually allocate, so a reader can match
 * a row to a case. Print, do not assert -- the numbers are the allocator's,
 * and the point is to record what this build does, not to require a value.
 */
#include <cheri/cheric.h>
#include <stdio.h>
#include <stdlib.h>

static const size_t SIZES[] = {
    4, 16, 24, 48, 97, 100, 184, 200, 216, 400, 512, 513, 8192, 8193, 524321,
};
static const size_t OFFSETS[] = {
    0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576,
};

int main(void)
{
    printf("%-10s %-10s %-8s  tag at offset ...\n", "request", "length", "slack");
    for (unsigned i = 0; i < sizeof SIZES / sizeof *SIZES; i++) {
        size_t n = SIZES[i];
        char *p = malloc(n);
        if (!p) { printf("%-10zu malloc failed\n", n); continue; }
        size_t len = cheri_length_get(p);
        printf("%-10zu %-10zu %-8ld ", n, len, (long)len - (long)n);
        /* just past the request, and just past the capability: the first is
         * the slack a small overread hides in, the second is where a bounds
         * fault becomes possible at all. */
        printf("+n=%d +len=%d ", (int)cheri_tag_get(p + n), (int)cheri_tag_get(p + len));
        for (unsigned j = 0; j < sizeof OFFSETS / sizeof *OFFSETS; j++)
            printf("+%zu=%d ", OFFSETS[j], (int)cheri_tag_get(p + OFFSETS[j]));
        printf("\n");
        free(p);
    }
    return 0;
}
