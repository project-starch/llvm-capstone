/* MEASURE the capability bounds CheriBSD's malloc actually hands out, instead of
 * inferring them from which over-read happens to fault.
 *
 * This exists because b1/b2 MISSED in the `patched` arm, where the prediction
 * was BLOCKED-SYNC: with REAL_SIZE(n)=n, reading s[n] should be past the object.
 * b3 faulting at s[32] but not s[26..31] suggested malloc bounds the SIZE CLASS
 * rather than the request -- but that is a guess until the bound is read off the
 * capability itself.
 *
 * __builtin_cheri_length_get / _base_get / _address_get report the real thing.
 * For each size this prints:
 *   mlen  - length of the capability malloc returned for hdr+REAL_SIZE(n)
 *   want  - what was asked for (hdr + REAL_SIZE(n))
 *   slack - mlen - want, i.e. how far past the request the bound reaches
 *   dlen  - remaining length from the emalloc'd data pointer
 * `slack > 0` means an over-read of up to that many bytes past the allocation is
 * IN BOUNDS and therefore invisible to CHERI, whatever PHP's allocator does.
 */
#include "row.h"
#include <stdlib.h>
#include <string.h>

ROW_MAIN("z1_bounds_probe", {
    static const size_t sizes[] = { 1, 7, 8, 9, 17, 26, 31, 32, 33, 48, 74, 100, 1000 };
    size_t i;
    printf("PROBE hdr=%zu\n", php_alloc_header_bytes());
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        size_t n    = sizes[i];
        size_t rs   = php_alloc_real_size(n);
        size_t want = php_alloc_header_bytes() + rs;
        void  *raw  = malloc(want);
        if (!raw) { continue; }
        char  *d    = (char *) emalloc(n);
        /* Print the capability fields RAW. An earlier version printed a derived
         * "remaining length" and produced a number that contradicted the
         * malloc length on the same line, so the arithmetic -- not the
         * platform -- was the thing in doubt. Derive nothing here; let the
         * host do it from base/addr/len. */
        printf("PROBE n=%zu rs=%zu want=%zu"
               " rawbase=%lu rawlen=%lu"
               " dbase=%lu daddr=%lu dlen=%lu\n",
               n, rs, want,
               (unsigned long) __builtin_cheri_base_get(raw),
               (unsigned long) __builtin_cheri_length_get(raw),
               (unsigned long) __builtin_cheri_base_get(d),
               (unsigned long) __builtin_cheri_address_get(d),
               (unsigned long) __builtin_cheri_length_get(d));
        free(raw);
        efree(d);
    }
    printf("PROBE-DONE\n");
})
