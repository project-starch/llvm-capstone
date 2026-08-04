/* lgi #122 — Lua record userdata ⟷ C cairo_region_t use-after-free.
 * Source: ../../lgi-122/boundary.md. valgrind: invalid READ size 4, 4 bytes
 * inside a 32-byte cairo_region_t freed by cairo_region_destroy.
 *
 * Two allocations: the cairo.Region lgi record userdata and the boxed
 * cairo_region_t (cairo_region_create).
 *   Free-site (record.c:139): collectgarbage("collect") finalizes the region
 *     record first -> record_gc -> record_free -> g_boxed_free ->
 *     cairo_region_destroy frees the 32-byte region.
 *   Stale-use (callable.c:943): the {}-proxy's __gc runs next -> r:get_extents()
 *     -> cairo_region_get_extents reads the freed region's extents fields.
 * READ size 4 at OFFSET 4 (an extents field) -> the interior address is formed
 * by cincoffset on the revoked capability (the emulator's assert-on-untagged
 * route; scored FAULT, see rows.tsv). Control: the read returns; row reports.
 */
#include "luac_shim.h"
#include <stdint.h>

#define CAIRO_REGION_BYTES 32
#define EXTENTS_OFF 4 /* the extents field valgrind names */

static volatile uint64_t sink;

int main(void) {
  unsigned char *region = (unsigned char *)malloc(CAIRO_REGION_BYTES);
  if (!region)
    abort();
  memset(region, 0, CAIRO_REGION_BYTES);

  unsigned char *proxy_ref = region; /* the {}-proxy borrows the same region */

  free(region); /* record_gc -> cairo_region_destroy -> REVOKE */

  /* cairo_region_get_extents reads an extents field at offset 4. */
  sink = *(volatile uint32_t *)(proxy_ref + EXTENTS_OFF); /* callable.c:943 */

  mock_report("luac_lgi_cairo_region_uaf", "use-after-free-survived");
  return 0;
}
