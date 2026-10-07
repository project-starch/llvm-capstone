/* What extent does ONE SQLite allocation actually carry, under each arm?
 *
 * WHY THIS EXISTS. This corpus's three arms differ by which allocator is under
 * the engine, and the verdicts alone do not show what each one hands out. Two
 * readings in particular rest on it, and both would otherwise be an argument
 * from source rather than a measurement:
 *
 *   * why the unprotected nested arm loses 13 of the 22 the system allocator
 *     catches -- if every sqlite3_malloc returns an interior pointer into ONE
 *     object, there is no per-allocation bound to cross and no free to revoke;
 *   * why the Sublet arm regains 12 of those 13 but not case 30, a spatial
 *     over-read -- memsys5 rounds a request up to a power-of-two multiple of
 *     its atom and the port issues THE BLOCK, so a short over-read lands in
 *     the round-up rather than past a bound.
 *
 * WHAT IT PRINTS. One line per request size, for a buffer sqlite3_malloc hands
 * out, with the extent read off the capability itself rather than from a size
 * the program remembers:
 *
 *   BOUNDS request=100 length=128 slack=28 ROUNDED-UP
 *
 * EXACT means the extent is the request; ROUNDED-UP means the allocator issued
 * more than was asked for, and the slack is how far a crossing may go before
 * any mechanism can see it; WHOLE-REGION means the extent is far larger than
 * the request, which is what an interior pointer into one object carries.
 *
 * It is built and run exactly as a case of this corpus is, through the same
 * repro_init, so the allocator measured is the allocator the cases ran on. It
 * is an instrument, not a case: no fix/defect pair, no verdict.
 */
#include "repro322_common.h"
#include <capstone/capability.h>

static const unsigned sizes[] = {8, 24, 40, 64, 100, 200, 1000, 4000};

static int run_case(void) {
  /* The cases call this themselves and so must the instrument: REPRO322_MAIN
   * does not. Without it SQLite is never configured, every arm falls through
   * to the platform allocator and all three print the same line -- which is
   * what the first run of this probe did, and the reason it is called here
   * with its result checked rather than ignored. */
  if (repro_init())
    return 1;
  for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; ++i) {
    void *p = sqlite3_malloc((int)sizes[i]);
    if (!p) {
      out_text("BOUNDS request="); out_uint(sizes[i]);
      out_text(" REFUSED\n");
      continue;
    }
    /* capstone_cap_store puts the register's capability into a slot; the
     * metadata readers restore it after each read. The alias is non-linear, so
     * p stays usable and the free below is the ordinary one. */
    capstone_cap_slot slot;
    capstone_cap_store(&slot, p);
    unsigned long base = capstone_cap_base(&slot);
    unsigned long length = capstone_cap_end(&slot) - base;
    out_text("BOUNDS request="); out_uint(sizes[i]);
    out_text(" length="); out_uint(length);
    out_text(" slack="); out_uint(length > sizes[i] ? length - sizes[i] : 0);
    out_text(" base="); out_uint(base);
    out_text(length == sizes[i] ? " EXACT"
             : length > 16UL * sizes[i] + 65536UL ? " WHOLE-REGION"
             : " ROUNDED-UP");
    out_text("\n");
    sqlite3_free(p);
  }
  return 0;
}
REPRO322_MAIN("allocation-bounds")
