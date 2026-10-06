#include "corpus.h"
#include <stdint.h>

/* H266RawPPS's two adjacent derived-value arrays, as cbs_h266.h:594-595 declares
 * them at the pin. The struct is ONE allocation, so the bound between
 * col_width_val and row_height_val is not a bound any allocator knows about.
 *
 * VVC_MAX_TILE_COLUMNS and VVC_MAX_TILE_ROWS are enum constants in vvc.h. They
 * are REDUCED here: the defect is that the loop's bound check sits AFTER the
 * loop, so the array's real length changes how many iterations it takes to
 * leave, not whether it leaves. Stated rather than silent; the offsets are
 * asserted below. */
#define TILE_COLUMNS 20 /* reduced from VVC_MAX_TILE_COLUMNS */
#define TILE_ROWS 20    /* reduced from VVC_MAX_TILE_ROWS */

struct h266_raw_pps {
  uint16_t col_width_val[TILE_COLUMNS];
  uint16_t row_height_val[TILE_ROWS];
};

__attribute__((noinline, used)) static void
write_probe(volatile uint16_t *p, uint16_t v) {
  *p = v;
}

FF2_CASE(8) {
  /* Case 8 -- CBS H.266 PPS tile-column derivation, fix fb862976df. SUB-OBJECT:
   * a two-byte WRITE walking off the end of one struct member into the next,
   * inside a single allocation.
   *
   * The uniform-tile-spacing loop at the pin (cbs_h266_syntax_template.c:1899-1912)
   * is:
   *
   *     while (remaining_size > 0) {
   *         if (current->num_tile_columns > VVC_MAX_TILE_COLUMNS) { ... return; }
   *         unified_size = FFMIN(remaining_size, unified_size);
   *         current->col_width_val[i] = unified_size;
   *         remaining_size -= unified_size;
   *         i++;
   *     }
   *     current->num_tile_columns = i;
   *
   * The guard inside the loop tests num_tile_columns, which is not assigned
   * until AFTER the loop -- so during the loop it holds a stale value and the
   * check cannot fire. The real counter is `i`, and `i` is tested nowhere. A
   * picture wide enough in CTBs relative to the last explicit column width makes
   * the loop iterate past the array. The fix tests `i == VVC_MAX_TILE_COLUMNS`
   * inside the loop and deletes the now-redundant post-loop copy of the check.
   *
   * This is the gate-that-cannot-fire shape, in upstream code: a guard that is
   * present, reads correct, and is keyed to a variable the loop does not update.
   *
   * NOTHING WE HAVE CAN CATCH THIS: every write stays inside one allocation. */
  struct h266_raw_pps *pps = av_refstruct_allocz(sizeof *pps);
  CHECK(pps, 661);
  /* The claim: one past col_width_val IS row_height_val. */
  CHECK((char *)&pps->col_width_val[TILE_COLUMNS] == (char *)&pps->row_height_val[0],
        662);

  const uint16_t sentinel = 0xA5A5u;
  for (unsigned k = 0; k < TILE_ROWS; k++)
    pps->row_height_val[k] = sentinel;

  /* A picture 6 CTBs wider than the array can describe at this column width.
   * num_tile_columns holds its stale pre-loop value throughout, exactly as at
   * the pin. */
  const unsigned overrun = 6;
  unsigned remaining_size = TILE_COLUMNS + overrun;
  const unsigned unified_size = 1;
  unsigned i = 0, past = 0;
  unsigned num_tile_columns = 0; /* stale: assigned only after the loop */
  int stopped_early = 0;

  while (remaining_size > 0) {
    if (fixed) {
      if (i == TILE_COLUMNS) { /* the fix: keyed to i, inside the loop */
        stopped_early = 1;
        break;
      }
    } else {
      /* The pin's guard, verbatim in effect: keyed to the stale counter, so it
       * never fires. Kept in the arm so the two differ by the key, not by the
       * presence of a check. */
      if (num_tile_columns > TILE_COLUMNS)
        break;
    }
    if (i >= TILE_COLUMNS)
      past++;
    write_probe(&pps->col_width_val[i], (uint16_t)unified_size);
    remaining_size -= unified_size;
    i++;
  }
  num_tile_columns = i;

  unsigned clobbered = 0;
  for (unsigned k = 0; k < TILE_ROWS; k++)
    if (pps->row_height_val[k] != sentinel)
      clobbered++;

  printf("cap=%zu columns=%u past=%u clobbered=%u stale_guard=%u\n",
         sizeof pps->col_width_val, i, past, clobbered, num_tile_columns);

  FF2_VERDICT(!fixed && past == overrun && clobbered == overrun,
              fixed && stopped_early && clobbered == 0,
              "the loop's guard was keyed to num_tile_columns, which is assigned only "
              "after the loop, so 6 writes ran past col_width_val into row_height_val",
              "the fix keys the guard to i and tests it inside the loop");
  av_refstruct_unref(&pps);
  return !fixed ? !(clobbered == overrun) : !!clobbered;
}
