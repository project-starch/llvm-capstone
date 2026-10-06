#include "corpus.h"

WM_CASE(18) {
/* Row 18 -- RTPS DATA_BATCH, fix 716a200295 ("RTPS: Fix OOB write in
 * DATA_BATCH sample info list"). SPATIAL, and the row that fills a cell this
 * inventory recorded as EMPTY: a nested-spatial defect that is LIVE at the pin.
 *
 * dissect_RTPS_DATA_BATCH walks the batch's sample-info list and writes
 * sample_info_flags[sample_info_count] and sample_info_length[...], arrays it
 * allocated from the PACKET pool with sample_info_max entries. The loop's guard
 * at :16734 is:
 *
 *     if (rtps_max_batch_samples_dissected > 0 &&
 *         (unsigned)sample_info_count >= rtps_max_batch_samples_dissected) {
 *
 * THE GUARD IS A GATE THAT ITS OWN DEFAULT DISABLES. rtps_max_batch_samples_
 * dissected is a user preference, and when it is 0 -- meaning "no limit" -- the
 * first clause is false and the comparison is never reached, so nothing bounds
 * sample_info_count at all. The arrays are still only sample_info_max long
 * (1024 in exactly that case), so a batch announcing more samples writes
 * straight off the end.
 *
 * The fix guards on sample_info_max instead, and its own comment says why:
 * "Guarding on sample_info_max here (rather than on the preference) keeps the 0
 * case bounded too."
 *
 * This is the shape this project keeps paying for -- a check that is present,
 * reads correct, and cannot fire in the configuration that ships. It is worth a
 * row for that alone, independently of the crossing.
 *
 * NESTED: the arrays come from the packet pool, so the write leaves its chunk
 * and lands in storage the SAME block handed out next. A malloc-granular bound
 * cannot see it; the block is one g_malloc. */
#define SAMPLE_INFO_MAX 8 /* reduced from 1024: the defect is that NOTHING bounds
                           * the counter, so the array's length changes how many
                           * iterations it takes to leave, not whether it does */
  /* The two arrays, carved consecutively from the packet pool exactly as
   * :16710-16711 does. */
  unsigned *flags = wmem_alloc(wm_packet, SAMPLE_INFO_MAX * sizeof *flags);
  CHECK(flags, 1);
  unsigned *lengths = wmem_alloc(wm_packet, SAMPLE_INFO_MAX * sizeof *lengths);
  CHECK(lengths, 2);
  /* The storage the overrun lands in: the next chunk of the same block. Its
   * position is asserted BEFORE anything is written, so the marker's presence
   * is evidence the crossing was created rather than merely described. */
  unsigned char *successor = wmem_alloc(wm_packet, 64);
  CHECK(successor, 3);
  CHECK((uintptr_t)successor > (uintptr_t)lengths, 4);
  memset(successor, 0x5a, 64);

  /* The preference at its default. This single value is the defect. */
  const unsigned rtps_max_batch_samples_dissected = 0;
  /* The guard as the pin has it. With the preference 0 it can never fire, which
   * the case asserts rather than assumes. */
  unsigned count = SAMPLE_INFO_MAX; /* a batch that has already filled the array */
  int guard_fires = (rtps_max_batch_samples_dissected > 0 &&
                     count >= rtps_max_batch_samples_dissected);
  CHECK(!guard_fires, 5);
  /* The fix's guard WOULD fire here, which is the other half of the claim. */
  CHECK(count >= SAMPLE_INFO_MAX, 6);

  /* The write the loop performs at :16740 once the guard has let it through.
   * Index SAMPLE_INFO_MAX of the lengths array is the successor's first word. */
  CHECK((uintptr_t)&lengths[SAMPLE_INFO_MAX] + sizeof(unsigned)
            <= (uintptr_t)successor + 64, 7);
  wm_held = (unsigned char *)&lengths[count];
  wm_mark();
  wm_write_probe(wm_held);
}
