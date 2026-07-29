#ifndef BLOBPEEK_KERNEL_H
#define BLOBPEEK_KERNEL_H
/* Return the value the DOMAIN actually sees in the monitor-copied blob.
 *
 * WHY. C-13 has been chased through four verified defects and still fails, because every
 * step so far INFERRED the blob's contents instead of observing them:
 *   stage 7  reads blob+0 (built_flag), branches on it        -> PASS
 *   stage 10 reads blob+8, DISCARDS the value                 -> PASS  (so the ACCESS works)
 *   stage 8  reads blob+8, uses it as `count`                 -> FAIL  (so the VALUE is wrong)
 * and stage 8 still fails after the monitor's copy was fixed to use scalar ld/sd.
 *
 * The trap is that `dom_data` is __get_free_pages memory the monitor never zeroes, and
 * fresh pages are USUALLY ZERO. So "blob+0 reads 0" is equally consistent with a correct
 * copy and with NO COPY AT ALL. If count also reads 0, the glue takes `beqz s4, 99f`,
 * skips the entire build, never establishes gp, and domain_main faults on its first
 * `ldc gp[i]` -- which is exactly the observed wedge, with no corruption required.
 *
 * This rung ends the guessing: the glue (INTERP_DIAG_STAGE=11) writes the 64-bit word it
 * read from blob+8 into this global's storage instead of zero-filling it, and
 * domain_main returns it. The harness prints retval, so the value appears in the log.
 *
 *   retval 1          -> the copy landed and count is correct; the fault is elsewhere
 *   retval 0          -> the blob is ZEROED/absent: the copy is not reaching the glue's
 *                        view of dom_data. Fix the copy's DESTINATION, not its width.
 *   anything else     -> the copy lands but is corrupted; that value names the corruption
 *
 * There is deliberately ONE global, so the descriptor is a single record and the value
 * cannot be confused with another slot's storage.
 */

/* NOT `static`, and `volatile`. A plain `static unsigned long` that no C code ever
   writes is provably zero to the compiler, so it folds the read to a constant, emits no
   `ldc gp[i]`, and the build gate rejects the domain (ldc-gp=0). The whole point is that
   this word is written BEHIND the compiler's back, by the entry glue. volatile forces the
   load; external linkage keeps it in the cap table. */
volatile unsigned long bp_slot;

static unsigned bp_compute(void) {
  /* Truncate to 32 bits because the harness's retval is unsigned. The values that matter
     (0, 1, small counts) survive; a large garbage value still reads as clearly not-1. */
  return (unsigned)bp_slot;
}
#endif
