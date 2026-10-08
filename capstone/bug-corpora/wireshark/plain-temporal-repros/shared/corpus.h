/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLAIN-TEMPORAL corpus, and deliberately a sibling of ../wmem-repros rather than part
 * of it: that corpus's objects are chunks wmem's BLOCK or BLOCK_FAST allocator carved out of a block g_malloc handed out. Here the object IS the g_malloc, with no wmem layer -- which is why these cases sit in wiretap and wsutil rather than in epan.
 *
 * It exists because the inventory's temporal x PLAIN-allocator cell was EMPTY for all three
 * target programs, while every temporal corpus in the tree sat on a nested allocator. That zero
 * was a property of which corpora existed, not of the upstream software -- Wireshark frees direct
 * allocations and uses them afterwards like any C program.
 *
 * THE OBSERVABLE IS ALIASING, NOT ORDERING. A read of freed memory is undefined but usually
 * quiet, so a case that only recorded "the access came after the free" would be asserting its own
 * construction rather than measuring anything. Each case instead:
 *
 *   1. frees the object,
 *   2. takes a FRESH allocation of the same size, which glibc's tcache satisfies from the very
 *      chunk just freed, and fills it with a marker,
 *   3. reads through the STALE pointer and finds the marker.
 *
 * The stale pointer now names a different live object. That is deterministic, it is the thing
 * that makes a use-after-free exploitable, and it is two-sided: under the upstream fix the stale
 * pointer either is not retained or is not followed, so no marker is seen.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case supplies its sequence
 * inside WST_CASE(NN) and fills the outcome the driver prints.
 */
#ifndef WST_CORPUS_H
#define WST_CORPUS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WST_CASE(n)                                                            \
  const int wst_case_number = (n);                                             \
  void wst_case_run(int fixed, struct wst_outcome *o)

/* What a case observed.
 *   `aliased`  THE CLAIM: the stale pointer read storage that now belongs to a
 *              different live object. This is what the verdict turns on.
 *   `freed`    the lifetime ender ran. Recorded so a case that never freed is
 *              visibly INCONCLUSIVE rather than quietly passing.
 *   `marker`   what the fresh object was filled with, and
 *   `observed` what the stale read returned. Equal means aliased.
 *   `bytes`    the allocation's size, which is what makes the reuse land. */
struct wst_outcome {
  int aliased;
  int freed;
  int damage;
  unsigned long bytes;
  unsigned marker, observed;
  const char *defect_text, *fixed_text;
};

extern const int wst_case_number;
void wst_case_run(int fixed, struct wst_outcome *o);

_Noreturn void wst_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      wst_fail(n);                                                            \
  } while (0)

/* The accesses through the stale pointer, labelled so a sanitiser's report or a
 * capability fault can be required to land HERE rather than merely somewhere in
 * the program.
 *
 * DECLARED here and DEFINED once in shared/driver.c, never `static` in the
 * header: a per-translation-unit copy cannot be resolved unambiguously by
 * `supervise`, which is why some older corpora's CheriBSD rows read
 * "attribution: not established". */
unsigned wst_read_probe(const volatile unsigned char *p);
void wst_write_probe(volatile unsigned char *p, unsigned char v);

#define read_probe wst_read_probe
#define write_probe wst_write_probe

#endif
