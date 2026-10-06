/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLAIN-HEAP corpus, and deliberately a sibling of ../wmem-repros rather than a
 * part of it. That corpus's boundary is wmem -- a chunk carved from a block the
 * system allocator handed out -- and all eighteen of its cases cross a bound wmem
 * owns. The defects here cross the `malloc` bound ITSELF, in wiretap, which does
 * not use wmem at all: it calls g_malloc directly for its page buffers. So the
 * allocator under test is the system's and there is no inner layer to port.
 *
 * It is the mirror of ../../memcached/plain-heap-repros, built for the same
 * reason: the inventory's not-nested spatial row stood empty because the hunt had
 * required its candidates to be LIVE AT THE PIN. They need not be -- the
 * convention is stated at ../../memcached/allocator-repros/README.md:132-135, and
 * most cases in this tree are fix-reversals. A fix that is an ancestor of the pin
 * is reconstructed by running the pre-fix consumer shape against the shipped
 * allocator.
 *
 * GLib is not linked. g_malloc is malloc plus abort-on-failure and g_free is
 * free, which is all these cases use of it; the per-case PROVENANCE.md says so
 * rather than leaving a reader to assume it.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case
 * supplies its sequence inside WSH_CASE(NN) and fills the outcome the driver
 * prints.
 */
#ifndef WSH_CORPUS_H
#define WSH_CORPUS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WSH_CASE(n)                                                            \
  const int wsh_case_number = (n);                                             \
  void wsh_case_run(int fixed, struct wsh_outcome *o)

/* What a case observed. `crossed` is the claim: the access left the allocation.
 * `extent` is how far past it would have run unreduced, because the reduction
 * probes only the first crossing byte and the magnitude is part of the finding. */
struct wsh_outcome {
  int crossed;                 /* the access left the allocation */
  int damage;                  /* the consequence the upstream report describes */
  unsigned long cap;           /* the allocation's size */
  unsigned long touched;       /* the offset the consumer reached */
  long extent;                 /* bytes past the allocation the unreduced copy spans */
  const char *defect_text, *fixed_text;
};

extern const int wsh_case_number;
void wsh_case_run(int fixed, struct wsh_outcome *o);

_Noreturn void wsh_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      wsh_fail(n);                                                             \
  } while (0)

/* The crossing, labelled so a sanitiser's report can be required to land HERE
 * rather than merely somewhere in the program. */
__attribute__((noinline, used)) static unsigned
read_probe(const volatile unsigned char *p) {
  return *p;
}

#endif
