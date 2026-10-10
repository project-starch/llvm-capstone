/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLAIN-HEAP corpus, and deliberately a sibling of ../allocator-repros rather
 * than a part of it. That corpus's boundary is memcached's NESTED allocators --
 * slabs.c and the per-thread object cache -- and every one of its eight cases
 * crosses a bound those allocators own. The defects here cross the `malloc`
 * bound ITSELF, so the allocator under test is the system's, there is no inner
 * layer to port, and pulling in slabs.h would be dead weight that implied a
 * nesting the defect does not have.
 *
 * That is also why this corpus exists at all: it is the not-nested spatial row
 * of docs/ref/spatial-and-temporal-bug-inventory.md, which stood empty because
 * the hunt had required its candidates to be LIVE AT THE PIN. They need not be
 * -- 27 of the 33 cases in this tree carry live_in_pin: false, and the
 * convention is stated at ../allocator-repros/README.md:132-135. A fix that is
 * an ancestor of the pin is reconstructed by running the pre-fix consumer shape
 * against the shipped allocator, which is exactly what this corpus does.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case
 * supplies its sequence inside MCH_CASE(NN) and fills the outcome the driver
 * prints.
 */
#ifndef MCH_CORPUS_H
#define MCH_CORPUS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A case declares the number its directory carries; the driver refuses a
 * fixture that names another case rather than silently running it. */
#define MCH_CASE(n)                                                            \
  const int mch_case_number = (n);                                             \
  void mch_case_run(int fixed, struct mch_outcome *o)

/* What a case observed, in the terms its result line uses. The case fills the
 * flags and names its two verdicts; the driver prints. `damage` is the case's
 * own consequence -- what went wrong for the program because of the crossing. */
struct mch_outcome {
  int crossed;                 /* the access left the allocation */
  int damage;                  /* the consequence the upstream report describes */
  unsigned long cap;           /* the allocation's size, as the arm chose it */
  unsigned long touched;       /* the offset the consumer actually wrote */
  const char *defect_text, *fixed_text;
};

extern const int mch_case_number;
void mch_case_run(int fixed, struct mch_outcome *o);

_Noreturn void mch_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      mch_fail(n);                                                             \
  } while (0)

/* The crossing, labelled so a sanitiser's report or a capability fault can be
 * required to land HERE rather than merely somewhere in the program.
 *
 * DECLARED here and DEFINED once in shared/driver.c. It was `static` in this
 * header until 2026-10-07, which gave each translation unit a private copy and
 * left `supervise` unable to resolve the symbol -- the reason this corpus's
 * CheriBSD row read "attribution: not established". One external definition
 * makes the probe address resolvable from the ELF, so a fault can be required
 * to land inside the labelled probe. The sibling ffmpeg/plain-heap-repros made
 * the same change and its native readings came out byte-identical, so the
 * change is inert to everything except attribution. */
void mch_write_probe(volatile unsigned char *p, unsigned char v);
#define write_probe mch_write_probe

/* The read crossing, on the same terms. Added 2026-10-08: this corpus began with
 * write-only defects, but three of memcached's reducible plain-heap defects are
 * over-READS -- a "%s" conversion handed a key that is not NUL-terminated, a
 * memchr whose remaining length underflows, and a backwards list shuffle that
 * reads one element past. The sibling ffmpeg/plain-heap-repros has carried both
 * probes from the start. Cases built before this do not reference it, so their
 * native readings must be unchanged by its addition; that was verified by
 * re-running runners/run-native.sh over cases 00 and 01. */
unsigned mch_read_probe(const volatile unsigned char *p);
#define read_probe mch_read_probe

#endif
