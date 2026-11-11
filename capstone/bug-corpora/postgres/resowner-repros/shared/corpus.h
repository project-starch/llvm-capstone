/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * The contract is the one in ../../SCHEMA.md; this header is its ResourceOwner
 * seam.  shared/driver.c supplies the entry point, the root memory context, the
 * error jump target and the labelled probe; a case supplies only its sequence,
 * inside RO_CASE(NN).
 *
 * WHAT IS REAL HERE, AND IT IS MORE THAN IN mmgr-repros:
 * resowner.c itself is the subject and is compiled unmodified from the pin,
 * together with the seven memory-manager files it allocates through.  The
 * defect lives in ResourceOwnerReleaseAll's order of operations, so a model of
 * resowner would measure nothing.  What a case supplies is a ResourceOwnerDesc
 * -- a resource KIND -- which is what every consumer of resowner.c supplies,
 * and a payload for it to own.  PostgreSQL's own error handling is
 * sigsetjmp/siglongjmp, so elog(ERROR) is modelled as the longjmp it is, and
 * the second release is driven the way xact.c drives it during abort cleanup.
 */
#ifndef RO_CORPUS_H
#define RO_CORPUS_H

#include "postgres.h"
#include "utils/memutils.h"
#include "utils/resowner.h"

#define RO_CASE(n)                                                             \
  const unsigned ro_case_number = (n);                                         \
  void ro_case_run(void)

extern const unsigned ro_case_number;
void ro_case_run(void);

/* The root context the driver created, as TopMemoryContext. */
extern MemoryContext ro_root;

/* Drive ResourceOwnerRelease the way a backend does: once normally, and, if
 * that raises an error, once more from abort cleanup on the SAME owner --
 * which is the second half of what upstream describes for 011eedcdc3.
 * Returns the number of times the error path was taken. */
unsigned ro_release_with_abort_cleanup(ResourceOwner owner);

/* elog(ERROR) as PostgreSQL implements it: a siglongjmp out to the enclosing
 * cleanup. A release callback calls this to model any resource whose release
 * can fail. */
_Noreturn void ro_raise(const char *what);

/* The labelled site. For a double-free defect the stale access IS the second
 * release, so what a case marks is the free, not a read: ro_free_probe is where
 * the payload is handed back, so a fault lands in a named frame. ro_mark()
 * prints the case's ready mark and must be the LAST thing before the sequence
 * whose second pass is invalid. */
void ro_free_probe(void *payload);
void ro_mark(void);

/* How many times each payload was released, keyed by the payload pointer. A
 * second release of one payload is the defect, and this is how a case states
 * it without depending on a sanitizer being present. */
unsigned ro_release_count(const void *payload);
void ro_note_release(const void *payload);

_Noreturn void ro_give_up(unsigned long code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      ro_give_up(n);                                                           \
  } while (0)

#endif
