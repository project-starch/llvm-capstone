/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * The contract is the one in ../../cpython/pymalloc-repros/SCHEMA.md; this
 * header is its PostgreSQL seam. shared/driver.c supplies the entry point, the
 * root context and the labelled probes; a case supplies only its sequence,
 * inside PG_CASE(NN).
 *
 * The allocator is real: PostgreSQL 17.0's aset.c, mcxt.c and slab.c, compiled
 * unmodified but for the capability-ABI and Sublet patches the port applies.
 * The consumers are reduced to the allocator calls the upstream defect makes,
 * in the same order, because reaching them in place needs a backend, a planner,
 * a walsender or a concurrently dropped partition.
 */
#ifndef PG_CORPUS_H
#define PG_CORPUS_H

#include "postgres.h"
#include "utils/memutils.h"
#include "utils/memutils_internal.h"

/* A case declares the number its directory carries. The driver refuses a
 * fixture that names another case rather than silently running it. */
#define PG_CASE(n)                                                             \
  const unsigned pg_case_number = (n);                                         \
  void pg_case_run(void)

extern const unsigned pg_case_number;
void pg_case_run(void);

/* The root context the driver created, as TopMemoryContext. */
extern MemoryContext pg_root;

/* Where a case parks the alias it will read after the lifetime ends. Volatile
 * and external, so the sequence cannot be optimised into nothing. */
extern unsigned char *volatile pg_held;

MemoryContext pg_aset_child(MemoryContext parent, const char *name);

/* The stale access, labelled so a run can require the fault to land HERE
 * rather than merely somewhere. pg_mark() publishes the case and both probe
 * addresses and must be the LAST thing before the access: its presence is the
 * evidence that the setup ran. */
unsigned pg_probe(const volatile unsigned char *p);
void pg_write_probe(volatile unsigned char *p);
void pg_mark(void);

_Noreturn void pg_give_up(unsigned long code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      pg_give_up(n);                                                           \
  } while (0)

/* sizeof(ReorderBufferChange) in 17.0, measured on the host rather than
 * assumed. Slab needs one fixed size per context. */
#define PG_CHANGE_BYTES 80

#endif
