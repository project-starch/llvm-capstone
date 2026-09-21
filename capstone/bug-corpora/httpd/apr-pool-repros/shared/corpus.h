/* What a case in this corpus needs, so a case.c is a complete translation unit
 * on both targets.
 *
 * The contract is the one in ../../cpython/pymalloc-repros/SCHEMA.md; this
 * header is its APR seam. A case writes its sequence inside APR_CASE(NN). The
 * macro supplies aprp_replay for the port's one-source seam, so the Capstone
 * domain build compiles the file on its own into one program per defect; the
 * native driver in shared/driver.c calls the same body twice, buggy and fixed.
 *
 * WHAT IS REAL AND WHAT IS REDUCED. The allocator is real: apr_pools.c from
 * the pinned APR 1.7.4, unmodified but for the freestanding-include and
 * node-lifetime patches the port applies. The consumer is reduced to the pool
 * calls the upstream defect makes, in the same order; the per-case
 * PROVENANCE.md says what was left out.
 *
 * TWO TARGETS, SAME SEQUENCE. Natively the arms differ by the upstream FIX and
 * the driver prints what the case observed. In a Capstone domain the arms
 * differ by PROTECTION -- mode 0 spatial must complete, mode 1 sublet must
 * fault at the labelled probe -- and the program reports nothing about itself:
 * the runner reads the fault off the monitor and the completion off the
 * report. Only the probe and the marker are target-specific, and neither is a
 * case. */
#ifndef APR_CORPUS_H
#define APR_CORPUS_H
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include <string.h>

#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      aprp_fail(n);                                                            \
  } while (0)

/* What a case observed, in the terms its result line uses. The case fills the
 * flags and names its two verdicts; the driver prints, the domain does not. */
struct apr_outcome {
  int pool_struct_reissued;    /* the destroyed pool's node came back as another */
  int allocated_through_stale; /* apr_palloc through the dead handle succeeded */
  int other_pool_corrupted;    /* the live pool's bytes changed underneath it */
  const char *defect_text, *fixed_text;
};

/* The stale handle, kept where the compiler cannot reason it away. */
static volatile unsigned char *held __attribute__((used));

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere in the program. Natively it is a plain load. */
__attribute__((noinline, unused)) static unsigned
read_probe(const volatile unsigned char *p) {
#ifdef APRP_DOMAIN
  unsigned long value;
  __asm__ volatile(".globl apr_defect_read\napr_defect_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
#else
  return *p;
#endif
}
/* No case writes. The label must still exist, because mark() publishes both
 * addresses; `used` keeps --gc-sections from dropping it. */
__attribute__((used, noinline)) static void write_probe(volatile unsigned char *p) {
#ifdef APRP_DOMAIN
  unsigned long value = 93;
  __asm__ volatile(
      ".globl apr_defect_write\napr_defect_write:\nsb %0, 0(%1)\n" ::"r"(value),
      "r"(p)
      : "memory");
#else
  *p = 93;
#endif
}
/* Publish the case and both probe addresses through the monitor, so the host
 * knows which instruction a fault is allowed to be at. Nothing to publish
 * natively: the driver's output is the result there. */
__attribute__((unused)) static void mark(unsigned which) {
#ifdef APRP_DOMAIN
  extern void apr_defect_read(void), apr_defect_write(void);
  unsigned long code = 0xcf1a000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(apr_defect_read), "r"(apr_defect_write)
                   : "memory");
#else
  (void)which;
#endif
}

extern const int apr_case_number;
void apr_case_body(int fixed, apr_pool_t *root, struct apr_outcome *o);

/* One case per program. The body follows the macro:
 *
 *     #include "../shared/corpus.h"
 *     APR_CASE(0) {
 *       ... the sequence, buggy unless `fixed` ...
 *     }
 *
 * The fixture's event id must name this case, so a fixture built for another
 * case is refused rather than silently running this one. The domain arms
 * always run the buggy sequence; protection is the variable there, not the
 * fix. */
#define APR_CASE(number)                                                       \
  const int apr_case_number = (number);                                        \
  void aprp_replay(const struct aprp_header *input, struct aprp_header *out) { \
    unsigned mode = out->mode;                                                 \
    memset(out, 0, sizeof *out);                                               \
    out->mode = mode;                                                          \
    out->magic = APRP_MAGIC;                                                   \
    out->count = 1;                                                            \
    const struct aprp_event *e = (const void *)(input + 1);                    \
    CHECK(input->magic == APRP_MAGIC && input->count == 1 &&                   \
              e->id == (number),                                               \
          700);                                                                \
    apr_pool_t *root = NULL;                                                   \
    CHECK(apr_pool_create(&root, NULL) == APR_SUCCESS, 701);                   \
    struct apr_outcome outcome = {0};                                          \
    apr_case_body(0, root, &outcome);                                          \
    apr_pool_destroy(root);                                                    \
    /* Only the spatial arm is expected to arrive here. */                     \
    out->completed = 1;                                                        \
    aprp_stats(out);                                                           \
  }                                                                            \
  void apr_case_body(int fixed, apr_pool_t *root, struct apr_outcome *o)
#endif /* APR_CORPUS_H */
