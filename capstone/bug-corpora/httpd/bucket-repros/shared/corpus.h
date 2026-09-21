/* What a case in this corpus needs, so a case.c is a complete translation unit
 * on every target.
 *
 * The contract is the one in ../../cpython/pymalloc-repros/SCHEMA.md; this
 * header is its APR-bucket seam. A case writes its sequence inside
 * APRB_CASE(NN). The macro supplies aprp_replay for the port's one-source
 * seam, so the Capstone domain build compiles the file on its own into one
 * program per defect; the native driver in shared/driver.c calls the same body
 * twice, buggy and fixed.
 *
 * REAL AND REDUCED. Both allocators are real: apr-util 1.6.3's
 * apr_buckets_alloc.c over APR 1.7.4's apr_pools.c, upstream byte for byte
 * but for the port's freestanding-include and lifetime-hook patches. Brigades,
 * filters, connections and requests are reduced to the holder and the storage:
 * what every one of these cases turns on is which pool the holder is on and
 * which allocator the storage came from, and neither changes what the
 * allocators are asked for or when they are asked to take it back.
 *
 * THREE TARGETS, SAME SEQUENCE. Natively the arms differ by the upstream FIX
 * and the driver prints what the case observed. In a Capstone domain the arms
 * differ by PROTECTION -- mode 0 spatial must complete, mode 1 sublet must
 * fault at the labelled probe, or complete where the case's oracle says the
 * reduced sequence ends no lifetime -- and the program reports nothing about
 * itself: the runner reads the fault off the monitor and the completion off
 * the report. On CheriBSD the same hosted program runs in mode 0 against the
 * platform's own malloc with a supervisor outside it. Only the probe and the
 * marker are target-specific, and neither is a case. */
#ifndef APRB_CORPUS_H
#define APRB_CORPUS_H
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include "apr_bucket_shim.h"
#include <string.h>

#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      aprp_fail(n);                                                            \
  } while (0)

/* What a case observed, in the terms its result line uses. The case fills the
 * flags and decides its two verdicts; the driver prints, the domain does not. */
struct aprb_outcome {
  int through_dead_allocator; /* storage held from an allocator already destroyed */
  int still_held;             /* the holder still names storage whose lifetime ended */
  int reissued_same_address;  /* that storage came back to another owner */
  int read_after_destroy;     /* the holder itself was read after its pool went */
  int now;                    /* the byte the stale read returned, or -1 */
  int defect, held_up;        /* the case's own verdicts, buggy and fixed arm */
  const char *defect_text, *fixed_text;
};

/* The stale handle, kept where the compiler cannot reason it away. */
static volatile unsigned char *held __attribute__((used));

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere in the program. Natively it is a plain load. */
__attribute__((noinline, unused)) static unsigned
read_probe(const volatile unsigned char *p) {
#if defined(APRP_DOMAIN)
  unsigned long value;
  __asm__ volatile(".globl apr_defect_read\napr_defect_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
#elif defined(__CHERI_PURE_CAPABILITY__)
  unsigned long value;
  __asm__ volatile(".globl apr_defect_read\napr_defect_read:\nclbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "C"(p)
                   : "memory");
  return value;
#else
  return *p;
#endif
}
/* No case writes. The label must still exist, because mark() publishes it;
 * `used` keeps --gc-sections from dropping it. */
__attribute__((used, noinline)) static void write_probe(volatile unsigned char *p) {
#if defined(APRP_DOMAIN)
  unsigned long value = 93;
  __asm__ volatile(".globl apr_defect_write\napr_defect_write:\nsb %0, 0(%1)\n" ::"r"(value),
                   "r"(p)
                   : "memory");
#elif defined(__CHERI_PURE_CAPABILITY__)
  unsigned long value = 93;
  __asm__ volatile(".globl apr_defect_write\napr_defect_write:\ncsb %0, 0(%1)\n" ::"r"(value),
                   "C"(p)
                   : "memory");
#else
  *p = 93;
#endif
}
/* Publish the case and the three labelled sites -- the read, the write and
 * the bucket allocator's own probe in apr_bucket_free -- through the monitor,
 * so the host knows which instruction a fault is allowed to be at. */
__attribute__((unused)) static void mark(unsigned which) {
#ifdef APRP_DOMAIN
  extern void apr_defect_read(void), apr_defect_write(void), aprb_free_probe(void);
  unsigned long code = 0xcf1c000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %3, x0\n" ::"r"(code),
                   "r"(apr_defect_read), "r"(apr_defect_write), "r"(aprb_free_probe)
                   : "memory");
#else
  (void)which;
#endif
}

/* The part of a brigade these defects touch: a holder, on a pool, of pointers
 * into bucket-allocator storage. apr_brigade.c is not ported and is not
 * needed. */
#define APRB_SLOTS 8
struct brigade {
  apr_pool_t *pool;                     /* the brigade's own lifetime */
  void *bucket[APRB_SLOTS];             /* storage from a bucket allocator */
  apr_bucket_alloc_t *from[APRB_SLOTS]; /* which allocator issued it */
  int n;
};

extern const int aprb_case_number;
void aprb_case_body(int fixed, apr_pool_t *root, struct aprb_outcome *o);

/* The seam's entry, on every target: the domain calls it from its entry, the
 * hosted main.c (CheriBSD, under the supervisor) from a trace file, and the
 * native driver leaves it unused beside its own main. */
#define APRB_DOMAIN_ENTRY(number)                                              \
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
    struct aprb_outcome outcome = {0};                                         \
    aprb_case_body(0, root, &outcome);                                         \
    apr_pool_destroy(root);                                                    \
    /* Only an arm whose oracle says "complete" is expected to arrive here. */ \
    out->completed = 1;                                                        \
    aprp_stats(out);                                                           \
  }

/* One case per program:
 *
 *     #include "../shared/corpus.h"
 *     APRB_CASE(0) {
 *       ... the sequence, buggy unless `fixed` ...
 *     }
 *
 * The fixture's event id must name this case, so a fixture built for another
 * case is refused rather than silently running this one. The domain arms
 * always run the buggy sequence; protection is the variable there. */
#define APRB_CASE(number)                                                      \
  const int aprb_case_number = (number);                                       \
  APRB_DOMAIN_ENTRY(number)                                                     \
  void aprb_case_body(int fixed, apr_pool_t *root, struct aprb_outcome *o)
#endif /* APRB_CORPUS_H */
