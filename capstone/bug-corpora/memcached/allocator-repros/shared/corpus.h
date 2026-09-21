/* What a case in this corpus needs, so a case.c is a complete translation unit
 * on both targets.
 *
 * The contract is the one in ../../cpython/pymalloc-repros/SCHEMA.md; this
 * header is its memcached seam. A case writes its sequence inside MC_CASE(NN).
 * The macro supplies mcp_replay for the port's one-source seam, so the Capstone
 * domain build compiles the file on its own into one program per defect; the
 * native driver in shared/driver.c calls the same body twice, buggy and fixed.
 *
 * WHAT IS REAL AND WHAT IS REDUCED. The allocators are real: slabs.c and
 * cache.c from the pinned memcached 1.6.45, unmodified but for the shim and
 * lifetime-hook patches the port applies. The consumer is reduced to the
 * allocator calls the upstream defect makes, in the same order; the per-case
 * PROVENANCE.md says what was left out. memcached is a threaded server and
 * two of its defects are races; the reduction serialises the interleaving the
 * upstream commit describes, which is the one thing a single thread can do.
 *
 * TWO TARGETS, SAME SEQUENCE. Natively the arms differ by the upstream FIX and
 * the driver prints what the case observed. In a Capstone domain the arms
 * differ by PROTECTION -- mode 0 spatial must complete, mode 1 sublet must
 * fault at the labelled probe -- and the program reports nothing about itself:
 * the runner reads the fault off the monitor and the completion off the
 * report. Only the probe and the marker are target-specific, and neither is a
 * case. */
#ifndef MC_CORPUS_H
#define MC_CORPUS_H
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
#include "cache.h"
#include <string.h>

#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      mcp_fail(n);                                                             \
  } while (0)

/* What a case observed, in the terms its result line uses. The case fills the
 * flags and names its two verdicts; the driver prints, the domain does not.
 * `damage` is the case's own consequence -- what went wrong for the program
 * because of the stale access -- and case.json says what it means there. */
struct mc_outcome {
  int unit_reissued;           /* the freed chunk or object came back as another */
  int accessed_through_stale;  /* the consumer read through the dead pointer */
  int damage;                  /* the consequence the upstream report describes */
  const char *defect_text, *fixed_text;
};

/* The stale pointer, kept where the compiler cannot reason it away. A case
 * sets it WHILE the pointer is live: in the protected arm the pointer is a
 * revoked alias afterwards, and this emulator faults on arithmetic with one
 * (`cincoffsetimm with an UNTAGGED rs1`) as much as on a load through it, so
 * `&stale->field` computed after the free would fault before the probe and
 * the oracle would rightly refuse the run. Compare addresses, never pointers. */
static volatile unsigned char *held __attribute__((used));

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere in the program. Natively it is a plain load. */
__attribute__((noinline, unused)) static unsigned
read_probe(const volatile unsigned char *p) {
#if defined(MCP_DOMAIN)
  unsigned long value;
  __asm__ volatile(".globl mc_defect_read\nmc_defect_read:\nlbu %0, 0(%1)\n"
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
#if defined(MCP_DOMAIN)
  unsigned long value = 93;
  __asm__ volatile(
      ".globl mc_defect_write\nmc_defect_write:\nsb %0, 0(%1)\n" ::"r"(value),
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
#ifdef MCP_DOMAIN
  extern void mc_defect_read(void), mc_defect_write(void);
  unsigned long code = 0xcf1c000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(mc_defect_read), "r"(mc_defect_write)
                   : "memory");
#else
  (void)which;
#endif
}

extern const int mc_case_number;
void mc_case_body(int fixed, struct mc_outcome *o);

/* One case per program. The body follows the macro:
 *
 *     #include "../shared/corpus.h"
 *     MC_CASE(0) {
 *       ... the sequence, buggy unless `fixed` ...
 *     }
 *
 * The fixture's event id must name this case, so a fixture built for another
 * case is refused rather than silently running this one. The domain arms
 * always run the buggy sequence; protection is the variable there, not the
 * fix. slabs_init has already run, with upstream's defaults, in the entry. */
#define MC_CASE(number)                                                        \
  const int mc_case_number = (number);                                         \
  void mcp_replay(const struct mcp_header *input, struct mcp_header *out) {    \
    unsigned mode = out->mode;                                                 \
    memset(out, 0, sizeof *out);                                               \
    out->mode = mode;                                                          \
    out->magic = MCP_MAGIC;                                                    \
    out->count = 1;                                                            \
    const struct mcp_event *e = (const void *)(input + 1);                     \
    CHECK(input->magic == MCP_MAGIC && input->count == 1 &&                    \
              e->id == (number),                                               \
          700);                                                                \
    struct mc_outcome outcome = {0};                                           \
    mc_case_body(0, &outcome);                                                 \
    /* Only the spatial arm is expected to arrive here. */                     \
    out->completed = 1;                                                        \
    mcp_stats(out);                                                            \
  }                                                                            \
  void mc_case_body(int fixed, struct mc_outcome *o)
#endif /* MC_CORPUS_H */
