/* Shared scaffolding for the twenty cases.
 *
 * Each case directory holds its own case.c: the claim (case.json), its
 * provenance (PROVENANCE.md) and its sequence, together. A case.c includes
 * this header and writes its sequence inside PYC_CASE; the macro supplies
 * pym_replay, so the file is a complete translation unit that the port's
 * one-source seam builds on its own into one program per defect.
 *
 * WHAT IS REAL AND WHAT IS REDUCED. The allocator is real: obmalloc.c from the
 * pinned CPython 3.13.7, compiled unmodified but for the capability-ABI and
 * lifetime patches the port applies. The consumers are reduced to the
 * allocator calls the upstream defect makes, in the same order, because
 * reaching them in place needs a running interpreter -- which the port
 * explicitly does not put in a domain. Each case names its upstream fix; the
 * per-case PROVENANCE.md says line by line what was reduced.
 *
 * TARGETS. The same sequences build for the virtual Capstone profile (the
 * capstone-application preset, with and without the port's patch 0003), for
 * stock CheriBSD purecap and natively. Only the probe instructions differ: on
 * a purecap ABI they are capability-base accesses (clbu/csb), so the label
 * sits on an instruction a fault's pc can be compared against.
 *
 * SIZE. Every block is well under pymalloc's 512-byte threshold, so all of it
 * is pool memory that never reaches malloc. Cases 6, 10 and 19 have upstream
 * defects that can exceed it; for those the same defect becomes an ordinary
 * malloc use-after-free that ASan does see, which is why the size is pinned
 * here and stated in each of their PROVENANCE.md files.
 */
#ifndef PYC_CORPUS_H
#define PYC_CORPUS_H
#include "port.h"
#include <string.h>

#define CHECK(x, n)                                                            \
  do {                                                                         \
    if (!(x))                                                                  \
      pym_fail(n);                                                             \
  } while (0)

/* Small enough to be a pymalloc block on any of the port's size classes, and
 * the same for every case so that a freed block is reused by the next
 * allocation rather than landing in a different class. */
#define OBJ 48

static volatile unsigned char *held;

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere in the program. */
__attribute__((noinline)) static unsigned
read_probe(const volatile unsigned char *p) {
  unsigned long value;
#if defined(__CHERI_PURE_CAPABILITY__)
  /* Capability-base byte load. The label sits ON the faulting instruction; a
   * "C" operand keeps the stale capability itself as the base. On a purecap
   * ABI a pointer cannot be held in an integer register, so an "r" operand
   * fails to compile there. */
  __asm__ volatile(".globl pyc_defect_read\npyc_defect_read:\nclbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "C"(p)
                   : "memory");
#else
  __asm__ volatile(".globl pyc_defect_read\npyc_defect_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
#endif
  return value;
}

/* No case writes, so nothing calls this; `used` keeps the pyc_defect_write
 * label in the image for any case that does. */
__attribute__((used, noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long value = 93;
#if defined(__CHERI_PURE_CAPABILITY__)
  __asm__ volatile(".globl pyc_defect_write\npyc_defect_write:\n"
                   "csb %0, 0(%1)\n" ::"r"(value),
                   "C"(p)
                   : "memory");
#else
  __asm__ volatile(
      ".globl pyc_defect_write\npyc_defect_write:\nsb %0, 0(%1)\n" ::"r"(value),
      "r"(p)
      : "memory");
#endif
}

/* The point a case reaches just before its stale access. Runners observe the
 * fault from outside -- the launcher or the CheriBSD supervisor -- and resolve
 * its pc against pyc_defect_read in the image, so nothing is published. */
static void mark(unsigned which) { (void)which; }

/* An odict node: a link the loop follows, and a payload. Sized so the whole
 * node is one pymalloc block. */
struct node {
  struct node *next;
  unsigned char payload[16];
};


/* The program reports nothing about its own outcome on either target: a
 * protected arm is judged by the fault the kernel delivers, a spatial arm by
 * the 96-byte report it writes out. */
#define PYC_INSTALL_HANDLER(n) ((void)0)
#define PYC_COMPLETED(n) ((void)0)

/* One case per program. The body follows the macro:
 *
 *     #include "../shared/corpus.h"
 *     PYC_CASE(11) {
 *       ... the sequence ...
 *     }
 *
 * The fixture's event id must name this case, so a fixture built for another
 * case is refused rather than silently running this one.
 */
#define PYC_CASE(number)                                                       \
  static void pyc_case_body(void);                                            \
  void pym_replay(const struct pym_header *input, struct pym_header *out,     \
                  void *scratch) {                                            \
    (void)scratch;                                                            \
    unsigned mode = out->mode;                                                \
    memset(out, 0, sizeof *out);                                              \
    out->mode = mode;                                                         \
    out->magic = PYM_MAGIC;                                                   \
    out->count = 1;                                                           \
    const struct pym_event *e = (const void *)(input + 1);                    \
    PYC_INSTALL_HANDLER(number);                                              \
    CHECK(input->magic == PYM_MAGIC && input->count == 1 &&                   \
              e->id == (number),                                              \
          700);                                                               \
    pyc_case_body();                                                          \
    PYC_COMPLETED(number);                                                    \
    /* Only the spatial arm is expected to arrive here. */                    \
    out->completed = 1;                                                       \
    pym_backing_stats(out);                                                   \
  }                                                                           \
  static void pyc_case_body(void)

#endif /* PYC_CORPUS_H */
