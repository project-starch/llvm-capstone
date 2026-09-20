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
 * TWO TARGETS, SAME SEQUENCES. With PYMALLOC_POISONCAP a case builds as an
 * ordinary CheriBSD purecap program against the port's PoisonCap adapter, and
 * keeps the same two modes: mode 0 is request-bounded spatial authority with
 * no per-object invalidation and must COMPLETE, mode 1 adds lifetime
 * invalidation and must FAULT. Only three things differ, and none of them is a
 * case: the probes become capability-base accesses (clbu/csb) so the label
 * sits on an instruction the trap PC can be compared against, the markers
 * become printed lines instead of Capstone marker instructions, and a SIGPROT
 * handler turns the fault into one machine-readable line before re-raising.
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
#ifdef PYMALLOC_POISONCAP
/* The hosted CheriBSD/PoisonCap build only. The Capstone domain build is
 * freestanding and has none of these. */
#include <cheri/cheric.h>
#include <signal.h>
#include <stdio.h>
#include <ucontext.h>
#include <unistd.h>
#endif

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
#ifdef PYMALLOC_POISONCAP
  /* Capability-base byte load. The label sits ON the faulting instruction, so
   * the SIGPROT handler can require the fault here rather than anywhere in the
   * program; a "C" operand keeps the stale capability itself as the base. */
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

/* No case writes, so nothing calls this. It is kept alive explicitly because
 * the label must exist: on the Capstone target mark() publishes the address of
 * pyc_defect_write, and with one program per case there is no longer a
 * `(void)write_probe;` at the end of a shared defect() to hold the reference.
 * Without `used`, --gc-sections drops the function and the link fails. */
__attribute__((used, noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long value = 93;
#ifdef PYMALLOC_POISONCAP
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

#ifdef PYMALLOC_POISONCAP
/* CheriBSD reports an access through a revoked alias as SIGPROT with
 * si_code PROT_CHERI_TAG. The handler exists so that the runner can require
 * the fault to be THAT fault at THAT instruction: a bounds fault, a permission
 * fault, an ordinary SIGSEGV or a tag fault at some other address is not this
 * corpus reproducing, and would otherwise be indistinguishable from it through
 * the process exit status alone, which is 162 for every SIGPROT.
 *
 * It prints one line and then re-raises with the default disposition, so the
 * process still ends the ordinary CheriBSD way instead of a fault being turned
 * into a normal exit. */
extern void pyc_defect_read(void);

static volatile sig_atomic_t fault_case;

/* Signal-handler formatting: a fixed local buffer and one write(). Every
 * append is bounded by the buffer size, so an over-long line is TRUNCATED
 * rather than overflowing -- and a truncated line fails the runner's
 * complete-line match, which is the safe direction. */
static size_t put_text(char *buffer, size_t size, size_t at, const char *text) {
  while (*text && at < size)
    buffer[at++] = *text++;
  return at;
}

static size_t put_number(char *buffer, size_t size, size_t at,
                         unsigned long value, unsigned base) {
  char digits[24];
  size_t count = 0;
  do {
    digits[count++] = "0123456789abcdef"[value % base];
    value /= base;
  } while (value && count < sizeof digits);
  while (count && at < size)
    buffer[at++] = digits[--count];
  return at;
}

static void report_fault(int sig, siginfo_t *info, void *context) {
  const ucontext_t *state = context;
  unsigned long pc = (unsigned long)cheri_getaddress(
      (void *)state->uc_mcontext.mc_capregs.cp_sepcc);
  unsigned long expected =
      (unsigned long)cheri_getaddress((void *)pyc_defect_read);
  int exact =
      sig == SIGPROT && info->si_code == PROT_CHERI_TAG && pc == expected;
  char line[192];
  size_t at = 0;
  at = put_text(line, sizeof line, at, "PYC_DEFECT_FAULT case=");
  at = put_number(line, sizeof line, at, (unsigned long)fault_case, 10);
  at = put_text(line, sizeof line, at, " signal=");
  at = put_number(line, sizeof line, at, (unsigned long)sig, 10);
  at = put_text(line, sizeof line, at, " code=");
  at = put_number(line, sizeof line, at, (unsigned long)info->si_code, 10);
  at = put_text(line, sizeof line, at, " pc=0x");
  at = put_number(line, sizeof line, at, pc, 16);
  at = put_text(line, sizeof line, at, " expected=0x");
  at = put_number(line, sizeof line, at, expected, 16);
  at = put_text(line, sizeof line, at, " exact=");
  at = put_text(line, sizeof line, at, exact ? "1" : "0");
  at = put_text(line, sizeof line, at, "\n");
  (void)write(STDOUT_FILENO, line, at);
  /* Restore the platform outcome: default disposition, unblocked, re-raised. */
  struct sigaction restore;
  sigset_t unblock;
  memset(&restore, 0, sizeof restore);
  restore.sa_handler = SIG_DFL;
  sigemptyset(&restore.sa_mask);
  sigaction(SIGPROT, &restore, NULL);
  sigemptyset(&unblock);
  sigaddset(&unblock, SIGPROT);
  sigprocmask(SIG_UNBLOCK, &unblock, NULL);
  raise(SIGPROT);
  _exit(70); /* raise() must not return; never let a fault exit cleanly. */
}

static void install_fault_handler(unsigned which) {
  struct sigaction action;
  memset(&action, 0, sizeof action);
  action.sa_sigaction = report_fault;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  fault_case = (sig_atomic_t)which;
  if (sigaction(SIGPROT, &action, NULL))
    pym_fail(750);
}
#endif

/* Publish the case and both probe addresses, so the host knows which
 * instruction a fault is allowed to be at. On CheriBSD the addresses travel
 * with the fault instead: the handler compares the trap PC against the
 * pyc_defect_read label itself, so the marker only has to say that the case
 * reached its critical access. */
static void mark(unsigned which) {
#ifdef PYMALLOC_POISONCAP
  printf("PYC_DEFECT case=%u ready\n", which);
  fflush(stdout);
#else
  extern void pyc_defect_read(void), pyc_defect_write(void);
  unsigned long code = 0xcf19000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(pyc_defect_read), "r"(pyc_defect_write)
                   : "memory");
#endif
}

/* An odict node: a link the loop follows, and a payload. Sized so the whole
 * node is one pymalloc block. */
struct node {
  struct node *next;
  unsigned char payload[16];
};


#ifdef PYMALLOC_POISONCAP
#define PYC_INSTALL_HANDLER(n) install_fault_handler(n)
#define PYC_COMPLETED(n)                                                       \
  do {                                                                        \
    printf("PYC_DEFECT case=%u completed\n", (unsigned)(n));                  \
    fflush(stdout);                                                           \
  } while (0)
#else
#define PYC_INSTALL_HANDLER(n) ((void)0)
#define PYC_COMPLETED(n) ((void)0)
#endif

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
