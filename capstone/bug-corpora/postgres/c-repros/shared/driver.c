/* Entry point and probes for client-repros. One program per case, as the
 * contract says: a capability fault ends the process, so a case that provokes
 * one cannot also report results beside it.
 */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#if defined(__FreeBSD__)
#include <sys/ucontext.h>
#endif

_Noreturn void pgclient_give_up(unsigned long code) {
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

void *pgclient_malloc(size_t n) {
  void *p = malloc(n);
  if (!p) pgclient_give_up(1);
  return p;
}

void pgclient_free(void *p) { free(p); }
void pgclient_memcpy(void *d, const void *s, size_t n) { memcpy(d, s, n); }

void pgclient_mark(void) {
  printf("PG_DEFECT case=%u mark\n", pgclient_case_number);
  fflush(stdout);
}

/* ---- fault attribution -------------------------------------------------
 * The case says which function the fault belongs to; the handler says where
 * it actually happened. Printing the two as offsets is what turns "the arm
 * faulted" into "the arm faulted ON THIS DEFECT". See corpus.h for why this
 * is done here rather than with a label on the faulting instruction. */
static const void *expect_fn;
static const char *expect_name = "(unset)";

void pgclient_expect_fault_in(const void *fn, const char *name) {
  expect_fn = fn;
  expect_name = name;
  printf("PG_DEFECT case=%u expect_fault_in=%s@%p\n",
         pgclient_case_number, name, fn);
  fflush(stdout);
}

#if defined(__CAPSTONE__)
/* A Capstone domain has no signal to catch. A capability violation ends the
 * domain, and the host prints the line the runner scores:
 *
 *   capstone-exec: domain fault cause=7 pc=0x... address=0x... image=...
 *
 * Attribution still works, and without a handler: pgclient_expect_fault_in
 * above prints expect_fault_in=<name>@<address> before the defect runs, so
 * the host's pc is compared against an address the run itself published
 * rather than against anything hardcoded. That is the same check the other
 * two arms make from inside the process; here it is made from outside it.
 *
 * Installing a handler anyway would be worse than useless: it would not run,
 * and its presence would suggest the arm reports through it. */
static void install_fault_handler(void) {}
#else
static void fault_handler(int sig, siginfo_t *si, void *uc) {
  uintptr_t pc = 0, ra = 0;
#if defined(__FreeBSD__) && defined(__riscv)
  const ucontext_t *u = (const ucontext_t *) uc;
# if defined(__CHERI_PURE_CAPABILITY__)
  pc = (uintptr_t) u->uc_mcontext.mc_capregs.cp_sepcc;
  ra = (uintptr_t) u->uc_mcontext.mc_capregs.cp_cra;
# else
  pc = (uintptr_t) u->uc_mcontext.mc_gpregs.gp_sepc;
  ra = (uintptr_t) u->uc_mcontext.mc_gpregs.gp_ra;
# endif
#else
  (void) uc;   /* host build: ASan reports first and carries its own trace */
#endif
  char b[320];
  uintptr_t fn = (uintptr_t) expect_fn;
  /* Signed offsets, printed whether or not they are small: a large one is
   * itself the finding -- it says the fault was not where the case claims. */
  long dpc = fn ? (long) (pc - fn) : 0;
  long dra = fn ? (long) (ra - fn) : 0;
  int n = snprintf(b, sizeof b,
      "\nPG_FAULT case=%u signal=%d si_code=%d addr=%p pc=%p ra=%p"
      " expect=%s@%p pc-expect=%+ld ra-expect=%+ld\n",
      pgclient_case_number, sig, si->si_code, (void *) si->si_addr,
      (void *) pc, (void *) ra, expect_name, expect_fn, dpc, dra);
  if (n > 0) (void) write(2, b, (size_t) n);
  /* Die by the signal, not by _exit, so the runner still sees 128+sig and
   * nothing downstream has to know this handler exists. */
  signal(sig, SIG_DFL);
  raise(sig);
}

static void install_fault_handler(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = fault_handler;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  (void) sigaction(SIGSEGV, &sa, 0);
  (void) sigaction(SIGBUS, &sa, 0);
#ifdef SIGPROT
  (void) sigaction(SIGPROT, &sa, 0);   /* CheriBSD capability violation (34) */
#endif
}
#endif /* !__CAPSTONE__ */

void pgclient_note_signed(const char *label, long v) {
  printf("PG_NOTE %s=%ld\n", label, v);
  fflush(stdout);
}

void pgclient_note_overrun(const void *p, size_t legitimate, size_t asked) {
  printf("PG_NOTE overrun object=%p legitimate=%zu asked=%zu\n",
         p, legitimate, asked);
  fflush(stdout);
}

void pgclient_note_overread(const void *p, size_t legitimate, size_t asked) {
  printf("PG_NOTE overread object=%p legitimate=%zu asked=%zu\n",
         p, legitimate, asked);
  fflush(stdout);
}

char *pgclient_short_value(void) {
  /* What libpq hands ecpg for a zero-length text value: a 1-byte buffer
   * holding just the terminator. Allocated at exactly 1 byte so that the
   * `pval + 2` at data.c:532 is already past the end. */
  char *v = pgclient_malloc(1);
  v[0] = '\0';
  return v;
}

char *pgclient_oid_list(int n) {
  /* "1 2 3 ... n", the textual form parseOidArray is handed out of the
   * catalog. Sized exactly, so nothing downstream depends on slack. */
  size_t cap = (size_t) n * 12 + 1;
  char *s = pgclient_malloc(cap);
  size_t off = 0;
  for (int i = 0; i < n; i++)
    off += (size_t) snprintf(s + off, cap - off, i ? " %d" : "%d", i + 1);
  return s;
}

volatile Oid pgclient_oid_sink;
void pgclient_consume_oid(Oid v) { pgclient_oid_sink = v; }

int main(int argc, char **argv) {
  unsigned want;
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc != 2) {
    fprintf(stderr, "usage: %s <case-number>\n", argv[0]);
    return 75;
  }
  want = (unsigned) strtoul(argv[1], NULL, 10);
  if (want != pgclient_case_number) {
    fprintf(stderr, "CONTROL-FAILED this image is case %u, not %u\n",
            pgclient_case_number, want);
    return 75;
  }
  install_fault_handler();
  printf("case %u BEGIN\n", pgclient_case_number);
  pgclient_case_run();
  printf("case %u RETURNED\n", pgclient_case_number);
  return 0;
}
