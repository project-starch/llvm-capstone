/* The entry point, the root context and the labelled probes. One program per
 * case, as the contract says: a capability fault ends the domain, so a case
 * that provokes one cannot also report results beside it.
 *
 * Both arms of both targets are built from this file. The Capstone domain
 * build keeps the four shared regions the harness has always passed; what
 * changes is that the selection region no longer DISPATCHES to a case, it only
 * checks that the image the harness loaded is the case the run asked for.
 */
#include "corpus.h"

#ifdef PG_CORPUS_HOSTED
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef PG_POISONCAP
#include "poisoncap.h"
#endif
#else
#include "domain-runtime.h"
#ifdef PG_DEFECTS_SUBLET
#include "pg_subpool.h"
#else
void pg_level0_init(void *, size_t);
#endif
static unsigned long arena_type;
static const volatile unsigned *selection;
#endif

MemoryContext pg_root;
unsigned char *volatile pg_held;

_Noreturn void pg_give_up(unsigned long code) {
#ifdef PG_CORPUS_HOSTED
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
#else
  give_up(code);
#endif
}

_Noreturn void pg_subpool_refuse(const char *why) {
#ifdef PG_CORPUS_HOSTED
  fprintf(stderr, "PG_DEFECT refused: %s\n", why);
#else
  fail(why);
#endif
  pg_give_up(0xbad90001);
}

__attribute__((noinline)) unsigned pg_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl pg_defect_probe\npg_defect_probe:\nlbu %0, 0(%1)"
                   : "=r"(value)
#ifdef __CHERI_PURE_CAPABILITY__
                   : "C"(p)      /* purecap, with or without PoisonCap */
#else
                   : "r"(p)
#endif
                   : "memory");
  return value;
}

__attribute__((noinline)) void pg_write_probe(volatile unsigned char *p) {
  __asm__ volatile(".globl pg_defect_write\npg_defect_write:\nsb %0, 0(%1)" ::"r"(
                       93UL),
#ifdef __CHERI_PURE_CAPABILITY__
                   "C"(p)
#else
                   "r"(p)
#endif
                   : "memory");
}

void pg_mark(void) {
#ifdef PG_CORPUS_HOSTED
  printf("PG_DEFECT case=%u ready\n", pg_case_number);
  fflush(stdout);
#else
  extern void pg_defect_probe(void), pg_defect_write(void);
  unsigned long code = 0xcf18000000000000UL | pg_case_number;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(pg_defect_probe), "r"(pg_defect_write)
                   : "memory");
#endif
}

MemoryContext pg_aset_child(MemoryContext parent, const char *name) {
  return AllocSetContextCreateInternal(parent, name, ALLOCSET_SMALL_MINSIZE,
                                       ALLOCSET_SMALL_INITSIZE,
                                       ALLOCSET_SMALL_MAXSIZE);
}

static void start(void) {
  pg_root = AllocSetContextCreateInternal(NULL, "root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = pg_root;
  (void)pg_write_probe; /* the label must exist even where no case writes */
}

#ifdef PG_CORPUS_HOSTED
int main(int argc, char **argv) {
  /* Strict input rejection also supplies the runner's negative control.
   * Without PoisonCap there is only the unprotected shape, so mode 0 is the
   * only mode this build accepts -- what varies for that arm is the guest's
   * own libc revocation, which the runner sets in the environment. */
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "0") && strcmp(argv[1], "1")))
    return 75;
  if (argc == 3 && (unsigned)atoi(argv[2]) != pg_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %u, run asked for %s\n",
            pg_case_number, argv[2]);
    return 75;
  }
  unsigned mode = (unsigned)(argv[1][0] - '0');
#ifdef PG_POISONCAP
  pg_poisoncap_init(mode);
#else
  if (mode)
    return 75; /* no protected mode exists in this build */
#endif
  start();
  pg_case_run();
  printf("PG_DEFECT case=%u mode=%u completed\n", pg_case_number, mode);
#ifdef PG_POISONCAP
  pg_poisoncap_report();
#endif
  return 0;
}
#else
void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    switch (shares++) {
    case 0:
      meta = (void *)res;
      break;
    case 1:
      payload = (void *)res;
      break;
    case 2:
#ifdef PG_DEFECTS_SUBLET
      arena_type = pg_subpool_arena(res, PG_REPLAY_ARENA_SIZE);
#else
      pg_level0_init(res, PG_REPLAY_ARENA_SIZE);
#endif
      break;
    case 3:
      selection = (void *)res;
      break;
    }
    return;
  }
  domain_result = res;
  CHECK(shares >= 4 && meta && payload && selection && arena_type == 0,
        0xbad90002);
  meta->length = 0;
  pg_domain_payload((char *)payload, (unsigned long *)&meta->length,
                    PG_REPLAY_PAYLOAD_SIZE);
  /* Not a dispatch: the image IS one case, and a harness asking for another
   * one is a control failure, not a run. */
  CHECK(selection[0] == pg_case_number, 0xbad90003);
  start();
  pg_case_run();
  /* Only the spatial arm is expected to arrive here. */
  pg_domain_text("__CAPSTONE_PG_DEFECT_COMPLETED__\n");
}
#endif
