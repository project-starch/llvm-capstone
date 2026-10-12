/* The entry point, the root context and the labelled probes. One program per
 * case, as the contract says: a capability fault ends the process, so a case
 * that provokes one cannot also report results beside it.
 *
 * Every arm is built from this file. What varies is the manager variant the
 * image links (PG_SUBLET in the port) and, on CheriBSD, the guest's own libc
 * revocation, which the runner sets in the environment.
 */
#include "corpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MemoryContext pg_root;
unsigned char *volatile pg_held;

_Noreturn void pg_give_up(unsigned long code) {
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

__attribute__((noinline)) unsigned pg_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl pg_defect_probe\npg_defect_probe:\nlbu %0, 0(%1)"
                   : "=r"(value)
#ifdef __CHERI_PURE_CAPABILITY__
                   : "C"(p)
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
  printf("PG_DEFECT case=%u ready\n", pg_case_number);
  fflush(stdout);
}

MemoryContext pg_aset_child(MemoryContext parent, const char *name) {
  return AllocSetContextCreateInternal(parent, name, ALLOCSET_SMALL_MINSIZE,
                                       ALLOCSET_SMALL_INITSIZE,
                                       ALLOCSET_SMALL_MAXSIZE);
}

int main(int argc, char **argv) {
  /* Strict input rejection also supplies the runner's negative control. Whether
   * the memory contexts are protected is decided by which manager variant the
   * image links, so mode 0 is the only mode: an arm's control is a different
   * binary, not the same binary under another argument. */
  if (argc < 2 || argc > 3 || strcmp(argv[1], "0"))
    return 75;
  if (argc == 3 && (unsigned)atoi(argv[2]) != pg_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %u, run asked for %s\n",
            pg_case_number, argv[2]);
    return 75;
  }
  pg_root = AllocSetContextCreateInternal(NULL, "root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = pg_root;
  (void)pg_write_probe; /* the label must exist even where no case writes */
  pg_case_run();
  printf("PG_DEFECT case=%u mode=0 completed\n", pg_case_number);
  return 0;
}
