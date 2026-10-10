/* The entry point, the scopes and the labelled probes. One program per case,
 * as the contract says: a capability fault ends the process, so a case that
 * provokes one cannot also report results beside it.
 *
 * Every arm is built from this file. What varies is the wmem variant the
 * image links (WM_SUBLET in the port) and, on CheriBSD, the guest's own libc
 * revocation, which the runner sets in the environment.
 */
#include "corpus.h"

#include <stdio.h>
#include <stdlib.h>

wmem_allocator_t *wm_packet;
unsigned char *volatile wm_held;
int wm_fixed, wm_observe, wm_defect;

void wm_reoccupy(const void *stale, unsigned long bytes) {
  if (!wm_observe)
    return; /* every arm but the native fix differential: nothing added */
  unsigned char *next = wmem_alloc(wm_packet, bytes);
  CHECK(next, 0xbad90101);
  /* In the buggy sequence the freed storage is what the next dissection gets: asserted, so a
   * marker read through the stale pointer is evidence of aliasing rather than coincidence. In the
   * fixed sequence the stale pointer names storage that outlived the packet, which this
   * allocation cannot land on. */
  if (!wm_fixed)
    CHECK(next == stale, 0xbad90102);
  memset(next, WM_MARKER, bytes);
}

_Noreturn void wm_give_up(unsigned long code) {
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

_Noreturn void wm_fail(unsigned code) { wm_give_up(code); }

void wm_next_packet(void) { wm_packet_pool_reset(wm_packet); }

#if !defined(__riscv) && !defined(__CAPSTONE__)
/* An x86 host build has no fault oracle; the accesses stay, the labels do not. */
__attribute__((noinline)) unsigned wm_probe(const volatile unsigned char *p) {
  return *p;
}
__attribute__((noinline)) void wm_write_probe(volatile unsigned char *p) {
  *p = 93;
}
#else
/* Capstone and CheriBSD purecap: the labelled instruction the oracle names. */
__attribute__((noinline)) unsigned wm_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl wm_defect_probe\nwm_defect_probe:\nlbu %0, 0(%1)"
                   : "=r"(value)
#ifdef __CHERI_PURE_CAPABILITY__
                   : "C"(p)
#else
                   : "r"(p)
#endif
                   : "memory");
  return value;
}

__attribute__((noinline)) void wm_write_probe(volatile unsigned char *p) {
  __asm__ volatile(".globl wm_defect_write\nwm_defect_write:\nsb %0, 0(%1)" ::"r"(
                       93UL),
#ifdef __CHERI_PURE_CAPABILITY__
                   "C"(p)
#else
                   "r"(p)
#endif
                   : "memory");
}
#endif

void wm_mark(void) {
  printf("WM_DEFECT case=%u ready\n", wm_case_number);
  fflush(stdout);
}

int main(int argc, char **argv) {
  /* Strict input rejection is the runner's control. Whether wmem is
   * protected is decided by which variant the image links, so mode 0 is the
   * only mode: an arm's control is a different binary, not the same binary
   * under another argument. */
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 2 || argc > 4 || strcmp(argv[1], "0"))
    return 75;
  /* `program 0 N buggy|fixed`: the native fix differential. It observes
   * aliasing, which a protected arm exists to prevent. */
  int differential = argc == 4;
  if (differential) {
    if (strcmp(argv[3], "buggy") && strcmp(argv[3], "fixed"))
      return 75;
    wm_fixed = !strcmp(argv[3], "fixed");
    wm_observe = 1;
  }
  if (argc >= 3 && (unsigned)atoi(argv[2]) != wm_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %u, run asked for %s\n",
            wm_case_number, argv[2]);
    return 75;
  }
  wm_scopes_init();
  wm_enter_file_scope();
  wm_packet = wm_packet_pool_acquire();
  (void)wm_write_probe; /* the label must exist even where no case writes */
  wm_case_run();
  printf("WM_DEFECT case=%u mode=0 completed\n", wm_case_number);
  if (differential) {
    /* The defect reproduces when the buggy sequence reached another object's storage and the
     * fixed one did not. A case that cannot say which is INCONCLUSIVE, never a pass. */
    if (!wm_fixed && wm_defect)
      printf("VERDICT DEFECT-REPRODUCED the access reached storage outside its object\n");
    else if (wm_fixed && !wm_defect)
      printf("VERDICT FIXED the fix's sequence does not reach another object's storage\n");
    else
      printf("VERDICT INCONCLUSIVE\n");
    return wm_fixed ? !!wm_defect : !wm_defect;
  }
  return 0;
}
