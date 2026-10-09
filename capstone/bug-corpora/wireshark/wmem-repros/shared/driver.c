/* The entry point, the scopes and the labelled probes. One program per case,
 * as the contract says: a capability fault ends the domain, so a case that
 * provokes one cannot also report results beside it.
 *
 * The Capstone domain build reuses the port's entry protocol unchanged (four
 * shared regions, the arm chosen by the loader at run time); the selection it
 * receives is CHECKED against the case this image carries, never dispatched.
 */
#include "corpus.h"

#ifdef WM_CORPUS_HOSTED
#include <stdio.h>
#include <stdlib.h>
#ifdef WM_POISONCAP
#include "poisoncap.h"
#endif
#endif

wmem_allocator_t *wm_packet;
unsigned char *volatile wm_held;

_Noreturn void wm_give_up(unsigned long code) {
#ifdef WM_CORPUS_HOSTED
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
#else
  wm_fail((unsigned)code);
#endif
}

void wm_next_packet(void) { wm_packet_pool_reset(wm_packet); }

#if defined(WM_CORPUS_HOSTED) && !defined(__riscv)
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
#ifdef WM_CORPUS_HOSTED
  printf("WM_DEFECT case=%u ready\n", wm_case_number);
  fflush(stdout);
#ifdef WM_POISONCAP
  /* The protected arm dies at the access that follows; the adapter's counters
   * up to this point are the evidence of which hook fired, so print them now. */
  wm_poisoncap_report();
#endif
#else
  extern void wm_defect_probe(void), wm_defect_write(void), wm_widen_probe(void);
  unsigned long code = 0xcf16000000000000UL | wm_case_number;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %3, x0\n" ::"r"(code),
                   "r"(wm_defect_probe), "r"(wm_defect_write), "r"(wm_widen_probe)
                   : "memory");
#endif
}

static void start(void) {
  wm_scopes_init();
  wm_enter_file_scope();
  wm_packet = wm_packet_pool_acquire();
  (void)wm_write_probe; /* the label must exist even where no case writes */
}

#ifdef WM_BORROW_LINEAR
/* The heap's lend API: one linear block, the heap keeping the senior handle so
 * its own free still reclaims everything derived from it. Declared rather than
 * included -- it is the allocator's internal interface, not a libc header. */
#include <capstone/capability.h>
unsigned long __capstone_sublet_malloc_linear(size_t, capstone_cap_slot *);
#endif
#ifdef WM_CORPUS_HOSTED
_Noreturn void wm_fail(unsigned code) { wm_give_up(code); }
int main(int argc, char **argv) {
  /* Strict input rejection is the runner's control. Which modes exist depends
   * on what is under the allocator: with the native region layer only the
   * unprotected shape does, so mode 0 is the only mode accepted; with
   * WM_BORROW_LINEAR the Sublet region and chunk layers are underneath and
   * both run. */
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 2 || argc > 3 || (strcmp(argv[1], "0") && strcmp(argv[1], "1")))
    return 75;
  if (argc == 3 && (unsigned)atoi(argv[2]) != wm_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %u, run asked for %s\n",
            wm_case_number, argv[2]);
    return 75;
  }
  unsigned mode = (unsigned)(argv[1][0] - '0');
#if defined(WM_BORROW_LINEAR)
  /* The payload is LENT by the system allocator as ONE LINEAR capability,
   * which is what lets the region layer carve inside it and revoke a carve
   * when a scope ends. wm_regions_init requires it linear, exactly
   * WM_PAYLOAD_BYTES long and 16-byte aligned, and calls wm_fail(201)
   * otherwise, so a region that is merely nearly right stops the case instead
   * of quietly measuring something else. */
  {
    capstone_cap_slot lent;
    if (!__capstone_sublet_malloc_linear(WM_PAYLOAD_BYTES, &lent)) {
      fprintf(stderr, "CONTROL-FAILED the heap refused a linear block of %lu "
                      "bytes\n", (unsigned long)WM_PAYLOAD_BYTES);
      return 75;
    }
    /* capstone_cap_load moves it out of the slot into a register, which is how
     * the region layer takes it: in a domain it arrives that way. */
    wm_init_backing(NULL, capstone_cap_load(&lent), mode);
  }
#elif defined(WM_POISONCAP)
  wm_init_backing(NULL, NULL, mode);
#else
  if (mode)
    return 75; /* no protected mode exists in this build */
  void *payload = aligned_alloc(16, WM_PAYLOAD_BYTES);
  if (!payload)
    return 75;
  wm_init_backing(NULL, payload, 0);
#endif
  start();
  wm_case_run();
  printf("WM_DEFECT case=%u mode=%u completed\n", wm_case_number, mode);
#ifdef WM_POISONCAP
  wm_poisoncap_report();
#endif
  return 0;
}
#else
void wm_replay(const struct wm_header *in, struct wm_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = WM_MAGIC;
  out->mode = mode;
  out->count = 1;
  const struct wm_event *e = (const void *)(in + 1);
  CHECK(in->magic == WM_MAGIC && in->count == 1, 0xbad90001);
  /* Not a dispatch: the image IS one case, and a harness asking for another
   * one is a control failure, not a run. */
  CHECK(e->arg == wm_case_number, 0xbad90003);
  start();
  wm_case_run();
  /* Only the spatial arm is expected to arrive here. */
  out->completed = 1;
}
#endif
