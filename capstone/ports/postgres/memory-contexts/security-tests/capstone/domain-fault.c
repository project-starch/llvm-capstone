/* Deliberate runtime faults, independent of the allocator under test. */
#include "domain-runtime.h"

static volatile unsigned long *observed;

__attribute__((noinline)) static void fault(void) {
  extern void pg_fault_site(void);
  unsigned char bytes[16] = {0};
  unsigned long ignored;
  const void *p = bytes + sizeof bytes;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0" ::"r"(pg_fault_site));
#if PG_FAULT_KIND == 4
  /* A legacy RWX vector is not a local-recovery opt-in. */
  __asm__ volatile(".insn i 0x5b, 0x7, t0, 0(%0)" ::"r"(pg_fault_site)
                   : "t0", "memory");
#elif PG_FAULT_KIND == 5
  /* The trampoline must disable its vector before touching broken state. */
  __asm__ volatile(".insn i 0x5b, 0x7, t0, 4(zero)" ::: "t0", "memory");
#endif
#if PG_FAULT_KIND == 0
  __asm__ volatile(".globl pg_fault_site\npg_fault_site:\nlbu %0, 0(%1)"
                   : "=r"(ignored)
                   : "r"(p)
                   : "memory");
#elif PG_FAULT_KIND == 2
  /* Recovery must not depend on either application SP or GP. Never returns. */
  __asm__ volatile("li sp, 0\nli gp, 0\n"
                   ".globl pg_fault_site\npg_fault_site:\nld t0, 0(sp)" ::
                       : "t0", "memory");
#else
  __asm__ volatile(".globl pg_fault_site\npg_fault_site:\nld %0, 0(zero)"
                   : "=r"(ignored)::"memory");
#endif
}

void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (!observed)
      observed = (void *)res;
#if PG_FAULT_KIND != 3
    return;
#endif
  }
  observed[0]++;
  fault();
  observed[1]++; /* Must remain zero, including after attempts to re-enter. */
  if (func == 0)
    *res = 0;
}
