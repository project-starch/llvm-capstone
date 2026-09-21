/* The positive control for the stock arm: does this guest's libc revocation
 * fire at all, at the very shape the corpus probes?
 *
 * malloc a block, free it, ask the revoker to sweep, malloc the same size, then
 * read through the OLD pointer at a labelled clbu. With revocation on the old
 * capability has lost its tag and the load faults HERE -- SIGPROT,
 * PROT_CHERI_TAG, pc == mc_defect_read; with it off the load returns whatever
 * is there and the program completes. The label is the corpus's, so the same
 * supervisor resolves it, and the same oracle grades this and the case: a
 * stock arm that completes means "the mechanism is active and did not fire",
 * not "the mechanism is absent" -- this program is the difference. */
#include <cheri/cheric.h>
#include <malloc_np.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile unsigned char *held;

__attribute__((noinline)) static unsigned read_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl mc_defect_read\nmc_defect_read:\nclbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "C"(p)
                   : "memory");
  return value;
}

int main(void) {
  unsigned char *a = malloc(64);
  if (!a)
    return 2;
  memset(a, 0x11, 64);
  held = a;
  free(a);
  unsigned on = (unsigned)malloc_revoke_enabled();
  if (on) /* the sweep, synchronously, so the outcome is not a race; the
             header deprecates malloc_revoke() in favour of this name */
    malloc_revoke_quarantine_force_flush();
  unsigned char *b = malloc(64);
  if (!b)
    return 2;
  memset(b, 0x22, 64);
  printf("REVOCATION_CONTROL revocation=%u tag_after_sweep=%u reissued=%d\n", on,
         (unsigned)cheri_gettag((void *)held),
         cheri_getaddress((void *)a) == cheri_getaddress(b));
  fflush(stdout);
  (void)read_probe(held);
  puts("REVOCATION_CONTROL stale read completed");
  return 0;
}
