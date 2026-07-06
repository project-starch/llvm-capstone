#include "sqlite_borrow_revoke_probe.h"

/* Host binding payload (borrower). Round 1: read the borrowed column buffer while
 * the row is current and cache the pointer in a .bss slot (the diesel bug).
 * Round 2 (after the engine's step/revoke): re-read the CACHED pointer -- the
 * use-after-free. The cached capability reloads untagged after the revoke, so the
 * round-2 read faults; the monitor terminates the domain and the engine sees the
 * fault sentinel instead of a stale column value. */

#define SBI_EXT_CAPSTONE 0x12345678
#define SBI_EXT_CAPSTONE_DOM_RETURN 0x5
#define SBI_EXT_CAPSTONE_REGION_QUERY 0x6
#define SBI_EXT_CAPSTONE_REGION_COUNT 0x8

#define CAPSTONE_REGION_FIELD_BASE 0x0

typedef unsigned long uintptr_t;
typedef unsigned long region_id_t;

struct sbiret {
  long error;
  long value;
};

static char stack[4096];
/* The cached sqlite3_column_text() pointer, held across the engine's step. */
static unsigned long *cached_column;

static struct sbiret sbi_ecall(int ext, int fid, unsigned long arg0,
                               unsigned long arg1, unsigned long arg2,
                               unsigned long arg3, unsigned long arg4,
                               unsigned long arg5) {
  struct sbiret ret;
  register uintptr_t a0 asm("a0") = (uintptr_t)(arg0);
  register uintptr_t a1 asm("a1") = (uintptr_t)(arg1);
  register uintptr_t a2 asm("a2") = (uintptr_t)(arg2);
  register uintptr_t a3 asm("a3") = (uintptr_t)(arg3);
  register uintptr_t a4 asm("a4") = (uintptr_t)(arg4);
  register uintptr_t a5 asm("a5") = (uintptr_t)(arg5);
  register uintptr_t a6 asm("a6") = (uintptr_t)(fid);
  register uintptr_t a7 asm("a7") = (uintptr_t)(ext);
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)
               : "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7)
               : "memory");
  ret.error = a0;
  ret.value = a1;
  return ret;
}

static unsigned long *borrowed_region_base(void) {
  struct sbiret count = sbi_ecall(SBI_EXT_CAPSTONE,
                                  SBI_EXT_CAPSTONE_REGION_COUNT, 0, 0, 0, 0, 0,
                                  0);
  region_id_t shared_region = (region_id_t)count.value - 1;
  struct sbiret base = sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_QUERY,
                                 shared_region, CAPSTONE_REGION_FIELD_BASE, 0, 0,
                                 0, 0);
  return (unsigned long *)base.value;
}

static void dom_return(unsigned long value) {
  (void)sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
                  0, 0);
}

static void start_impl(void) {
  /* Round 1: read the column while it is current, and cache the pointer. */
  cached_column = borrowed_region_base();
  unsigned long v1 = cached_column[0];
  dom_return(v1);

  /* Round 2 (after step/revoke): re-read the cached pointer -> use-after-free.
   * The cached capability reloads untagged; this read faults. */
  unsigned long v2 = cached_column[0];
  dom_return(v2);

  while (1) {
  }
}

__attribute__((naked)) void _start(void) {
  __asm__ volatile("mv sp, %0\n"
                   "j start_impl\n"
                   :
                   : "r"(stack + sizeof(stack))
                   : "memory");
}
