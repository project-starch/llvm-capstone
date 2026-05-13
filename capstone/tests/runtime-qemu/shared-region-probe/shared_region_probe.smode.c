#include "shared_region_probe.h"

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

/* Minimal SBI ecall wrapper used by the custom .smode probe payload. */
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

static char stack[4096];

static unsigned long *shared_region_base(void) {
  /* For this probe the newly shared region is expected to be the last region. */
  struct sbiret count =
      sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_COUNT, 0, 0, 0, 0, 0,
                0);
  region_id_t region_n = (region_id_t)count.value;
  region_id_t shared_region = region_n - 1;
  struct sbiret base = sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_QUERY,
                                 shared_region, CAPSTONE_REGION_FIELD_BASE, 0,
                                 0, 0, 0);
  return (unsigned long *)base.value;
}

static void dom_return(unsigned long value) {
  struct sbiret ignored =
      sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
                0, 0);
  (void)ignored;
}

static void start_impl(void) {
  /* First call_dom() round: publish stage 1, then return to the helper. */
  unsigned long *region = shared_region_base();
  region[0] = SHARED_REGION_PROBE_SENTINEL_STAGE1;

  dom_return(0x101);

  /* Second call_dom() round: publish stage 2, then return again. */
  region[0] = SHARED_REGION_PROBE_SENTINEL_STAGE2;
  dom_return(0x202);

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



