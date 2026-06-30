#include "revoke_matrix_probe.h"

/* Borrower payload. Receives a revocable borrow; holds the delegated cap across
 * the lender's revoke per REVOKE_MATRIX_CASE; dereferences after revoke. */

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
/* CASE 2: the live pointer slot. CASE 3: a separate capability slot the cap is
 * stc'd into and ldc'd back out of. Both are .bss (in domain memory). */
static unsigned long *slot;
static unsigned long *capslot;

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
                                 shared_region, CAPSTONE_REGION_FIELD_BASE, 0,
                                 0, 0, 0);
  return (unsigned long *)base.value;
}

static void dom_return(unsigned long value) {
  (void)sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
                  0, 0);
}

static void start_impl(void) {
  unsigned long *b = borrowed_region_base();

#if REVOKE_MATRIX_CASE == 3
  /* Round 1: write stage 1, then stc the cap into a separate slot. */
  b[0] = REVOKE_MATRIX_SENTINEL_STAGE1;
  capslot = b;
  dom_return(REVOKE_MATRIX_RET_ROUND1);
  /* Round 2 (after revoke): ldc the cap back and dereference it. */
  {
    unsigned long *r = capslot;
    r[0] = REVOKE_MATRIX_SENTINEL_STAGE2;
  }
  dom_return(REVOKE_MATRIX_RET_ROUND2);
#else /* CASE 2: memory-stored pointer slot */
  slot = b;
  slot[0] = REVOKE_MATRIX_SENTINEL_STAGE1;
  dom_return(REVOKE_MATRIX_RET_ROUND1);
  /* Round 2 (after revoke): reload slot and dereference. */
  slot[0] = REVOKE_MATRIX_SENTINEL_STAGE2;
  dom_return(REVOKE_MATRIX_RET_ROUND2);
#endif

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
