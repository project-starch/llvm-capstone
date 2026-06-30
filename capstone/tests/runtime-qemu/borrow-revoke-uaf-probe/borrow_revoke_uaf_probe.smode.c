#include "borrow_revoke_uaf_probe.h"

/* Borrower payload (runs inside the domain, S-mode). It receives a borrowed
 * region capability from the lender (the .user controller). In round 1 it
 * caches the delegated pointer and writes the stage-1 sentinel. Between the
 * two rounds the lender calls revoke_region(). In round 2 the borrower
 * dereferences the *cached* pointer: this is the use-after-revoke. The
 * expected behaviour is a deterministic capability fault; if instead the
 * write lands (stage-2 sentinel becomes visible to the lender), that records
 * a no-trap gap. */

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

/* Cached across the two call_dom rounds: start_impl runs once and resumes
 * after each dom_return in the same frame, so this pointer survives the
 * yield (exactly the stale-handle shape the SQLite contract warns about). */
static unsigned long *borrowed;
static char stack[4096];

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
  /* The borrowed region is the last one shared into the domain. */
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
  struct sbiret ignored =
      sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
                0, 0);
  (void)ignored;
}

static void start_impl(void) {
  /* Round 1: borrow is live. Cache the delegated pointer and write stage 1. */
  borrowed = borrowed_region_base();
  borrowed[0] = BORROW_REVOKE_UAF_SENTINEL_STAGE1;
  dom_return(BORROW_REVOKE_UAF_RET_ROUND1);

  /* Round 2: the lender has revoked the region between the two calls.
   * Dereferencing the cached pointer is the use-after-revoke. If the cap was
   * invalidated by revoke, this store faults here (deterministic capability
   * fault). If it does NOT fault, the stage-2 sentinel becomes visible to the
   * lender, which the controller reports as a no-trap gap. */
  borrowed[0] = BORROW_REVOKE_UAF_SENTINEL_STAGE2;
  dom_return(BORROW_REVOKE_UAF_RET_ROUND2);

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
