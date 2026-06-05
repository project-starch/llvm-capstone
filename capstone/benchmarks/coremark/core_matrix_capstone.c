#include "coremark.h"

/* Convert a LINEAR capability to NONLIN so it can be used as a base in
 * multiple cincoffset operations without being consumed.  Must only be
 * applied to a register that holds a tagged LINEAR cap. */
#define CAPSTONE_DELIN(rd) \
    __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static inline void *capstone_align4_ptr(void *ptr) {
  ee_ptr_int misalign = ((ee_ptr_int)ptr) & (ee_ptr_int)3u;
  ee_ptr_int adjust = ((ee_ptr_int)4u - misalign) & (ee_ptr_int)3u;
  return (void *)((char *)ptr + adjust);
}

static inline void capstone_fill_matrix_data(MATDAT *A,
                                             MATDAT *B,
                                             ee_u32 Count,
                                             ee_s32 seed) {
  ee_s32 order = 1;
  ee_s32 running_seed = seed;
  ee_u32 idx;

  if (running_seed == 0)
    running_seed = 1;

  for (idx = 0; idx < Count; ++idx) {
    ee_s32 val;
    running_seed = (order * running_seed) % 65536;
    val = running_seed + order;
    val = val & 0x0ffff;         /* matrix_clip(val, 0): clip before use as B */
    B[idx] = (MATDAT)val;
    /* Re-sign-extend to 16-bit before adding order, matching upstream where
     * val is MATDAT (ee_s16): upstream assigns to MATDAT first which truncates
     * to signed 16-bit, then does val+order on the signed value. */
    val = (ee_s32)(MATDAT)val;
    val = val + (ee_s32)order;   /* add order to sign-extended B value */
    A[idx] = (MATDAT)(val & 0x00ff);   /* matrix_clip(val, 1) */
    ++order;
  }
}

ee_u32 core_init_matrix(ee_u32 blksize, void *memblk, ee_s32 seed, mat_params *p) {
  ee_u32 N = 0;
  ee_u32 used = 0;
  ee_u32 Count;
  MATDAT *A;
  MATDAT *B;
  MATRES *C;

  while (used < blksize) {
    ++N;
    used = N * N * 2u * 4u;
  }

  if (N == 0)
    return 0;
  --N;
  Count = N * N;

  A = (MATDAT *)capstone_align4_ptr(memblk);
  /* A is a gp-derived LINEAR cap.  Convert to NONLIN before computing B so
   * that "B = A + Count" does not consume A via cincoffset(rd≠rs1, LINEAR).
   * B and C are derived from A (NONLIN) so they inherit NONLIN and need no
   * separate delin. */
  CAPSTONE_DELIN(A);
  B = A + Count;
  C = (MATRES *)capstone_align4_ptr(B + Count);

  p->A = A;
  p->B = B;
  p->C = C;
  p->N = (int)N;

  capstone_fill_matrix_data(A, B, Count, seed);

  return N;
}



