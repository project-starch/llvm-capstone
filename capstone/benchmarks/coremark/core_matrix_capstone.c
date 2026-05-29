#include "coremark.h"

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
    B[idx] = (MATDAT)(val & 0x0ffff);
    val += order;
    A[idx] = (MATDAT)(val & 0x00ff);
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
  B = A + Count;
  C = (MATRES *)capstone_align4_ptr(B + Count);

  p->A = A;
  p->B = B;
  p->C = C;
  p->N = (int)N;

  capstone_fill_matrix_data(A, B, Count, seed);

  return N;
}



