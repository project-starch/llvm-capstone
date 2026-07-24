#ifndef COREMARK_MATRIX_KERNEL_H
#define COREMARK_MATRIX_KERNEL_H
/* Silicon-ladder rung 7: CoreMark 1.01 matrix benchmark (a *found* headline
 * benchmark), amalgamated into ONE translation unit.
 *
 * Source: EEMBC CoreMark v1.01 (commit cfa9ab37...), files core_matrix.c and
 * core_util.c (crc16 chain). The matrix compute (matrix_add_const /
 * matrix_mul_const / matrix_mul_vect / matrix_mul_matrix /
 * matrix_mul_matrix_bitextract / matrix_sum, CRC-folded by crc16) is reproduced
 * verbatim; only two things are adapted for the pure-cap silicon target and both
 * are #ifdef-guarded so the native `cc -O0` oracle compiles the identical math:
 *   1. align_mem is OMITTED. Upstream 4-byte-aligns A and C out of a malloc/stack
 *      block by building a pointer from an integer `(void*)(4 + ((int)x-1 & ~3))`,
 *      which would DROP the capability tag on PureCap. Here the block is a static
 *      16-aligned array and every A/B/C offset (0, 2*N*N, 4*N*N bytes) is already
 *      a multiple of 4, so align_mem is the identity function -- a bare pointer is
 *      both correct and keeps the tag.
 *   2. B/C are carved from a single gp-delivered block (B = A + N*N). On silicon
 *      the block cap is LINEAR, so `B = A + Count` (cincoffset rd!=rs1) would
 *      consume A; CAPSTONE_DELIN(A) makes it non-linear first. A no-op on native.
 *
 * WHY the matrix algorithm (and not list/state): the list and state CRCs are
 * pointer-size-dependent -- CoreMark packs pointers into list nodes, and at
 * sizeof(void*)=16 (capability) vs 8 the node count and traversal differ, so a
 * native oracle would NOT fold the same value (see design/coremark-purecap.md).
 * The matrix kernel operates purely on ee_s16/ee_s32 integer matrices, so its
 * crc16 result is pointer-size-INDEPENDENT: the domain and a native host compute
 * the identical checksum by construction (the crc32 rung's oracle contract).
 *
 * The benchmark is driven standalone with CoreMark's real validation-run matrix
 * parameters: block size 666 bytes (= TOTAL_DATA_SIZE 2000 / 3 algorithms ->
 * N=9), init seed 0 (-> 1), bench seed 0x66 (seed3). Deterministic; the crc16 is
 * itself the oracle. */

/* --- CoreMark integer types (barebones/core_portme.h) --- */
typedef signed short   ee_s16;
typedef unsigned short ee_u16;
typedef signed int     ee_s32;
typedef unsigned int   ee_u32;
typedef unsigned char  ee_u8;
typedef ee_s16 MATDAT;
typedef ee_s32 MATRES;
typedef struct MAT_PARAMS_S {
  int N;
  MATDAT *A;
  MATDAT *B;
  MATRES *C;
} mat_params;

/* --- CRC (core_util.c, verbatim) --- */
static ee_u16 crcu8(ee_u8 data, ee_u16 crc) {
  ee_u8 i = 0, x16 = 0, carry = 0;
  for (i = 0; i < 8; i++) {
    x16 = (ee_u8)((data & 1) ^ ((ee_u8)crc & 1));
    data >>= 1;
    if (x16 == 1) {
      crc ^= 0x4002;
      carry = 1;
    } else
      carry = 0;
    crc >>= 1;
    if (carry)
      crc |= 0x8000;
    else
      crc &= 0x7fff;
  }
  return crc;
}
static ee_u16 crcu16(ee_u16 newval, ee_u16 crc) {
  crc = crcu8((ee_u8)(newval), crc);
  crc = crcu8((ee_u8)((newval) >> 8), crc);
  return crc;
}
static ee_u16 crc16(ee_s16 newval, ee_u16 crc) {
  return crcu16((ee_u16)newval, crc);
}

/* --- matrix (core_matrix.c, verbatim compute) --- */
#define matrix_clip(x, y) ((y) ? (x)&0x0ff : (x)&0x0ffff)
#define matrix_big(x) (0xf000 | (x))
#define bit_extract(x, from, to) (((x) >> (from)) & (~(0xffffffff << (to))))

static ee_s16 matrix_sum(ee_u32 N, MATRES *C, MATDAT clipval) {
  MATRES tmp = 0, prev = 0, cur = 0;
  ee_s16 ret = 0;
  ee_u32 i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      cur = C[i * N + j];
      tmp += cur;
      if (tmp > clipval) {
        ret += 10;
        tmp = 0;
      } else {
        ret += (cur > prev) ? 1 : 0;
      }
      prev = cur;
    }
  }
  return ret;
}
static void matrix_mul_const(ee_u32 N, MATRES *C, MATDAT *A, MATDAT val) {
  ee_u32 i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      C[i * N + j] = (MATRES)A[i * N + j] * (MATRES)val;
}
static void matrix_add_const(ee_u32 N, MATDAT *A, MATDAT val) {
  ee_u32 i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i * N + j] += val;
}
static void matrix_mul_vect(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B) {
  ee_u32 i, j;
  for (i = 0; i < N; i++) {
    C[i] = 0;
    for (j = 0; j < N; j++)
      C[i] += (MATRES)A[i * N + j] * (MATRES)B[j];
  }
}
static void matrix_mul_matrix(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B) {
  ee_u32 i, j, k;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (k = 0; k < N; k++)
        C[i * N + j] += (MATRES)A[i * N + k] * (MATRES)B[k * N + j];
    }
}
static void matrix_mul_matrix_bitextract(ee_u32 N, MATRES *C, MATDAT *A,
                                         MATDAT *B) {
  ee_u32 i, j, k;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (k = 0; k < N; k++) {
        MATRES tmp = (MATRES)A[i * N + k] * (MATRES)B[k * N + j];
        C[i * N + j] += bit_extract(tmp, 2, 4) * bit_extract(tmp, 5, 7);
      }
    }
}
static ee_s16 matrix_test(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B, MATDAT val) {
  ee_u16 crc = 0;
  MATDAT clipval = matrix_big(val);
  matrix_add_const(N, A, val); /* make sure data changes */
  matrix_mul_const(N, C, A, val);
  crc = crc16(matrix_sum(N, C, clipval), crc);
  matrix_mul_vect(N, C, A, B);
  crc = crc16(matrix_sum(N, C, clipval), crc);
  matrix_mul_matrix(N, C, A, B);
  crc = crc16(matrix_sum(N, C, clipval), crc);
  matrix_mul_matrix_bitextract(N, C, A, B);
  crc = crc16(matrix_sum(N, C, clipval), crc);
  matrix_add_const(N, A, -val); /* return matrix to initial value */
  return crc;
}
static ee_u16 core_bench_matrix(mat_params *p, ee_s16 seed, ee_u16 crc) {
  ee_u32 N = p->N;
  MATRES *C = p->C;
  MATDAT *A = p->A;
  MATDAT *B = p->B;
  MATDAT val = (MATDAT)seed;
  crc = crc16(matrix_test(N, C, A, B, val), crc);
  return crc;
}

static ee_u32 core_init_matrix(ee_u32 blksize, void *memblk, ee_s32 seed,
                               mat_params *p) {
  ee_u32 N = 0;
  MATDAT *A;
  MATDAT *B;
  ee_s32 order = 1;
  MATDAT val;
  ee_u32 i = 0, j = 0;
  if (seed == 0)
    seed = 1;
  while (j < blksize) {
    i++;
    j = i * i * 2 * 4;
  }
  N = i - 1;
  A = (MATDAT *)memblk; /* block is 16-aligned; align_mem is identity (see top) */
#ifdef __capstone
  /* Make the gp-delivered block cap non-linear so B = A + N*N (cincoffset
     rd!=rs1) does not consume A. No-op on the native oracle. */
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(A));
#endif
  B = A + N * N;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      seed = ((order * seed) % 65536);
      val = (seed + order);
      val = matrix_clip(val, 0);
      B[i * N + j] = val;
      val = (val + order);
      val = matrix_clip(val, 1);
      A[i * N + j] = val;
      order++;
    }
  }
  p->A = A;
  p->B = B;
  p->C = (MATRES *)(B + N * N);
  p->N = N;
  return N;
}

/* CoreMark validation-run matrix parameters (see header comment). */
#define CM_MATRIX_BLKSIZE 666
static ee_u8 cm_memblk[CM_MATRIX_BLKSIZE + 16] __attribute__((aligned(16)));

static unsigned coremark_matrix_compute(void) {
  mat_params p;
  core_init_matrix(CM_MATRIX_BLKSIZE, cm_memblk, 0, &p);
  ee_u16 crc = 0;
  crc = core_bench_matrix(&p, 0x66, crc);
  return (unsigned)crc;
}
#endif
