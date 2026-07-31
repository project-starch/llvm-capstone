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
 *   2. (WITHDRAWN 2026-07-27.) B/C are carved from a single gp-delivered block
 *      (B = A + N*N). We used to CAPSTONE_DELIN(A) first, believing the block cap
 *      was LINEAR and that `cincoffset rd!=rs1` would consume A. Board tests
 *      disproved it: without the delin the derivation works and the probe returns
 *      correctly (#67f), while WITH it the domain wedges the board (#67c vs the
 *      size-matched control #67e). The delin is gone from the default build, so
 *      align_mem (item 1) is now the ONLY deviation from upstream.
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

  /* ---- SPLIT POINT 1 (board task #67a) -------------------------------------
   * Return N with ONLY the dimension while-loop executed: no block-capability
   * use at all, no delin, no seeding. Isolates candidate (A), the `mulw`-driven
   * `bgeu` exit, from everything downstream. Expected N = 9. */
#ifdef LADDER_CM_STOP_AFTER_WHILE
  return N;
#endif

  A = (MATDAT *)memblk; /* block is 16-aligned; align_mem is identity (see top) */
#if defined(__capstone) && defined(LADDER_CM_WITH_DELIN)
  /* REMOVED FROM THE DEFAULT BUILD 2026-07-27 -- kept only as a board probe.
   *
   * This `delin` was OUR adaptation, not upstream CoreMark. It was added
   * defensively on the theory that the gp-delivered block cap is LINEAR, so
   * `B = A + N*N` (a `cincoffset` with rd != rs1) would consume A.
   *
   * That theory is WRONG on hardware, and the defensive instruction was worse
   * than the problem it guarded against:
   *   - board #67f: with this delin removed, `B = A + N*N` works fine and the
   *     probe RETURNS the correct N. A is not consumed.
   *   - board #67c/#67e: with it present, the domain WEDGES the board -- proven
   *     against a size-matched `addi x0,x0,0` control at the same address, so it
   *     is the instruction and not code layout.
   *   - QEMU computes the correct 14343 either way.
   * The cap-table glue already delins every entry before `stc`-ing it, so on a
   * machine that preserves capability type through a memory round-trip this was
   * a redundant NONLIN->NONLIN delin all along.
   *
   * Dropping it moves this kernel CLOSER TO VANILLA CoreMark. The only remaining
   * deviation is the align_mem omission (item 1 in the header), which is forced
   * by PureCap -- upstream builds a pointer from an integer, which drops the tag.
   *
   * Define LADDER_CM_WITH_DELIN to restore it for board experiments only. */
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(A));
#endif
#if defined(__capstone) && defined(LADDER_CM_DELIN_NOP_CONTROL)
  /* SIZE-MATCHED CONTROL for board #67c (task #67e).
   * #67c = #67a + one `delin` and it HANGS -- but the 2026-07-26 A/B showed that
   * adding FOUR instructions to domain_main flipped a passing rung, so a
   * one-instruction delta cannot be attributed to the delin without controlling
   * for code layout. This emits a 4-byte `addi x0, x0, 0` in the delin's place,
   * with the identical "+r"(A) register plumbing, so the image differs from #67c
   * only in that instruction's ENCODING.
   *   control RETURNS => the delin itself is the fault.
   *   control HANGS   => it is layout/perturbation, not the delin. */
  __asm__ volatile(".insn i 0x13, 0x0, x0, 0(x0)" : "+r"(A));
#endif

  /* ---- SPLIT POINT 1b (board task #67c) ------------------------------------
   * Return N after the `delin` but BEFORE `B = A + N*N`. #67a RETURNS and #67b
   * HANGS, so the fault is in the two-operation window between them; this splits
   * that window:
   *   #67c HANGS    => the `delin` itself diverges on the RTL (cf. C_GEN_CAP,
   *                    which is a QEMU-only op the RTL decodes as a no-op).
   *   #67c RETURNS  => the fault is `cincoffset rd, rs1, rs2` with rd != rs1 off
   *                    the block cap -- i.e. exactly the linear-consume the
   *                    delin above is supposed to prevent. */
#ifdef LADDER_CM_STOP_AFTER_DELIN
  (void)A;
  return N;
#endif

  B = A + N * N;

  /* ---- SPLIT POINT 2 (board task #67b) -------------------------------------
   * Return N after the delin + `B = A + N*N` derivation, but BEFORE the seeding
   * loop. Isolates candidate (B) -- the delin / rd!=rs1 derivation off the block
   * cap -- from candidate (C), the N*N seeding loop.
   *
   * WHY THIS EXISTS: #66 ("return N after the whole function") HANGS, and the
   * pre-registered follow-up "return N before the seeding loop" was described as
   * a 2-way split of (A) vs (C). It is not -- this derivation block sits between
   * them, so that probe alone leaves two candidates on its HANG branch. With
   * both split points the three are separated in two boots:
   *   #67a HANGS                  => (A) the while loop
   *   #67a returns, #67b HANGS    => (B) the delin / B-derivation
   *   #67a and #67b both return   => (C) the seeding loop
   * (C) is where the surviving static candidate lives: coremark_matrix is the
   * only rung in the corpus doing narrow `sh` stores through the block cap. */
#ifdef LADDER_CM_STOP_AFTER_DERIVE
  (void)B;
  return N;
#endif

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

/* LADDER_CM_INIT_ONLY -- board test B (task #66).
 *
 * This rung hangs on silicon at -O0, and unlike matmult_int it does NOT fit the
 * fragile-exit story: its exits are overwhelmingly ordered (17 bgeu, 1 bge, 1 blt),
 * and an ordered exit cannot be overshot. But it has the one thing matmult_int lacks --
 * a loop bound computed at RUNTIME. core_init_matrix derives the matrix dimension with
 *
 *     while (j < blksize) { i++; j = i * i * 2 * 4; }   N = i - 1;
 *
 * which lowers to `bgeu a0, a1` at 0x10428 driven by `mulw a0, a0, a0` at 0x10444. An
 * ordered exit still never fires if the multiply never produces a value reaching
 * blksize (666) -- and N then feeds EVERY downstream loop bound via p->N.
 *
 * So return N immediately and skip the benchmark. Three-way outcome:
 *   HANGS          => the fault IS this while loop (the mulw).
 *   returns N != 9 => the miscompute is caught red-handed at its source, and it
 *                     explains every downstream bound in one step.
 *   returns N == 9 => init is clean; the hang is downstream of it.
 *
 * Expected N = 9: i*i*8 first reaches 666 at i=10 (800), so N = i-1 = 9. The native
 * host oracle computes it from this same header, so the runner's gate still applies.
 *
 * RAN ON SILICON 2026-07-27 (task #66): **HANGS.** Against mode 7 at the same
 * -O0 @32KiB config (entry path only -> RETURNS), this localizes the fault to INSIDE
 * core_init_matrix -- the whole benchmark narrowed to this one function.
 *
 * But the three-way reading above was TOO NARROW: "hangs => it is the while loop" is
 * wrong, because core_init_matrix contains a second candidate -- the N*N seeding loop
 * below, which runs `seed = ((order * seed) % 65536)` (a multiply AND a remainder) per
 * element and writes A[]/B[] through the gp-delivered block capability. Both would hang.
 * NEXT PROBE: return N *before* the seeding loop to separate them (one boot). */
static unsigned coremark_matrix_compute(void) {
  mat_params p;
  ee_u32 n = core_init_matrix(CM_MATRIX_BLKSIZE, cm_memblk, 0, &p);
#if defined(LADDER_CM_INIT_ONLY) || defined(LADDER_CM_STOP_AFTER_WHILE) \
    || defined(LADDER_CM_STOP_AFTER_DELIN) || defined(LADDER_CM_STOP_AFTER_DERIVE)
  return (unsigned)n;
#else
  (void)n;
  ee_u16 crc = 0;
  crc = core_bench_matrix(&p, 0x66, crc);
  return (unsigned)crc;
#endif
}
#endif
