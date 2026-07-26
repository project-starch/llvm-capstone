#ifndef MATMULT_INT_KERNEL_H
#define MATMULT_INT_KERNEL_H
/* Silicon-ladder rung 1: integer NxN matmul.
 *
 * All globals are .bss (declared without an initializer, filled in code), so the
 * gp cap-table builder only has to ZERO their carved storage -- the board-proven
 * path (the 554745961 probe was .bss-only). No initializer template is read from
 * the code image, so this rung deliberately sidesteps the "initialized global on
 * silicon" question. It exercises: (1) multiple globals reached via `ldc gp[i]`,
 * (2) array stores inside loops (the shrink-off config), (3) a non-inlined call
 * graph (mm_cell). Shared by the domain and a native host oracle so the expected
 * checksum is computed identically on both sides. */
#define MM_N 8
static int mmA[MM_N][MM_N];
static int mmB[MM_N][MM_N];
static int mmC[MM_N][MM_N];

/* LADDER_ORDERED_EXITS -- board test A (task #65).
 *
 * At -O1 this rung HANGS on silicon; the identical source at -O0 MISCOMPUTES. The
 * codegen difference is total and one-dimensional: -O1 emits 8 conditional branches,
 * ALL `bne`; -O0 emits 8, ALL `blt`. `bne` exits on exact equality, so a loop-control
 * value perturbed by the (still unexplained) silicon fault can step past the bound and
 * the test never fires again -> infinite loop. `blt` exits on ordering and cannot be
 * overshot -> the same perturbation just yields a wrong answer.
 *
 * This knob forces -O1 to emit ordered exits WITHOUT changing the math: mm_bound is
 * the identity, so the oracle is unchanged and a returned value stays checkable (a
 * genuine advantage over mode 7, where retval was 0 by construction). The empty asm
 * emits zero instructions; it only makes the bound opaque to analysis, so the
 * optimizer can no longer prove the induction variable lands on it exactly and must
 * compare with blt/bge.
 *
 *   returns a WRONG value  => PREDICTION CONFIRMED: the hang and the -O0 miscompute
 *                             are ONE fault, and the branch kind selects the symptom.
 *   returns the ORACLE     => the perturbation does not reach this rung's loop control
 *                             after all; the -O1 hang is something else.
 *   still HANGS            => the fragile-exit mechanism is dead.
 *
 * RAN ON SILICON 2026-07-27 (task #65): **STILL HANGS**, identically, two attempts --
 * even though this build verifies as 0 fragile / 8 ordered branches and returns the
 * oracle under QEMU through the same board controller. **The fragile-exit mechanism is
 * REFUTED.** The -O1-all-bne / -O0-all-blt split is real but is a correlate, not the
 * cause. Kept in-tree as the record of a falsified hypothesis (and because the knob is
 * a useful general lever), NOT as a live lead. Trail:
 * history/27-07-2026_00-58-47_RESULTS-65-falsified-66-localizes-hang-to-core_init_matrix.md
 *
 * Laundering the BOUND alone is not enough -- measured: 8 bne became 5 bne + 2 beq,
 * because IndVarSimplify still proves the counter lands on the bound exactly and
 * rewrites `i < n` back to `i != n`. The INCREMENT has to be opaque too (MM_STEP), so
 * no pass can reason about the induction variable at all.
 *
 * Verify the knob did its job before spending a boot: the -O1 build must show ZERO
 * bne/beq and only ordered branches. */
#ifdef LADDER_ORDERED_EXITS
static inline int mm_bound(int x) { __asm__("" : "+r"(x)); return x; }
#define MM_STEP(i) ((i) = mm_bound((i) + 1))
#else
static inline int mm_bound(int x) { return x; }
#define MM_STEP(i) (++(i))
#endif

__attribute__((noinline)) static int mm_cell(int i, int j) {
  int s = 0;
  const int n = mm_bound(MM_N);
  for (int k = 0; k < n; MM_STEP(k)) s += mmA[i][k] * mmB[k][j];
  return s;
}

static unsigned mm_compute(void) {
  const int n = mm_bound(MM_N);
  for (int i = 0; i < n; MM_STEP(i))
    for (int j = 0; j < n; MM_STEP(j)) {
      mmA[i][j] = i + 2 * j + 1;
      mmB[i][j] = 3 * i - j + 2;
    }
  for (int i = 0; i < n; MM_STEP(i))
    for (int j = 0; j < n; MM_STEP(j))
      mmC[i][j] = mm_cell(i, j);
  unsigned h = 2166136261u; /* FNV-1a over the result matrix, LE bytes */
  const int nb = mm_bound(4);
  for (int i = 0; i < n; MM_STEP(i))
    for (int j = 0; j < n; MM_STEP(j)) {
      unsigned v = (unsigned)mmC[i][j];
      for (int b = 0; b < nb; MM_STEP(b)) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
    }
  return h;
}
#endif
