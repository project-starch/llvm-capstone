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

__attribute__((noinline)) static int mm_cell(int i, int j) {
  int s = 0;
  for (int k = 0; k < MM_N; k++) s += mmA[i][k] * mmB[k][j];
  return s;
}

static unsigned mm_compute(void) {
  for (int i = 0; i < MM_N; i++)
    for (int j = 0; j < MM_N; j++) {
      mmA[i][j] = i + 2 * j + 1;
      mmB[i][j] = 3 * i - j + 2;
    }
  for (int i = 0; i < MM_N; i++)
    for (int j = 0; j < MM_N; j++)
      mmC[i][j] = mm_cell(i, j);
  unsigned h = 2166136261u; /* FNV-1a over the result matrix, LE bytes */
  for (int i = 0; i < MM_N; i++)
    for (int j = 0; j < MM_N; j++) {
      unsigned v = (unsigned)mmC[i][j];
      for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
    }
  return h;
}
#endif
