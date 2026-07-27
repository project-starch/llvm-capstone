#ifndef BEEBS_BS_KERNEL_H
#define BEEBS_BS_KERNEL_H
/* Silicon-ladder rung: BEEBS bs (binary search) -- a *found* benchmark, faithful.
 *
 * Source: Bristol/Embecosm BEEBS `bs`. Binary search over a sorted 15-entry
 * table. Verbatim compute; no delin, no adaptation.
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS -- and this is the
 * INFORMATIVE one. `bs_data[mid]` is a genuine REGISTER-INDEXED load through a
 * derived capability, exactly the addressing form present in every failing rung.
 * What is absent is the other half of R-1: nothing is ever stored to the table.
 * If R-1 is correctly characterised this must pass, which would show that
 * register-indexed array access is fine on its own and that the intervening
 * store is the necessary ingredient. If it FAILS, R-1 is overstated and the
 * store is not required -- either result is worth the boot. */

struct bs_item { int key; int value; };

static struct bs_item bs_data[15] = {
  {  1, 100}, {  5, 200}, {  6, 300}, {  7, 400}, {  8, 500},
  {  9, 600}, { 10, 700}, { 11, 800}, { 12, 900}, { 15,1000},
  { 20,1100}, { 21,1200}, { 22,1300}, { 23,1400}, { 24,1500},
};

static int bs_search(int x) {
  int fvalue = -1, mid, up = 14, low = 0;
  while (low <= up) {
    mid = (low + up) >> 1;
    if (bs_data[mid].key == x) { up = low - 1; fvalue = bs_data[mid].value; }
    else if (bs_data[mid].key > x) up = mid - 1;
    else                           low = mid + 1;
  }
  return fvalue;
}

static unsigned bs_compute(void) {
  unsigned h = 2166136261u;
  /* every key, plus misses on both sides and in a gap */
  static const int probes[18] = {1,5,6,7,8,9,10,11,12,15,20,21,22,23,24, 0,13,99};
  for (int i = 0; i < 18; i++) {
    int key = probes[i];
    __asm__ volatile("" : "+r"(key));   /* defeat constant-folding of the search */
    int v = bs_search(key);
    h ^= (unsigned)v; h *= 16777619u;
  }
  return h;
}
#endif
