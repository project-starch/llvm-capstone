#ifndef REENTRY_KERNEL_H
#define REENTRY_KERNEL_H
/* S2 gate: does a global SURVIVE a domreturn?
 *
 * Before S2 both glue entry points rebuilt the cap table and re-ran every
 * initializer, so a global written on entry 1 was reset before entry 2 read it.
 * This rung makes that visible: entry_count is bumped on every entry and folded
 * into the returned value, so entry 2 returns a DIFFERENT number from entry 1
 * only if the global persisted.
 *
 * Both an initialized (.data) and a zero-init (.bss) global are covered, because
 * the glue treats them through different paths (blob copy vs zero loop).
 *
 * This is the shape sqlite_capstone_domain.c has: two CAPSTONE_DPI_REGION_SHARE
 * entries stashing host capabilities in globals, then a run entry. If globals do
 * not survive, SQLite runs with no host channel. */
unsigned re_entry_count;              /* .bss  -- zero-init path                  */
unsigned re_accum = 0x1234u;          /* .data -- blob-copy path                  */

static unsigned reentry_compute(void){
  re_entry_count += 1;
  re_accum = re_accum * 31u + re_entry_count;
  unsigned h = 2166136261u;
  h ^= re_entry_count; h *= 16777619u;
  h ^= re_accum;       h *= 16777619u;
  return h;
}
#endif
