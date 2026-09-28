#include "reuse-gap-observer.h"
#include <assert.h>
#include <stdint.h>

int main(void) {
  struct reuse_gap_slot slots[8];
  struct reuse_gap_observer o;
  reuse_gap_init(&o, slots, 8);
  reuse_gap_attempt(&o);
  reuse_gap_issue(&o, 0x1000, 16);
  reuse_gap_resize(&o, 0x1000, 24);
  assert(reuse_gap_find(&o, 0x1000)->size == 24);
  reuse_gap_release(&o, 0x1000);
  reuse_gap_attempt(&o); /* failed attempt still increases the gap */
  reuse_gap_attempt(&o);
  reuse_gap_issue(&o, 0x1000, 32);
  assert(o.issues == 2 && o.attempts == 3 && o.releases == 1);
  assert(o.reuses == 1 && o.distinct_starts == 1 && o.bins[1] == 1);
  reuse_gap_release(&o, 0x1000);
  reuse_gap_attempt(&o);
  reuse_gap_issue(&o, 0x2000, 16);
  reuse_gap_attempt(&o);
  reuse_gap_issue(&o, 0x1000, 16);
  assert(o.bins[1] == 2 && o.error == 0);
  reuse_gap_release(&o, 0x2000);
  reuse_gap_release(&o, 0x2000);
  assert(o.error == 3);
  return 0;
}
