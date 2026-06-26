// Authority suite: out-of-bounds read that stays INSIDE the loadable segment
// but crosses out of the victim object into an adjacent global (PI Q8, T3).
//
// Runtime fact established by this suite: a pointer to a global inherits the
// capability bounds of its loadable *segment*, not its *object* (observed
// bounds span ~hundreds of bytes, not sizeof(obj)). So an over-read that leaves
// the object but stays within the segment is NOT caught today. `neighbour` is a
// large global placed after `a` so that a[100] (36 bytes past a[64]) lands in
// `neighbour`, comfortably inside the segment bound.
//
// Oracle TODAY: no-trap. The read succeeds (cross-object over-read) with no
//   "Cap mem access OOB" diagnostic -- this is the granularity gap.
// Oracle AFTER object-SHRINK (Step 3): bounds-fault -- `a`'s capability is
//   narrowed to [a, a+64), so a[100] traps.
//
// This is the measurable before/after for the granularity contribution.

// `a` is INITIALIZED so it lands in `.data`; the image segment order is
// .data -> .capstone_cap_init -> .bss, so the `.bss` objects (including
// `neighbour` and the harness's own bss) sit AFTER `a` in the same loadable
// segment. That guarantees a[100] lands at an in-segment address past `a`'s
// object, exercising the cross-object (not cross-segment) over-read.
static unsigned char a[64] = {1};
static unsigned char neighbour[4096]; // .bss, after `a`'s .data; in-segment space

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  neighbour[0] = (unsigned char)func;   // keep `neighbour` live in the segment
  volatile unsigned idx = 100;          // a[100]: 36 bytes past a[64], into neighbour
  *res = 0x00B00000u | (unsigned)a[idx]; // in-segment over-read; traps only after SHRINK
}
