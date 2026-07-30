#ifndef GPN1USE0_KERNEL_H
#define GPN1USE0_KERNEL_H
/* Tight control for gpn2use0: byte-for-byte the same compute, one fewer global.
   beebs_primer1 is a valid count=1 control but differs from gpn2use0 in its whole
   kernel; this differs ONLY by the absence of the second (never-accessed) global, so
   a gpn1use0-pass / gpn2use0-fail pair isolates the extra descriptor record and the
   extra cap-table entry as the sole variable. */
unsigned gpn1use0_a[4];
static unsigned gpn1use0_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<4;i++) gpn1use0_a[i] = (unsigned)(i + 1);
  for (int i=0;i<4;i++) { h ^= gpn1use0_a[i]; h *= 16777619u; }
  return h;
}
#endif
