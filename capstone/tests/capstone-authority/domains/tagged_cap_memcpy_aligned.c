// Authority suite: a TAGGED capability must survive a real memcpy() call when
// source and destination share 16-byte alignment.
//
// This is the discriminating test for the SQLite gap-5 hypothesis. memcpy's
// capability-preserving fast path copies the pointer-aligned middle via ldc/stc
// (which preserve the tag); it only fires when src and dst share alignment. If
// the alignment logic is wrong, a stored capability comes back untagged and the
// deref of the copied pointer tag-faults. The spill_reachability probe already
// shows the raw ldc/stc tagged path works, so a fault here isolates the fault
// to memcpy's own logic rather than the emulator primitive.
//
// We #include the freestanding string source so the probe links the exact
// memcpy under test (build-domain.sh provides no libc). A volatile length keeps
// the compiler from lowering the call to an inline builtin.
//
// Oracle: ok, retval = 0x22AA0001 (tag survived; copied pointer equals the
// original and its deref works). A regression loses the tag and the deref
// tag-faults (no retval), failing the oracle.

typedef __SIZE_TYPE__ bsize_t;
#include "../../../benchmarks/beebs/adapted/beebs_freestanding_string.c"

static long backing[8];
static void *src_buf[2] __attribute__((aligned(16)));
static void *dst_buf[2] __attribute__((aligned(16)));

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  src_buf[0] = &backing[3]; // a tagged capability in a 16-aligned slot
  src_buf[1] = (void *)0;
  dst_buf[0] = (void *)0;
  dst_buf[1] = (void *)0;

  volatile bsize_t n = 16;
  memcpy(dst_buf, src_buf, n); // aligned -> ldc/stc fast path

  long *p = (long *)dst_buf[0]; // the COPIED capability
  *p = 0x1234;                  // tag-faults here if the copy dropped the tag
  *res = 0x22AA0000u | (unsigned)(p == &backing[3]);
}
