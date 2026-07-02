// Authority suite: a TAGGED capability survives being saved into and restored
// from an intermediate byte buffer via memcpy() *iff* that buffer is 16-byte
// aligned. This is the exact shape of SQLite gap 6: sqlite3NestedParse saves the
// capability-bearing tail of its Parse struct into a `char saveBuf[]` and later
// restores it. When saveBuf is a bare char[] the compiler may place it at a
// non-16-aligned slot; the save memcpy (aligned Parse tail -> misaligned buffer)
// then falls to a byte loop and strips the pointer's out-of-band tag, so the
// restored pointer comes back untagged and later tag-faults.
//
// The fix (and this probe) is layout, not a memcpy change: force the buffer to
// 16-byte alignment so source and destination share alignment and memcpy's
// ldc/stc fast path carries the tag through both the save and the restore. The
// negative counterpart is tagged_cap_memcpy_misaligned (unaligned destination,
// tag necessarily dropped). Together they encode the paper point: on a
// capability machine, storage *layout* is a provenance-correctness property.
//
// Oracle: ok, retval = 0x22AC0001 (the pointer round-trips through the buffer
// with its tag intact and the deref of the restored pointer works). A regression
// (buffer not 16-aligned, or the fast path broken) loses the tag and the restore
// deref tag-faults (no retval), failing the oracle.

typedef __SIZE_TYPE__ bsize_t;
#include "../../../benchmarks/beebs/adapted/beebs_freestanding_string.c"

static long backing[8];

// The cap-bearing "struct tail": a 16-aligned region holding a tagged capability
// (mirrors PARSE_TAIL(pParse), which is 16-aligned in SQLite).
static void *tail[2] __attribute__((aligned(16)));

// The save buffer — 16-aligned, exactly the gap-6 fix (SQLite's saveBuf gets the
// same __attribute__((aligned(16))) via the build's sed patch).
static char saveBuf[32] __attribute__((aligned(16)));

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  tail[0] = &backing[3]; // a tagged capability in the 16-aligned tail
  tail[1] = (void *)0;

  volatile bsize_t n = 16;
  memcpy(saveBuf, tail, n);        // save:    aligned tail  -> aligned buffer
  tail[0] = (void *)0;             // clobber the live copy (as the nested parse does)
  memcpy(tail, saveBuf, n);        // restore: aligned buffer -> aligned tail

  long *p = (long *)tail[0];       // the restored capability
  *p = 0x1234;                     // tag-faults here if the round trip dropped the tag
  *res = 0x22AC0000u | (unsigned)(p == &backing[3]);
}
