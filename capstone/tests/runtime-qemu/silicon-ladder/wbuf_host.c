#include <stdio.h>
/* Oracle for the QEMU arms (0/1/3/4). QEMU's capability store is one atomic 16-byte-plus-tag
   operation with no write buffer, no per-word entries and no drain arbiter, so the reordering
   under test cannot occur and no tag is ever lost: WBUF_OK with a zero count = 0xB0000000.
   Arm 2 is board-only and has no QEMU oracle -- it clears the tag deliberately and
   op_helper.c:719 aborts on the resulting type query. */
/* ARMS 9 and 10 carry an IN-ARM POSITIVE CONTROL in bit 24 (see wbuf_kernel.h): the type
   query is asked about a known non-capability and must answer 7. Correct hardware sets that
   bit, so the oracle for those arms is 0xB1000000, not 0xB0000000. The bit is NOT optional
   padding -- if it is clear the arm's loss count carries no verdict, so the oracle demanding
   it is what makes a zero count meaningful. Selected by WBUF_ARM so the other arms' oracles
   are untouched. */
int main(void){
#if WBUF_ARM == 13
  /* Arm 13 additionally reports the SUBJECT capability's type in bits 25-27. The subject is
     `shrink`-derived, and the type MEASURED for it is 1 (NONLIN post-shift) -- the same class
     as the SQLite capability that faults, whose healthy baseline also read 1. So correct
     hardware returns 0xB3000000: type 1, control fired, zero loss. Demanding the type here is
     deliberate: if silicon returns a different class, the arm is not testing what it claims. */
  printf("3003121664\n");      /* 0xB3000000 */
#elif WBUF_ARM == 9 || WBUF_ARM == 10 || WBUF_ARM == 11 || WBUF_ARM == 12
  printf("2969567232\n");      /* 0xB1000000: control fired, zero loss */
#else
  printf("2952790016\n");      /* 0xB0000000: zero loss */
#endif
  return 0;
}
