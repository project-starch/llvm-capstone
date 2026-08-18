#include <stdio.h>
/* Oracle for the QEMU arms (0/1/3/4). QEMU's capability store is one atomic 16-byte-plus-tag
   operation with no write buffer, no per-word entries and no drain arbiter, so the reordering
   under test cannot occur and no tag is ever lost: WBUF_OK with a zero count = 0xB0000000.
   Arm 2 is board-only and has no QEMU oracle -- it clears the tag deliberately and
   op_helper.c:719 aborts on the resulting type query. */
int main(void){printf("2952790016\n");return 0;}
