#define R26_DELAY
#define R26_OLDER ld t6, 0(s11)
#define R26_WRITE csrr t3, mstatus; csrw mstatus, t3
#define R26_EXPECT_OLD 1
#define R26_FENCE fence.i
#include "r26-cscratch-body.h"
