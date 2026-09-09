#define R26_OLDER
#define R26_WRITE csrr t3, mstatus; csrw mstatus, t3
#define R26_EXPECT_OLD 1
#define R26_FENCE fence.i
#define R26_PAD
#include "r26-mret-body.h"
