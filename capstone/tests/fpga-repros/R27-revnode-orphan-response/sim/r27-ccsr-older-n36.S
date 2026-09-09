#define R27_OLDER ld t6, 0(s11)
#define R27_FENCE CCSRRW(x0, CCSR_CPMP(0), s10)
#define R27_PAD .rept 36; nop; .endr
#include "r27-ccsr-body.h"
