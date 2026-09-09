#define R27_OLDER
#define R27_FENCE CCSRRW(x0, CCSR_CPMP(0), s10)
#define R27_PAD .rept 16; nop; .endr
#include "r27-ccsr-body.h"
