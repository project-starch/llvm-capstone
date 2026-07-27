#ifndef CTRSANITY4_KERNEL_H
#define CTRSANITY4_KERNEL_H
/* 4x the work of ctrsanity, same kernel. Two lengths separate a PROPORTIONAL
   counter effect (ratio unchanged as work grows) from a FIXED one (ratio moves
   toward 1.0). See ctrsanity_kernel.h for the full rationale. */
#define CTRSANITY_N 400000L
#include "ctrsanity_kernel.h"
#endif
