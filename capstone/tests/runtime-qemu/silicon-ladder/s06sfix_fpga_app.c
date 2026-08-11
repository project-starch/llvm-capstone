/* FPGA domain for rung 's06sfix': the fixup store pattern (ld,ld,ldc,sd,sd,stc), run over a 64 KB working set with capabilities
   interspersed, SQLite removed. Read only as a PAIR with its sibling -- see
   s06scale_kernel.h for what each outcome means. */
#include "s06scale_kernel.h"
#define LADDER_COMPUTE s06scale_fix
#include "ladder_perf_domain.h"
