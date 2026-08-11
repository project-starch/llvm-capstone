/* FPGA domain for rung 's06sbase': the baseline store pattern (ldc,stc), run over a 64 KB working set with capabilities
   interspersed, SQLite removed. Read only as a PAIR with its sibling -- see
   s06scale_kernel.h for what each outcome means. */
#include "s06scale_kernel.h"
#define LADDER_COMPUTE s06scale_base
#include "ladder_perf_domain.h"
