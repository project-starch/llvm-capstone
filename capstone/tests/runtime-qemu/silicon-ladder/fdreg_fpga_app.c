#include "fdreg_kernel.h"
#define LADDER_COMPUTE fdreg_compute
/* Mode 0 (mcycle only, no minstret / no extra region store) on purpose: the extra
   region store in the default mode 4 is a CONFIRMED miscompute trigger on this silicon
   (ladder_perf_domain.h, modes 3/5/6). This rung is a bisection of a wedge, so it must
   not carry a second known-bad construct -- a failure would then be unattributable. */
#define LADDER_INSTR_MODE 0
#include "ladder_perf_domain.h"
