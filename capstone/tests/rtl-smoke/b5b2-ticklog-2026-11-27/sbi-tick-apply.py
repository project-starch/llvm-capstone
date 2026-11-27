#!/usr/bin/env python3
"""sbi-tick-apply.py SBI_TREE: on a clone of sbi-keep 25b14da6, select event 2 (L1D miss, perf_counters.sv:115 at
b42c8c644) on mhpmcounter3 and let it run, once, on the cold boot path after sbi_pmu_init (runtime lane, 2026-11-27).
Stock OpenSBI leaves mhpmevent3 at 0 and inhibits counters 3-31 (sbi_hart.c mstatus_init); the board kernel has no
SBI PMU driver, so nothing changes them afterwards. Both writes go through csr_*_allowed, and one console line reports
the trap causes (0 = written) and the read-back values: "TICKFW mhpmevent3=... mcountinhibit=... trap=...,..."."""
import sys

f = sys.argv[1] + '/lib/sbi/sbi_init.c'
t = open(f).read()
if 'TICKFW' in t:
    sys.exit('sbi-tick-apply: already applied to ' + f)
INC = '#include <sbi/sbi_version.h>\n'
FN = r'''
#include <sbi/sbi_csr_detect.h>
/* tick log (runtime lane, 2026-11-27): hpmcounter3 counts L1D misses for the kernel's per-tick log */
static void tick_hpm3_init(void)
{
	struct sbi_trap_info t1 = {0}, t2 = {0};
	unsigned long ev, inh;

	csr_write_allowed(CSR_MHPMEVENT3, (ulong)&t1, 2UL);
	inh = csr_read_allowed(CSR_MCOUNTINHIBIT, (ulong)&t2);
	if (!t2.cause)
		csr_write_allowed(CSR_MCOUNTINHIBIT, (ulong)&t2, inh & ~(1UL << 3));
	ev = t1.cause ? 0 : csr_read(CSR_MHPMEVENT3);
	inh = t2.cause ? 0 : csr_read(CSR_MCOUNTINHIBIT);
	sbi_printf("TICKFW mhpmevent3=%lu mcountinhibit=0x%lx trap=%lu,%lu\n", ev, inh, t1.cause, t2.cause);
}
'''
OLD = '''	rc = sbi_pmu_init(scratch, true);
	if (rc) {
		sbi_printf("%s: pmu init failed (error %d)\\n",
			   __func__, rc);
		sbi_hart_hang();
	}
'''
for old, new in ((INC, INC + FN), (OLD, OLD + '\ttick_hpm3_init();\n')):
    n = t.count(old)
    if n != 1:
        sys.exit(f'sbi-tick-apply: {n} matches for {old[:40]!r}')
    t = t.replace(old, new)
open(f, 'w').write(t)
print('sbi-tick-apply: applied to', f)
