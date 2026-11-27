#!/usr/bin/env python3
"""sbi-tick-apply.py SBI_TREE: the tick log's firmware changes, on a clone of sbi-keep 25b14da6 (runtime lane,
2026-11-27; part 2 added by Amendment 1). Each part is refused if already present.

1. sbi_init.c: select event 2 (L1D miss, perf_counters.sv:115 at b42c8c644) on mhpmcounter3 and let it run, once, on
   the cold boot path after sbi_pmu_init. Stock OpenSBI leaves mhpmevent3 at 0 and inhibits counters 3-31
   (sbi_hart.c mstatus_init); the board kernel has no SBI PMU driver, so nothing changes them afterwards. Both writes
   go through csr_*_allowed, and one console line reports the trap causes (0 = written) and the read-back values:
   "TICKFW mhpmevent3=... mcountinhibit=... trap=...,...".
2. sbi_trap.c and sbi_ecall.c: the M path of every trap taken from U (MPP = U, MPV = 0) is counted. sbi_trap_handler
   becomes a wrapper that reads minstret before and after the stock handler, and adds the difference to cumulative
   counters in a page the kernel names with one ecall (EID 0x0A544B4C, a0 = the page's physical address; the reply
   is a0 = 0, a1 = 0x544B). Counters, u64: [0] machine-timer interrupts from U, [1] their instret, [2] every other
   trap from U, [3] its instret. Before the kernel names a page nothing is counted. The asm path around the wrapper
   (_trap_handler's save and restore) and the wrapper's own instructions outside the two reads are fixed per trap
   and counted from the build's disassembly (PREREG, Amendment 1)."""
import sys

root = sys.argv[1]


def patch(rel, pairs, mark):
    f = f'{root}/{rel}'
    t = open(f).read()
    if mark in t:
        sys.exit(f'sbi-tick-apply: already applied to {f}')
    for old, new in pairs:
        n = t.count(old)
        if n != 1:
            sys.exit(f'sbi-tick-apply: {n} matches for {old[:40]!r} in {f}')
        t = t.replace(old, new)
    open(f, 'w').write(t)
    print('sbi-tick-apply: applied to', f)


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
patch('lib/sbi/sbi_init.c', [(INC, INC + FN), (OLD, OLD + '\ttick_hpm3_init();\n')], 'TICKFW')

TRAP_OLD = 'struct sbi_trap_regs *sbi_trap_handler(struct sbi_trap_regs *regs)\n{\n'
TRAP_NEW = 'static struct sbi_trap_regs *sbi_trap_handler_stock(struct sbi_trap_regs *regs)\n{\n'
WRAP = r'''
/* tick log (runtime lane, 2026-11-27): the M path of traps from U, cumulative in the kernel's page */
volatile unsigned long *tick_mpath;

struct sbi_trap_regs *sbi_trap_handler(struct sbi_trap_regs *regs)
{
	ulong i0 = csr_read(CSR_MINSTRET);
	ulong mcause = csr_read(CSR_MCAUSE);
	bool from_u = ((regs->mstatus & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT) == PRV_U;
	struct sbi_trap_regs *r;

#if __riscv_xlen == 64
	if (regs->mstatus & MSTATUS_MPV)
		from_u = false;
#endif
	r = sbi_trap_handler_stock(regs);
	if (from_u && tick_mpath) {
		ulong d = csr_read(CSR_MINSTRET) - i0;

		if (mcause == (IRQ_M_TIMER | (1UL << (__riscv_xlen - 1)))) {
			tick_mpath[0]++;
			tick_mpath[1] += d;
		} else {
			tick_mpath[2]++;
			tick_mpath[3] += d;
		}
	}
	return r;
}
'''
# the wrapper goes at the end of the file, after the stock handler it calls
f = f'{root}/lib/sbi/sbi_trap.c'
t = open(f).read()
patch('lib/sbi/sbi_trap.c', [(TRAP_OLD, TRAP_NEW), (t[-200:], t[-200:] + WRAP)], 'tick_mpath')

ECALL_OLD = '\tbool is_0_1_spec = 0;\n\n'
ECALL_NEW = '''\tbool is_0_1_spec = 0;

\t/* tick log (runtime lane, 2026-11-27): the kernel names the page for the M-path counters */
\tif (extension_id == 0x0A544B4CUL) {
\t\textern volatile unsigned long *tick_mpath;

\t\ttick_mpath = (volatile unsigned long *)regs->a0;
\t\tregs->mepc += 4;
\t\tregs->a0 = 0;
\t\tregs->a1 = 0x544B;
\t\treturn 0;
\t}

'''
patch('lib/sbi/sbi_ecall.c', [(ECALL_OLD, ECALL_NEW)], 'tick_mpath')
