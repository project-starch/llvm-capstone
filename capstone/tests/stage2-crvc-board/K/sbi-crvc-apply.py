#!/usr/bin/env python3
"""sbi-crvc-apply.py OPENSBI_TREE: stage 2's firmware change on a copy of sbi-keep 25b14da6 (plan 2026-11-25, "The
mode's home", Firmware). A misaligned load or store trapped from U while the CRVC mode is on (srevroot bit 0 set and
srevroot[63:12] != 0, MPP = U) whose instruction is in a slot the mode reinterprets (Q0 or Q2, funct3 001 or 101:
c.ldc, c.stc, c.ldcsp, c.stcsp) is redirected to S with its own cause, never emulated as C.FLD/C.FSD/C.FLDSP/C.FSDSP.
On b42c8c644 and its stage-2 seal a misaligned LDC/STC traps with mtval 0 and mtinst 0 (rtl, cva6.sv:1630-1632), so
the test reads only the instruction at mepc and srevroot. Two counters, redirects and emulations from U, print a line
each time either reaches a power of two; a banner line at boot names the change. Every edit asserts one match."""
import sys
T = sys.argv[1].rstrip('/')


def edit(path, old, new, count=1):
    p = f'{T}/{path}'
    s = open(p).read()
    assert s.count(old) == count, (path, s.count(old), old[:60])
    open(p, 'w').write(s.replace(old, new))


HELPER = r'''
/*
 * Capstone stage 2 (CRVC). In protected U with the CRVC mode on, the 16-bit slots Q0/Q2 funct3
 * 001/101 are c.ldc, c.stc, c.ldcsp and c.stcsp, not C.FLD, C.FSD, C.FLDSP and C.FSDSP. A misaligned
 * one is not emulated: it goes to S with its own cause. The mode is bit 0 of srevroot, read back,
 * and protected U is srevroot[63:12] != 0 with MPP = U (the RTL's cap_data_ctx[0] less sup_active,
 * which Line B never sets). The trap carries mtval 0 and mtinst 0 on the board's RTL, so only the
 * instruction at mepc is read.
 */
static unsigned long crvc_redirects, crvc_u_emulations;

static void crvc_report(unsigned long n)
{
	if (n && !(n & (n - 1)))
		sbi_printf("FWCRVC redirect=%lu emulate-u=%lu\n", crvc_redirects,
			   crvc_u_emulations);
}

static bool crvc_from_u(struct sbi_trap_regs *regs)
{
#if __riscv_xlen == 64
	if (regs->mstatus & MSTATUS_MPV)
		return false;
#endif
	return ((regs->mstatus & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT) == PRV_U;
}

static bool crvc_decoded(struct sbi_trap_regs *regs, ulong insn, ulong tinst)
{
	ulong root, q, f3;

	if (!crvc_from_u(regs))
		return false;
	root = csr_read(0x5c1);
	if (!(root & 1) || !(root >> 12))
		return false;
	if (tinst & 0x1)	/* a transformed instruction: a 16-bit one's slot is unknown, so redirect */
		return !(tinst & 0x2);
	if ((insn & 3) == 3)
		return false;
	q  = insn & 3;
	f3 = (insn >> 13) & 7;
	return q != 1 && (f3 == 1 || f3 == 5);
}

static void crvc_count_emulation(struct sbi_trap_regs *regs)
{
	if (crvc_from_u(regs))
		crvc_report(++crvc_u_emulations);
}
'''

edit('lib/sbi/sbi_misaligned_ldst.c', '#include <sbi/sbi_error.h>\n',
     '#include <sbi/sbi_console.h>\n#include <sbi/sbi_error.h>\n')
edit('lib/sbi/sbi_misaligned_ldst.c',
     'static ulong sbi_misaligned_tinst_fixup(',
     HELPER.lstrip('\n') + '\nstatic ulong sbi_misaligned_tinst_fixup(')

# the load handler: the check right after the instruction is known, the count at the emulation's end
edit('lib/sbi/sbi_misaligned_ldst.c',
     '\t\tinsn_len = INSN_LEN(insn);\n\t}\n\n\tif ((insn & INSN_MASK_LW) == INSN_MATCH_LW) {',
     '\t\tinsn_len = INSN_LEN(insn);\n\t}\n\n'
     '\tif (crvc_decoded(regs, insn, tinst)) {\n'
     '\t\tuptrap.epc = regs->mepc;\n\t\tuptrap.cause = CAUSE_MISALIGNED_LOAD;\n'
     '\t\tuptrap.tval = addr;\n\t\tuptrap.tval2 = tval2;\n\t\tuptrap.tinst = tinst;\n'
     '\t\tuptrap.gva   = sbi_regs_gva(regs);\n'
     '\t\tcrvc_report(++crvc_redirects);\n'
     '\t\treturn sbi_trap_redirect(regs, &uptrap);\n\t}\n\n'
     '\tif ((insn & INSN_MASK_LW) == INSN_MATCH_LW) {')
edit('lib/sbi/sbi_misaligned_ldst.c',
     '\t\tinsn_len = INSN_LEN(insn);\n\t}\n\n\tval.data_ulong = GET_RS2(insn, regs);\n',
     '\t\tinsn_len = INSN_LEN(insn);\n\t}\n\n'
     '\tif (crvc_decoded(regs, insn, tinst)) {\n'
     '\t\tuptrap.epc = regs->mepc;\n\t\tuptrap.cause = CAUSE_MISALIGNED_STORE;\n'
     '\t\tuptrap.tval = addr;\n\t\tuptrap.tval2 = tval2;\n\t\tuptrap.tinst = tinst;\n'
     '\t\tuptrap.gva   = sbi_regs_gva(regs);\n'
     '\t\tcrvc_report(++crvc_redirects);\n'
     '\t\treturn sbi_trap_redirect(regs, &uptrap);\n\t}\n\n'
     '\tval.data_ulong = GET_RS2(insn, regs);\n')
edit('lib/sbi/sbi_misaligned_ldst.c',
     '\tregs->mepc += insn_len;\n\n\treturn 0;\n}\n',
     '\tregs->mepc += insn_len;\n\tcrvc_count_emulation(regs);\n\n\treturn 0;\n}\n', count=2)

# the banner
edit('lib/sbi/sbi_init.c',
     '\tsbi_printf("Platform Name             : %s\\n",',
     '\tsbi_printf("FWCRVC: stage-2 misaligned redirect (sbi-keep 25b14da6 + sbi-crvc-apply.py)\\n");\n'
     '\tsbi_printf("Platform Name             : %s\\n",')
print('sbi-crvc-apply: 6 edits applied to', T)
