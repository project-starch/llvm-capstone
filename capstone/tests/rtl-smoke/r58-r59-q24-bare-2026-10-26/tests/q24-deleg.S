# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# q24-deleg.S -- Q-24's acceptance arm (2026-10-26): can firmware delegate a CAPABILITY cause to S?
#-----------------------------------------------------------------------------
#
# Q-24 (docs/ref/ISSUES.md) was that the medeleg write mask had no bits for the capability causes 24-30, so an
# S-mode capability fault always landed in M. Fixed in RTL by capstone-ariane 3244170b0 (the mask admits 24-30 under
# CAPSTONE_EXT, clear by default), on every bitstream since the batch c721c1a2a; this arm is its first reading.
#
# THE ARM. M enters capmode, opens PMP entry 0, sets mtvec and stvec, and (under -DQ24_DELEG) writes medeleg = 1 << 24
# (UNEXPECTED_OPERAND_TYPE). It drops to S, where the first instruction is an incoffsetimm on an INTEGER register,
# which raises 24 whatever the privilege (the shape of Q-23's S-mode 24). Everything the arm records stays in
# registers (s6..s9) so that M never needs a capability base for a data access; the readings are printed in M at the
# end, and the test leaves through the capability pass macro, since capability mode refuses the integer tohost store.
#
# PRE-REGISTERED READINGS (written before any run), in print order:
#   medeleg read back      -DQ24_DELEG: 0x1000000 (the bit took; 0 is the pre-fix mask)   control: 0x0
#   scause seen in S       -DQ24_DELEG: 0x18 (24, delivered to S)                          control: 0x0 (S's handler never ran)
#   sepc - fault           -DQ24_DELEG: 0x0                                                 control: 0x0 (never written)
#   mcause seen in M       -DQ24_DELEG: 0x9 (S's ecall, the only trap M sees)               control: 0x18 (24, kept in M)
#   mepc - fault (M)       -DQ24_DELEG: the ecall's offset from fault, 0xc                 control: 0x0 (the fault itself)
# The control (no define) is the pre-fix behaviour by construction: with medeleg clear the 24 stays in M on every tree.
#
# READ 2026-10-26 on the bitstream-2 core (capstone-ariane vp3-ringcut, core/ = dcc6daf6e's): every pre-registered value.
#   q24-deleg-on   0x1000000 0x18 0x0 0x9 0xc
#   q24-deleg      0x0 0x0 0x0 0x18 0x0
# The first cut gave S no CPMP entry over its own code and read 0x1 (instruction access fault) at S's first instruction
# in both arms; the medeleg read-back 0x1000000 was already informative there, the delivery was not.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:
li   gp, 1
la   t0, m_trap_handler
csrw mtvec, t0
csrw mcause, x0
lla  a0, _start
lla  a1, _end_of_text
CAPENTER(a0, a1)
li   t0, 0x001fffffffffffff
csrw pmpaddr0, t0
li   t0, 0x1f
csrw pmpcfg0, t0
# S fetches through the CPMP in capmode: CPMP[0] = an RX capability over S's code (the first cut had none, and S's first
# instruction took an instruction access fault, mcause 1, in both arms)
CAPCREATE(a1)
li   a0, CAP_TYPE_NONLIN
li   a2, CAP_PERM_RX
la   a3, s_mode_start
la   a4, s_code_end
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a2)
CCSRRW(x0, CCSR_CPMP(0), a1)
la   t0, s_trap_handler
csrw stvec, t0
li   s6, 0
li   s7, 0
li   s8, 0
li   s9, 0
#ifdef Q24_DELEG
li   t0, (1 << 24)
csrw medeleg, t0
#else
csrw medeleg, x0
#endif
li   t0, MSTATUS_MPP
csrc mstatus, t0
li   t0, (1 << 11)
csrs mstatus, t0
la   t0, s_mode_start
csrw mepc, t0
mret

s_mode_start:
li   t2, 0x1000                 # an integer: incoffsetimm on it raises 24
fault:
CINCOFFSETIMM(t1, t2, 0)
nop
nop
ecall_site:
ecall

.align 4
s_trap_handler:
csrr s6, scause
csrr t0, sepc
la   t1, fault
sub  s7, t0, t1
la   t0, ecall_site
csrw sepc, t0
sret
s_code_end:

.align 4
m_trap_handler:
csrr s8, mcause
csrr t0, mepc
la   t1, fault
sub  s9, t0, t1
li   t0, MSTATUS_MPP
csrs mstatus, t0
la   t0, verify
csrw mepc, t0
mret

verify:
csrr t0, medeleg
CAPPRINT(t0)                   # 0x1000000 (-DQ24_DELEG) / 0x0
CAPPRINT(s6)                   # 0x18 / 0x0
CAPPRINT(s7)                   # 0x0 / 0x0
CAPPRINT(s8)                   # 0x9 / 0x18
CAPPRINT(s9)                   # 0xc / 0x0
CAP_PASS(s11)

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
