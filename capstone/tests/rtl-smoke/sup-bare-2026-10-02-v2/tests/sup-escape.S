# See LICENSE for license details.

#*****************************************************************************
# sup-escape.S  (supervised CALL on silicon, verification step 3: the synchronous-fault escape)
#-----------------------------------------------------------------------------
#
# The monitor arms a seal (CSSUPERVISE rd=t3, rs1=the seal, rs2=the 1 KiB private save area), CALLs it, and the
# domain faults. The fault must NOT be a trap: commit strips it, saves the domain's registers 3..66 into its seal
# region, restores the monitor's from the save area, hands the SEALED seal back in a3 and continues at CALL+4.
# Five arms, each with its own seal and stub (a faulted domain is never CALLed again):
#   0 ecall (cause 11)   1 an illegal instruction (2)   2 `csrw mepc` under the supervision CSR gate (2)
#   3 a misaligned load through a NONLIN capability in x26 (4, tval = the address)   4 mret under supervision (2)
# The domain scribbles x30/x7/x29 before faulting: the monitor must get its own values back.
#
# Pre-registered readings per arm (CAPPRINT values in the retirement trace, in order):
#   status 0 (armed)   csupstatus 5 (kind 2 = fault, valid)   csupcause = the cause above   csupepc - fault_k = 0
#   csuptval (ecall/illegal/csr: the instruction bits; misaligned: the reference trap's mtval; mret: its bits)   lcc (a3) = 4 (SEALED)
#   x30 0xA5   x7 0x707   x29 0x1D1D   mcause 0 (no trap was taken)   mie 0x888   mepc 0 (restored from the private area, so
#   it CANNOT tell the strip -- an earlier header said it could)   the domain's saved mcause slot in its seal (+864, read
#   through a NONLIN alias the monitor keeps) = 0 -- THE no-strip mutant's tell: with the strip gone the fault is taken as a
#   trap, which writes the cause into the live mcause before the SAVE walk parks it there (the walks then restore the
#   monitor's own mcause/mepc, which is why the readings above cannot see it; found by the board lane's auditor, 2026-10-02)
#   csupstatus again 4 (valid cleared by the read)
# A trap anywhere ends the test with the trap handler's mcause as the last reading.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2
#define CSUPTVAL    0xFC3

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

# s5: the monitor's private save area, LINEAR RWX, 1 KiB, 64-byte aligned
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# x26: a NONLIN capability over a 64-byte RAM scratch region for the misaligned-load arm (the domain inherits it at
# first entry; x6 is t1, which the readout's `la t1` overwrites -- measured 2026-10-01 as a plain load at a code address,
# cause 4, tval pc+1). It used to cover [0x3000, 0x4000), which the test harness does not map; a misaligned load belongs
# in mapped memory, so it now covers a written RAM granule (2026-10-02).
CAPCREATE(x26)
li a0, CAP_TYPE_NONLIN
lla a3, scratch
lla a4, scratch_end
li a5, CAP_PERM_RWX
CAPTYPE(x26, a0)
CAPBOUND(x26, a3, a4)
CAPPERM(x26, a5)
sd x0, 0(x26)                  # write the granule once, aligned: a never-written granule's tag is not reliably 0 in this
sd x0, 8(x26)                  # simulation (the tag array starts uninitialised), and a misaligned load's return shows the tag

# s6..s9 and x31: NONLIN aliases of the five seal regions, so the monitor can read each domain's SAVED mcause slot
# (+864) after its escape (the SEALED seal in a3 cannot be read through). NOT s10: s10 is x26, the misaligned arm's
# capability -- a first version aliased seal4 in s10, so arm 3's domain loaded from seal4 + 1 instead, a never-written
# granule whose tag is not 0 in this simulation, and the sim-only S-06 invariant stopped the run (2026-10-02).
CAPCREATE(s6)
li a0, CAP_TYPE_NONLIN
lla a3, seal0
lla a4, seal0_end
li a5, CAP_PERM_RWX
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)
CAPCREATE(s7)
li a0, CAP_TYPE_NONLIN
lla a3, seal1
lla a4, seal1_end
li a5, CAP_PERM_RWX
CAPTYPE(s7, a0)
CAPBOUND(s7, a3, a4)
CAPPERM(s7, a5)
CAPCREATE(s8)
li a0, CAP_TYPE_NONLIN
lla a3, seal2
lla a4, seal2_end
li a5, CAP_PERM_RWX
CAPTYPE(s8, a0)
CAPBOUND(s8, a3, a4)
CAPPERM(s8, a5)
CAPCREATE(s9)
li a0, CAP_TYPE_NONLIN
lla a3, seal3
lla a4, seal3_end
li a5, CAP_PERM_RWX
CAPTYPE(s9, a0)
CAPBOUND(s9, a3, a4)
CAPPERM(s9, a5)
CAPCREATE(x31)
li a0, CAP_TYPE_NONLIN
lla a3, seal4
lla a4, seal4_end
li a5, CAP_PERM_RWX
CAPTYPE(x31, a0)
CAPBOUND(x31, a3, a4)
CAPPERM(x31, a5)

li t0, 0x100000
csrw CSUPQUANTUM, t0           # a quantum far longer than any arm
csrwi CSUPCTL, 0               # first entries, never a resume

# reference: the same misaligned load as arm 3, executed by the monitor itself -> a real trap; the handler prints
# mcause (4) and mtval (scratch + 1) and steps over it, so arm 3's csupcause/csuptval must equal these two readings
la t0, ref_trap_handler
csrw mtvec, t0
ld t0, 1(x26)
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

# ---------------------------------------------------------------- arm 0: ecall
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub0
lla a4, stub0_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal0
lla a4, seal0_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)                 # slot 0: the stub's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
li x30, 0xA5
li x7, 0x0707
li x29, 0x1D1D
csrw mcause, x0
csrw mepc, x0                  # mepc is neither exchanged nor walked: a trap taken instead of an escape would leave the faulting pc here
li t0, 0x888
csrw mie, t0                   # the monitor's mie, to be read back after the escape
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status: 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 = fault event, unread
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 11
csrr t0, CSUPEPC
la t1, fault0
sub t0, t0, t1
CAPPRINT(t0)                   # 0: the faulting instruction's own pc
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # 0x73 (the instruction bits)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4: the seal came back SEALED
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x7, 0
CAPPRINT(s2)                   # 0x707
addi s2, x29, 0
CAPPRINT(s2)                   # 0x1D1D
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's own mie, not the domain's image
csrr t0, mepc
CAPPRINT(t0)                   # 0: restored from the private area (cannot tell the strip)
ld t0, 864(s6)
CAPPRINT(t0)                   # 0: the domain's saved mcause slot -- the no-strip mutant parks the cause here
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 4: the first read cleared valid

# ---------------------------------------------------------------- arm 1: illegal instruction
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub1
lla a4, stub1_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal1
lla a4, seal1_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)                 # slot 0: the stub's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
li x30, 0xA5
li x7, 0x0707
li x29, 0x1D1D
csrw mcause, x0
csrw mepc, x0                  # mepc is neither exchanged nor walked: a trap taken instead of an escape would leave the faulting pc here
li t0, 0x888
csrw mie, t0                   # the monitor's mie, to be read back after the escape
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status: 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 = fault event, unread
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault1
sub t0, t0, t1
CAPPRINT(t0)                   # 0: the faulting instruction's own pc
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # 0xffffffff
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4: the seal came back SEALED
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x7, 0
CAPPRINT(s2)                   # 0x707
addi s2, x29, 0
CAPPRINT(s2)                   # 0x1D1D
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's own mie, not the domain's image
csrr t0, mepc
CAPPRINT(t0)                   # 0: restored from the private area (cannot tell the strip)
ld t0, 864(s7)
CAPPRINT(t0)                   # 0: the domain's saved mcause slot -- the no-strip mutant parks the cause here
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 4: the first read cleared valid

# ---------------------------------------------------------------- arm 2: csrw mepc under the CSR gate
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub2
lla a4, stub2_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal2
lla a4, seal2_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)                 # slot 0: the stub's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
li x30, 0xA5
li x7, 0x0707
li x29, 0x1D1D
csrw mcause, x0
csrw mepc, x0                  # mepc is neither exchanged nor walked: a trap taken instead of an escape would leave the faulting pc here
li t0, 0x888
csrw mie, t0                   # the monitor's mie, to be read back after the escape
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status: 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 = fault event, unread
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault2
sub t0, t0, t1
CAPPRINT(t0)                   # 0: the faulting instruction's own pc
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # the csrw's bits
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4: the seal came back SEALED
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x7, 0
CAPPRINT(s2)                   # 0x707
addi s2, x29, 0
CAPPRINT(s2)                   # 0x1D1D
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's own mie, not the domain's image
csrr t0, mepc
CAPPRINT(t0)                   # 0: restored from the private area (cannot tell the strip)
ld t0, 864(s8)
CAPPRINT(t0)                   # 0: the domain's saved mcause slot -- the no-strip mutant parks the cause here
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 4: the first read cleared valid

# ---------------------------------------------------------------- arm 3: misaligned load through x26
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub3
lla a4, stub3_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal3
lla a4, seal3_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)                 # slot 0: the stub's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
li x30, 0xA5
li x7, 0x0707
li x29, 0x1D1D
csrw mcause, x0
csrw mepc, x0                  # mepc is neither exchanged nor walked: a trap taken instead of an escape would leave the faulting pc here
li t0, 0x888
csrw mie, t0                   # the monitor's mie, to be read back after the escape
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status: 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 = fault event, unread
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 4
csrr t0, CSUPEPC
la t1, fault3
sub t0, t0, t1
CAPPRINT(t0)                   # 0: the faulting instruction's own pc
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # scratch + 1 (equal to the reference trap's mtval above)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4: the seal came back SEALED
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x7, 0
CAPPRINT(s2)                   # 0x707
addi s2, x29, 0
CAPPRINT(s2)                   # 0x1D1D
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's own mie, not the domain's image
csrr t0, mepc
CAPPRINT(t0)                   # 0: restored from the private area (cannot tell the strip)
ld t0, 864(s9)
CAPPRINT(t0)                   # 0: the domain's saved mcause slot -- the no-strip mutant parks the cause here
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 4: the first read cleared valid

# ---------------------------------------------------------------- arm 4: mret under supervision
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub4
lla a4, stub4_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal4
lla a4, seal4_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)                 # slot 0: the stub's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
li x30, 0xA5
li x7, 0x0707
li x29, 0x1D1D
csrw mcause, x0
csrw mepc, x0                  # mepc is neither exchanged nor walked: a trap taken instead of an escape would leave the faulting pc here
li t0, 0x888
csrw mie, t0                   # the monitor's mie, to be read back after the escape
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status: 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 = fault event, unread
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault4
sub t0, t0, t1
CAPPRINT(t0)                   # 0: the faulting instruction's own pc
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # 0x30200073
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4: the seal came back SEALED
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x7, 0
CAPPRINT(s2)                   # 0x707
addi s2, x29, 0
CAPPRINT(s2)                   # 0x1D1D
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's own mie, not the domain's image
csrr t0, mepc
CAPPRINT(t0)                   # 0: restored from the private area (cannot tell the strip)
ld t0, 864(x31)
CAPPRINT(t0)                   # 0: the domain's saved mcause slot -- the no-strip mutant parks the cause here
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 4: the first read cleared valid

CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
ref_trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # reference: 4
  csrr s4, mtval
  CAPPRINT(s4)                 # reference: what a trap gives as tval for `ld t0, 1(x26)`
  csrr s4, mepc
  addi s4, s4, 4
  csrw mepc, s4
  mret

.align 4
stub0:
  li x30, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
fault0:
  ecall
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub0_end:

.align 4
stub1:
  li x30, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
fault1:
  .word 0xffffffff
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub1_end:

.align 4
stub2:
  li x30, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
fault2:
  csrw mepc, x0
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub2_end:

.align 4
stub3:
  li x30, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
fault3:
  ld t0, 1(x26)
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub3_end:

.align 4
stub4:
  li x30, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
fault4:
  mret
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub4_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
scratch:
  .zero 64
scratch_end:
.align 6
seal0:
  .zero 2048
seal0_end:
.align 6
seal1:
  .zero 2048
seal1_end:
.align 6
seal2:
  .zero 2048
seal2_end:
.align 6
seal3:
  .zero 2048
seal3_end:
.align 6
seal4:
  .zero 2048
seal4_end:
  TEST_DATA
RVTEST_DATA_END
