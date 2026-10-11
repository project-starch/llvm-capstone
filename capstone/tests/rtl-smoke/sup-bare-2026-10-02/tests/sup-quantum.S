# See LICENSE for license details.

#*****************************************************************************
# sup-quantum.S  (supervised CALL on silicon, verification step 4: the quantum escape and exactly-once resume)
#-----------------------------------------------------------------------------
#
# The monitor arms a seal with a 64-cycle quantum and CALLs it. The domain runs 400 `addi x20, x20, 1` in a
# straight line, stores x20 through s1 (a LINEAR capability the monitor set up; the domain inherits it at first
# entry and the walks carry it across every escape) and RETURNs. Every quantum expiry escapes to the monitor as
# kind 1 (preempted, cause 0x8000000000000010); the monitor sets csupctl.resume, re-arms and CALLs again, until
# the RETURN arrives as kind 0. Exactly-once resume: the stored x20 must be exactly 400 -- a skipped instruction
# (resume at pc + 4) reads below 400, a doubled one above. The monitor's own loop counter s7 and signature x30
# must survive every switch (its registers 3..66 are restored from the private save area).
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   per escape: csupstatus 3 (kind 1, valid), csupcause 0x8000000000000010 -- repeated N >= 3 times
#   then: csupstatus 1 (kind 0, valid: the return), the stored x20 = 0x190 (400), the stored x1 = 0x1234 (the
#   domain's live return address survives every resume), s7 = N, x30 = 0xA5, mie 0x888, mcause 0.
#   Mutants (positive controls): resume at pc + 4 -> the stored value < 400; the counter decrementing during the
#   switcher's walks with this quantum -> the resume never completes (the harness times out).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1

#ifndef QUANTUM
#define QUANTUM 64
#endif

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

# s5: the monitor's private save area
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# s1: the result word the domain stores x20 through
CAPCREATE(s1)
li a0, CAP_TYPE_LIN
lla a3, result
lla a4, result_end
li a5, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)

# a2: the stub's code capability; a1 -> the seal
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal
lla a4, seal_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)

li t0, QUANTUM
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0
li s7, 0                       # the number of resumes
li x30, 0xA5
li t0, 0x888
csrw mie, t0                   # the monitor's mie: id 7 lives in the seal's writable window during the domain's run

CSSUPERVISE(t3, a3, s5)
CALL(a3, a3)
supervise_loop:
  csrr t0, CSUPSTATUS
  CAPPRINT(t0)                 # 3 while preempted, 1 on the RETURN
  andi t1, t0, 6
  li t2, 2
  bne t1, t2, done             # kind != 1: the return (0) or a fault (2) -- stop either way
  csrr t0, CSUPCAUSE
  CAPPRINT(t0)                 # 0x8000000000000010
  addi s7, s7, 1
  csrwi CSUPCTL, 1             # resume
  CSSUPERVISE(t3, a3, s5)
  CALL(a3, a3)
  j supervise_loop
done:
ld t0, 0(s1)
CAPPRINT(t0)                   # 0x190: every addi ran exactly once
ld t0, 8(s1)
CAPPRINT(t0)                   # 0x1234: the domain's live x1 survived every resume (not a fresh SEALEDRET)
addi s2, s7, 0
CAPPRINT(s2)                   # N, the number of resumes (>= 3 with QUANTUM 64)
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
csrr t0, mie
CAPPRINT(t0)                   # 0x888: the monitor's mie came back from the private area every time
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub:
  MOVC(s3, x1)                 # park the SEALEDRET; x1 becomes a plain return-address sentinel for the ladder
  li x1, 0x1234
  li x20, 0
  li x5, 0xBAD5
  .rept 400
  addi x20, x20, 1
  .endr
  sd x20, 0(s1)
  sd x1, 8(s1)                 # 0x1234 if every resume left x1 alone (after-audit 3C: it used to be re-sealed)
  MOVC(x1, s3)
  la a1, stub
  RETURN(ra, a1, x0)
stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
seal:
  .zero 2048
seal_end:
.align 4
result:
  .zero 64
result_end:
  TEST_DATA
RVTEST_DATA_END
