# See LICENSE for license details.

#*****************************************************************************
# sup-guards.S  (supervised CALL on silicon, verification step 6: the guards)
#-----------------------------------------------------------------------------
#
# What a supervised domain may not do, one arm each, every arm its own seal and stub: the instruction must
# escape to the monitor as a fault event (kind 2, cause 2 = illegal instruction) with nothing landed.
#   0 sret   1 wfi   2 `csrw mepc` (mepc is NOT exchanged: the monitor reads its own 0 back afterwards)
#   3 `csrw mie`   4 CCSRRW cpmp0   5 cssupervise   6 CALL of its own seal   7 CAPENTER   8 CAPCREATE
#   9 `csrr csupstatus`  10 a LEGAL control: `csrr cycle`, CCSRRW ctvec, then RETURN -> kind 0 (status 1)
#   11: a LEGAL `csrr cycle` followed by a spin: the quantum must still end it (csupstatus 3, measured)
#   12: `csrr fcsr` followed by a spin: with FS = Off in the image the read is itself illegal, so this one escapes
#       as a fault (csupstatus 5, cause 2, measured) -- not a quantum; kept as the record of that
#   13: CCSRRW cpmp0 with a tagged capability (blocked): csupstatus 5, and the monitor reads CPMP0 back EMPTY (LCC 7)
#   14: CCSRRW cepc with a capability (legal), then a spin: csupstatus 3; the monitor's mepc reads 0 and its cepc EMPTY
# Pre-registered readings per illegal arm: status 0, csupstatus 5, csupcause 2, csupepc - fault_k = 0,
# lcc (a3) = 4, mcause 0, and after arm 2: mepc 0. Arm 10: status 0, csupstatus 1 (the RETURN), mcause 0.
# Mutants (positive controls): no-csr-gate -> arm 2 reads csupstatus 1 and mepc 0x1234; no-mint-guard ->
# arm 8 reads csupstatus 1; no-mret-guard is step 3's.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw mepc, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0

# ---------------------------------------------------------------- arm 0: sret
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
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault0
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 1: wfi
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
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault1
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 2: csrw mepc (not exchanged)
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
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault2
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0
csrr t0, mepc
CAPPRINT(t0)                   # 0: the write never landed

# ---------------------------------------------------------------- arm 3: csrw mie
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
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault3
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 4: CCSRRW cpmp0
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
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault4
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 5: cssupervise under supervision
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub5
lla a4, stub5_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal5
lla a4, seal5_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault5
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 6: CALL under supervision (its own SEALEDRET in x1 is not a seal; the decoder refuses first)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub6
lla a4, stub6_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal6
lla a4, seal6_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault6
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 7: CAPENTER
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub7
lla a4, stub7_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal7
lla a4, seal7_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault7
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 8: capcreate (a mint)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub8
lla a4, stub8_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal8
lla a4, seal8_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault8
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 9: csrr csupstatus
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub9
lla a4, stub9_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal9
lla a4, seal9_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 2
csrr t0, CSUPEPC
la t1, fault9
sub t0, t0, t1
CAPPRINT(t0)                   # 0
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 10: LEGAL control: csrr cycle, CCSRRW ctvec, RETURN
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub10
lla a4, stub10_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal10
lla a4, seal10_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 11: LEGAL csrr cycle, then a spin: the quantum must still end it
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub11
lla a4, stub11_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal11
lla a4, seal11_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3 (the quantum, after a legal CSR read)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 12: csrr fcsr (illegal with FS = Off), then a spin
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub12
lla a4, stub12_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal12
lla a4, seal12_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 (a fault: fcsr is illegal while FS = Off)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 13: CCSRRW cpmp0 with a TAGGED capability (blocked): the monitor must read CPMP0 back EMPTY (after-audit 4A)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub13
lla a4, stub13_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal13
lla a4, seal13_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 (fault: CCSRRW cpmp0 is blocked)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0
CCSRRW(t0, 0x10, x0)           # read (and clear) CPMP0
LCC(s3, t0, x1)
CAPPRINT(s3)                   # 7: nothing landed (a LINEAR capability would read 0)

# ---------------------------------------------------------------- arm 14: CCSRRW cepc with a capability (LEGAL), then a spin: mepc/cepc are walked as id 25 (after-audit 4B)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub14
lla a4, stub14_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal14
lla a4, seal14_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
csrw mcause, x0
csrw mepc, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3 (the quantum)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # 4
csrr t0, mcause
CAPPRINT(t0)                   # 0
csrr t0, mepc
CAPPRINT(t0)                   # 0: the monitor's own mepc came back from the private area
CCSRRW(t0, CCSR_CEPC, x0)      # read (and clear) CEPC
LCC(s3, t0, x1)
CAPPRINT(s3)                   # 7: the monitor's own (empty) cepc, not the domain's capability

CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub0:
fault0:
  sret
  li x20, 0xDEAD
  1: j 1b
stub0_end:

.align 4
stub1:
fault1:
  wfi
  li x20, 0xDEAD
  1: j 1b
stub1_end:

.align 4
stub2:
fault2:
  li t0, 0x1234
  csrw mepc, t0
  li x20, 0xDEAD
  1: j 1b
stub2_end:

.align 4
stub3:
fault3:
  csrw mie, x0
  li x20, 0xDEAD
  1: j 1b
stub3_end:

.align 4
stub4:
fault4:
  CCSRRW(t0, 0x10, x0)         # CCSR_CPMP0 (asm_insn.h names only CTVEC and CIH)
  li x20, 0xDEAD
  1: j 1b
stub4_end:

.align 4
stub5:
fault5:
  CSSUPERVISE(t0, x0, x0)
  li x20, 0xDEAD
  1: j 1b
stub5_end:

.align 4
stub6:
fault6:
  CALL(x1, x1)
  li x20, 0xDEAD
  1: j 1b
stub6_end:

.align 4
stub7:
fault7:
  CAPENTER(a0, a1)
  li x20, 0xDEAD
  1: j 1b
stub7_end:

.align 4
stub8:
fault8:
  CAPCREATE(t0)
  li x20, 0xDEAD
  1: j 1b
stub8_end:

.align 4
stub9:
fault9:
  csrr t0, CSUPSTATUS
  li x20, 0xDEAD
  1: j 1b
stub9_end:

.align 4
stub10:
fault10:
  csrr t0, cycle
  CCSRRW(t0, CCSR_CTVEC, x0)
  la a1, stub10
  RETURN(ra, a1, x0)
  li x20, 0xDEAD
  1: j 1b
stub10_end:

.align 4
stub11:
fault11:
  csrr t0, cycle
  li x20, 0xDEAD
  1: j 1b
stub11_end:

.align 4
stub12:
fault12:
  csrr t0, fcsr
  li x20, 0xDEAD
  1: j 1b
stub12_end:

.align 4
stub13:
fault13:
  CCSRRW(t0, 0x10, s5)         # a tagged LINEAR capability (the monitor's save area) into CPMP0: blocked
  li x20, 0xDEAD
  1: j 1b
stub13_end:

.align 4
stub14:
fault14:
  CCSRRW(t0, CCSR_CEPC, s5)    # legal: cepc := the capability, mepc := its cursor -- the domain's own now
  li x20, 0xDEAD
  1: j 1b
stub14_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
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
.align 6
seal5:
  .zero 2048
seal5_end:
.align 6
seal6:
  .zero 2048
seal6_end:
.align 6
seal7:
  .zero 2048
seal7_end:
.align 6
seal8:
  .zero 2048
seal8_end:
.align 6
seal9:
  .zero 2048
seal9_end:
.align 6
seal10:
  .zero 2048
seal10_end:
.align 6
seal11:
  .zero 2048
seal11_end:
.align 6
seal12:
  .zero 2048
seal12_end:
.align 6
seal13:
  .zero 2048
seal13_end:
.align 6
seal14:
  .zero 2048
seal14_end:
  TEST_DATA
RVTEST_DATA_END
