# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# csmint.S -- B5b-1 step 3: CSMINT rd, rs1 = slot, rs2 = desc through the LSU's sequencer, on the physical pool (2026-11-23)
#-----------------------------------------------------------------------------
#
# Integer M (no capenter), so the record goes to the physical pool (header at 0xbff00000: capacity 65,536, RECYCLE,
# next 3; slot 2 is the debug mints' root), as sublet-life. x26 = capcreate NONLIN RW over area, which holds the slots,
# so an ldc can read a minted granule back. Each arm: the descriptor at desc, the slot(s) pre-filled with 0x5a5a...,
# csmint, then the trap cause (0 = none; the handler skips the instruction) and rd. A faulting or refused arm also
# prints mtval and the slot's first word (it must still be the fill: nothing stored).
#   a1  a valid mint, one slot                      0, rd, slot lo, slot hi
#   a2  a valid mint with slot2                     0, rd, slot2 lo, slot2 hi (= slot's, the same granule)
#   a3  desc & 7 != 0                               4, tval desc
#   a4  slot & 15 != 0                              6, tval slot
#   a5  slot2 & 15 != 0                             6, tval slot2; slot unchanged
#   a6  base >= end                                 29, tval 0
#   a7  perms 0xf (NONLIN missing)                  29
#   a8  reserved word 1                             29
#   a9  cursor = end + 1                            29 (the cursor's edge, past)
#   a10 cursor >= 2^38                              29
#   a11 base one byte off the granule (len 8 KiB)   29 (exactness at the base edge, past)
#   a12 the same region on the granule              0, rd (exactness at the base edge, representable)
#   a13 end one byte off the granule                29 (the end edge, past)
#   a14 cursor = end                                0, rd (the cursor's edge, representable)
#   a15 rs1 = sp                                    2 (illegal instruction)
#   a16 ldc the a1 granule back; cderive from it    0; crevoke it 0; cderive again 25 (the record is a1's, live, then dead)
#   a17 csmint, then an immediately dependent add   0, rd + 0 (the writeback is rd, never a descriptor word)
#   a18 perms and cursor are a stored capability's  29, tval 0, slot unchanged (the descriptor is read RAW)
#   p1..p8 the pool's header and records 3..5 by plain loads
# PRE-REGISTERED: csmint.expect, computed BEFORE any simulation by the reference's own code (capstone-qemu c1985bb529:
# the table code through sublet-life/harness.c for every id and record, the codec through csmint-exact/ for every stored
# word) on this arm list; the causes from helper_csmint's order. Register map: cause s1, tval s3, trap count a7.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2
#define CHK  mv t2, s1; CAPPRINT(t2); li s1, 0
#define TV   mv t2, s3; CAPPRINT(t2); li s3, 0
#define PR(r) mv t2, r; CAPPRINT(t2)
#define FILL(off) li t0, 0x5a5a5a5a5a5a5a5a; lla t1, area; sd t0, off(t1); sd t0, off+8(t1)
#define FIRST(off) lla t1, area; ld t2, off(t1); CAPPRINT(t2)
/* the descriptor {base, end, perms, cursor, slot2, reserved}, each a register or an immediate loaded into t0 */
#define DW(i, v) li t0, v; lla t1, desc; sd t0, (8*i)(t1)
#define DR(i, r) lla t1, desc; sd r, (8*i)(t1)

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0
li s1, 0
li s3, 0

# x26: NONLIN RW over [desc, area_end) (the descriptor and the slots), cursor desc
CAPCREATE(x26)
li a0, CAP_TYPE_NONLIN
CAPTYPE(x26, a0)
lla a3, desc
lla a4, area_end
CAPBOUND(x26, a3, a4)
li a5, CAP_PERM_RW
CAPPERM(x26, a5)
SCC(x26, x26, a3)

# the object every valid arm mints: obj, 256 bytes (exact: below 4 KiB), cursor obj + 16
lla s4, obj
addi s5, s4, 256
addi s6, s4, 16
lla s7, desc
lla s8, area                   # slot 0 at area + 0
addi s9, s8, 32                # slot 1 at area + 32

#define VALID DR(0, s4); DR(1, s5); DW(2, 0x10f); DR(3, s6); DW(4, 0); DW(5, 0)

# a1: valid, one slot
FILL(0)
VALID
CSMINT(a0, s8, s7)
CHK
PR(a0)
FIRST(0)
lla t1, area; ld t2, 8(t1); CAPPRINT(t2)

# a2: valid, slot2 = slot 1 (slot 0 is overwritten again)
FILL(0)
FILL(32)
VALID
DR(4, s9)
CSMINT(a0, s8, s7)
CHK
PR(a0)
FIRST(32)
lla t1, area; ld t2, 40(t1); CAPPRINT(t2)

# a3: desc misaligned
VALID
addi t3, s7, 4
CSMINT(a0, s8, t3)
CHK
TV

# a4: slot misaligned
FILL(0)
addi t3, s8, 8
CSMINT(a0, t3, s7)
CHK
TV
FIRST(0)

# a5: slot2 misaligned: 6 with tval slot2, and slot 0 unchanged
FILL(0)
addi t3, s9, 8
DR(4, t3)
CSMINT(a0, s8, s7)
CHK
TV
FIRST(0)

# a6..a10: the value refusals, each 29 with tval 0 and slot 0 unchanged
FILL(0)
VALID
DR(1, s4)                      # end = base
CSMINT(a0, s8, s7)
CHK
TV
VALID
DW(2, 0xf)
CSMINT(a0, s8, s7)
CHK
VALID
DW(5, 1)
CSMINT(a0, s8, s7)
CHK
VALID
addi t3, s5, 1
DR(3, t3)                      # cursor = end + 1
CSMINT(a0, s8, s7)
CHK
VALID
li t3, 0x4000000000            # 2^38
DR(3, t3)
li t3, 0x4000000100
DR(1, t3)
CSMINT(a0, s8, s7)
CHK
FIRST(0)

# a11..a14: the representability edges. An 8 KiB region has E = 1 and a granule of 16 bytes.
VALID
li t3, 0x10000001              # base one byte off the granule
DR(0, t3)
li t3, 0x10002001
DR(1, t3)
DR(3, t3)
CSMINT(a0, s8, s7)
CHK
FILL(0)
VALID
li t3, 0x10000000              # on the granule
DR(0, t3)
DR(3, t3)
li t3, 0x10002000
DR(1, t3)
CSMINT(a0, s8, s7)
CHK
PR(a0)
VALID
li t3, 0x10000000
DR(0, t3)
DR(3, t3)
li t3, 0x10002001              # end one byte off the granule
DR(1, t3)
CSMINT(a0, s8, s7)
CHK
FILL(0)
VALID
DR(3, s5)                      # cursor = end
CSMINT(a0, s8, s7)
CHK
PR(a0)

# a15: rs1 = sp
VALID
CSMINT(a0, x2, s7)
CHK

# a16: the a1 granule is re-minted into slot 0 by a16's own csmint, read back by ldc, used, revoked
FILL(0)
VALID
CSMINT(a0, s8, s7)
CHK
PR(a0)
LDC(x18, x26, 0x40)          # slot 0 = area = desc + 0x40. x18 IS s2 and x16 IS a6, both unused elsewhere: the first cut
li a1, 16                    # used x20 / x21, which ARE s4 / s5 (the descriptor's base and end), so a17 read base = end
CDERIVE(x16, x18, a1)
CHK
CREVOKE(x18, x0)
CHK
CDERIVE(x16, x18, a1)
CHK

# a17: a dependent read right behind csmint
VALID
CSMINT(a0, s9, s7)
add t2, a0, x0
CAPPRINT(t2)
CHK

# a18: the descriptor's perms and cursor words are a STORED CAPABILITY (x27 over [0x100, 0x200), cursor 0x10f).
# helper_csmint reads them raw (cpu_ldq): perms is the encoded first word, not 0x10f, so 29. A plain ld's projection
# (the address, 0x10f) would pass every refusal and mint: the arm tells the raw descriptor load from LD.
DW(0, 0)
DW(1, 0x1000)
DW(4, 0)
DW(5, 0)
CAPCREATE(x27)
li a0, CAP_TYPE_NONLIN
CAPTYPE(x27, a0)
li a3, 0x100
li a4, 0x200
CAPBOUND(x27, a3, a4)
li a5, CAP_PERM_RW
CAPPERM(x27, a5)
li t0, 0x10f
SCC(x27, x27, t0)
STC(x26, x27, 16)             # desc + 16: perms (first doubleword), cursor (second)
FILL(0)
CSMINT(a0, s8, s7)
CHK
TV
FIRST(0)

# the pool, by plain loads: header quadwords 0 and 1, records 3..5
li t3, 0xbff00000
ld t2, 0(t3); CAPPRINT(t2)
ld t2, 8(t3); CAPPRINT(t2)
ld t2, 16(t3); CAPPRINT(t2)
ld t2, 24(t3); CAPPRINT(t2)
li t4, 3
1:
slli t5, t4, 4
add t5, t5, t3
ld t2, 0(t5); CAPPRINT(t2)
ld t2, 8(t5); CAPPRINT(t2)
addi t4, t4, 1
li t5, 6
bne t4, t5, 1b

CAPPRINT(a7)
CAP_PASS(s10)

.align 4
trap_handler:
addi a7, a7, 1
csrr s1, mcause
csrr s3, mtval
csrr t6, mepc
addi t6, t6, 4
csrw mepc, t6
mret

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
.align 8
obj:
.zero 256
.align 6
desc:
.zero 64
.align 6
area:
.zero 64
area_end:
RVTEST_DATA_END
