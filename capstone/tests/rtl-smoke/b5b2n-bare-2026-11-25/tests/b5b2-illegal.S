# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-illegal.S -- B5b-2 step 1's positive control: every capability instruction the reference deleted (2026-11-24)
#-----------------------------------------------------------------------------
#
# The port of capstone-qemu 9289114e06 tests/sublet-format/illegal.S. The reference deleted these encodings between
# c1985bb529 and 9289114e06 (insn32.decode); each must raise cause 2 in M, with mtval = the instruction bits. The
# reference runs one arm per build and exits through semihosting; here every site runs in one build, in M, with
# untagged operands, and the handler records the trap and resumes after the site. Per site, three prints:
#   cause   mcause, or 0xff if the site retired without a trap
#   at      1 if mepc was the site, 0 if the trap came from another pc, 2 if there was no trap
#   tval    1 if mtval equals the site's instruction word, else 0
# Registers, both spellings: s1/x9 cause, s3/x19 tval, s4/x20 at, s2/x18 the site's address (set before each site, read
# by the handler); a7/x17 the trap count; t0/x5 (the handler) and t1/x6 (the prints) are temporaries only.
#
# Sites, in order (the reference's funct7 under custom-2 0x5b funct3 1 unless noted):
#    1 revoke 0x00     2 tighten 0x02    3 delin 0x03      4 split 0x06      5 seal 0x07       6 mrev 0x08
#    7 init 0x09       8 drop 0x0b       9 capenter 0x0d  10 call 0x20      11 return 0x21    12 supervise 0x22
#   13 supervisor_gc 0x23                14 csretire 0x53  15 gencap 0x40    16 capmem 0x41    17 clearcmmap 0x42
#   18 cjalr (funct3 5, rs1 = the address of cj_land, so a retired cjalr lands there and comes back)
#   19-23 the CVA6-only debug mints under custom-3 0x7b funct3 0: capcreate 0x04, captype 0x05, capnode 0x06,
#         capperm 0x07, capbound 0x08 (the reference has no custom-3; its debug mint gencap is site 15)
#   24 ccsrrw ctvec, rs1 = x0 (funct3 7)
#   25 capprint a0 (custom-3 funct7 9, its raw encoding: the board wrapper redefines the CAPPRINT macro)
#   then the trap count
# csrunv is NOT here: from M with a frame address it enters a context when legal, and the run would not come back. It is
# its own arm (b5b2-illegal-runv, -DARM_RUNV), run alone.
#
# PRE-REGISTERED (2026-11-24, written before any run), on the Sublet build (no CAPSTONE_IDS55_FMT):
#   b5b1-step3 (27e75c38a, the base):
#     sites 1-12 and 14: 2 / 1 / 1 (B5b-1 makes them illegal before any other check, decoder.sv :1212-1456)
#     sites 13, 15, 16, 17: 2 / 1 / 1 (no arm for those funct7: the custom-2 default, illegal at decode, :1467)
#     sites 18-24: NOT 2 (still decoded; their exact readings are recorded, not predicted)
#   step 1's first cut (cjalr deleted under the Sublet format): as the base, except site 18 = 2 / 1 / 1.
#   Sites 19-24 move only when their deletion is ruled (the custom-3 mints and ccsrrw wait on the semantics owner and on
#   step 4). The trap count is the number of sites that trapped.
#   arm RUNV: the base NOT 2; after csrunv's deletion 2 / 1 / 1.
# CCSRRW's deletion (pre-registered 2026-11-24 before its run, on b5b2-step4 eb03e7c99 + the decoder term): as step 1's
#   reading, except site 24 = 2 / 1 / 1 and the trap count 0x13 (19). On eb03e7c99 itself site 24 retires (0xff / 2 / 0)
#   and the count is 0x12, as read there.
#   The other tests that execute ccsrrw (all inert under Sublet: a ccsr write needs capmode), pre-registered the same
#   way: sublet-ns.S and every arm built from it (-f4, -edge, -edgem, -wide, -snapu, -ctxint, -ctxoob) and va-runv-snap
#   trap 2 at their one ccsrrw (CPMP 0); their handlers print 2 and skip it (t0 and t1 are rewritten right after), so
#   each reads exactly one more 0x2 and every other reading unchanged in value and order. r79-cpmp (built here without
#   its IDS55 define, so not its own configuration) traps at its four: its trap count a7 reads 0x5 for 0x1, the rest
#   unchanged.

# STEP 8a (pre-registered 2026-11-25, written before any run; base b5b2-ccsrrw2 7997b51a2, 8a = b5b2-step8a): CSRUNV and
#   the supervisor CSRs are deleted under the Sublet format (capstone-qemu 35de150cef; the node census 0xFC4 was never in
#   the reference).
#   arm RUNV: base 0xff / 2 / 0 and count 0x0 (it retires, as read on silicon in b5b2-bare-2026-11-24); 8a 2 / 1 / 1 and
#     count 0x1.
#   arm CSUP (-DARM_CSUP), in M, three prints per site as above, then the trap count. Every supervisor CSR is M-level, so
#   an S access raises 2 on either tree (privilege); the arm runs in M only, where the deletion is visible:
#     sites 1-7   csrr of 0x7C3, 0x7C4, 0xFC0, 0xFC1, 0xFC2, 0xFC3, 0xFC4: base 0xff / 2 / 0 each; 8a 2 / 1 / 1 each
#     sites 8-9   csrw of 0x7C3, 0x7C4 (rs1 = x0): base 0xff / 2 / 0; 8a 2 / 1 / 1
#     sites 10-14 csrw of 0xFC0-0xFC4 (read-only addresses): 2 / 1 / 1 on BOTH trees (the control that a write to a
#                 read-only CSR traps either way)
#     site 15     csrr of srevroot (0x5C1): 0xff / 2 / 0 on BOTH trees (the control that a surviving CSR retires)
#     trap count: base 0x5, 8a 0xe
# STEP 9 (pre-registered 2026-11-25, written before any run; capstone-ariane b5b2-step9): the custom-3 debug family
#   (capcreate, captype, capnode, capperm, capbound, capprint) is illegal under the Sublet format unless the
#   simulation-only scaffold CAPSTONE_SIM_DEBUG_MINT is defined (the legacy tests' build). Site 25 is added.
#   With the scaffold (the in-tree runs, every legacy test's build): every site as before; site 25 retires
#     0xff / 2 / 0, and, being a real capprint there, capprint-readings also lists the site itself (a0 = 0x1000) just
#     ahead of those three prints; the count 0x13.
#   Without it (the bare kit's image, recorded by plain stores, simulated without the define, and on silicon):
#     sites 19-23 and 25 move to 2 / 1 / 1; the count 0x19.
#   The scaffold is the only difference between the two builds: the arms read the old values with it.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define I_R(f7)    .insn r 0x5B, 0x1, f7, a2, a0, a1
#define I_R3(f7)   .insn r 0x7B, 0x0, f7, a2, a0, a1
#define I_CJALR    .insn i 0x5B, 0x5, a2, a0, 0
#define I_CCSRRW   .insn i 0x5B, 0x7, a2, zero, 0
#define I_RUNV     .insn r 0x5B, 0x1, 0x24, a2, a0, zero
#define I_CSRR(c)  csrrs a2, c, zero
#define I_CSRW(c)  csrrw zero, c, zero

#define SITE(lab, insn)                                    \
  li s1, 0xff;  li s3, 0;  li s4, 2;  lla s2, lab;         \
lab:                                                       \
  insn;                                                    \
  CAPPRINT(s1);  CAPPRINT(s4);                             \
  lwu t1, 0(s2);  sub t1, s3, t1;  seqz t1, t1;  CAPPRINT(t1)

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0
li a0, 0x1000
li a1, 0x2000
li a2, 0

#ifdef ARM_RUNV
SITE(s_runv, I_RUNV)
#elif defined(ARM_CSUP)
SITE(c_r7c3, I_CSRR(0x7c3))
SITE(c_r7c4, I_CSRR(0x7c4))
SITE(c_rfc0, I_CSRR(0xfc0))
SITE(c_rfc1, I_CSRR(0xfc1))
SITE(c_rfc2, I_CSRR(0xfc2))
SITE(c_rfc3, I_CSRR(0xfc3))
SITE(c_rfc4, I_CSRR(0xfc4))
SITE(c_w7c3, I_CSRW(0x7c3))
SITE(c_w7c4, I_CSRW(0x7c4))
SITE(c_wfc0, I_CSRW(0xfc0))
SITE(c_wfc1, I_CSRW(0xfc1))
SITE(c_wfc2, I_CSRW(0xfc2))
SITE(c_wfc3, I_CSRW(0xfc3))
SITE(c_wfc4, I_CSRW(0xfc4))
SITE(c_rsrev, I_CSRR(0x5c1))
#else
SITE(s_revoke, I_R(0x00))
SITE(s_tighten, I_R(0x02))
SITE(s_delin, I_R(0x03))
SITE(s_split, I_R(0x06))
SITE(s_seal, I_R(0x07))
SITE(s_mrev, I_R(0x08))
SITE(s_init, I_R(0x09))
SITE(s_drop, I_R(0x0b))
SITE(s_capenter, I_R(0x0d))
SITE(s_call, I_R(0x20))
SITE(s_return, I_R(0x21))
SITE(s_supervise, I_R(0x22))
SITE(s_supgc, I_R(0x23))
SITE(s_csretire, I_R(0x53))
SITE(s_gencap, I_R(0x40))
SITE(s_capmem, I_R(0x41))
SITE(s_clearcm, I_R(0x42))
lla a0, cj_land
SITE(s_cjalr, I_CJALR)
li a0, 0x1000
SITE(s_capcreate, I_R3(0x04))
SITE(s_captype, I_R3(0x05))
SITE(s_capnode, I_R3(0x06))
SITE(s_capperm, I_R3(0x07))
SITE(s_capbound, I_R3(0x08))
SITE(s_ccsrrw, I_CCSRRW)
SITE(s_capprint, I_R3(0x09))
#endif
CAPPRINT(a7)
RVTEST_PASS

# a retired cjalr jumps here: mark it 0xfe and return to the instruction after the site
.align 4
cj_land:
  li s1, 0xfe
  addi t0, s2, 4
  jr t0

.align 4
trap_handler:
  addi a7, a7, 1
  csrr s1, mcause
  csrr s3, mtval
  csrr t0, mepc
  sub t0, t0, s2
  seqz s4, t0
  addi t0, s2, 4
  csrw mepc, t0
  mret

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
