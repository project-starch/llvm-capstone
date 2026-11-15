# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a
# parenthesis is a call).
#
#*****************************************************************************
# tag-content.S -- after plain overwrites of tagged memory, are the DRAM tag bytes right? (twc-reach plan, 2026-11-15)
#-----------------------------------------------------------------------------
#
# The adapter frees a store's tag write when the TWC holds its word and the cached byte already equals the store's
# tag. A stale entry that reads 0 over a DRAM 1 would let a plain store skip its clear and leave a forgeable tag that
# no output check sees while the entry lives. This test reads the tag bytes back RAW, with plain byte loads from the
# tag region (tag address = 0xBC2D2D2D + ((paddr - 0x80000000) >> 4)), after a fence, a delay and a D$ flush, and
# compares each against the value its store sequence implies. Not capmode: stc / ldc take the capability, plain
# loads and stores integer addresses.
#
# Four pages P0..P3 (1,024 granules). Every granule of P0..P2 gets a capability (stc), then a load of every granule
# installs P0..P2's tag words in the TWC. Then, with the words cached:
#   C1 P0: plain sd over every even granule (cached byte 1: the tag write must happen);
#   C2 P0: the same stores again (cached byte 0: a legitimate skip);
#   C3 P1: per pair, stc over the odd granule (cached 1: legitimate skip) and plain sd over the even one (clear);
#   C4 P1: per pair, stc over the even granule (0 -> 1) and plain sd over the odd one (1 -> 0): tag writes in both
#          directions to the same words, back to back;
#   C5 P2: per granule, stc then plain sd over the same granule (merge, then the clear of the merged byte);
#   C6 P3 (unmarked until now; its first tag word is shared with P2's last): load P2's last granule (its word is
#          cached), stc into P3 granule 1 (the page's first tagged store: TAG_FILL), plain sd over P3 granule 0, load
#          P3 granule 2 (installs a P3 word), stc then plain sd over P3 granule 3.
# Expected tag bytes: P0 odd 1 / even 0; P1 even 1 / odd 0; P2 0; P3 granule 1 is 1, every other 0 (the fill zeroed
# the page). Ones expected: 128 + 128 + 0 + 1 = 257.
# PRINTS (CAPPRINT, in order): ok (1024), bad (0), first bad granule index (0xffff when none), its byte (0),
# ones read correctly (257, the positive control that the raw path sees tags), page fills during the test (4 on a
# fresh boot: P0..P3's first tagged stores; P3's arm is a fill only if this reads 4), traps (0). The checks run in
# integer mode; the test enters capmode only for the prints and the capability exit (CAP_PASS).
# POSITIVE CONTROL in simulation: +define+TAG_AUDIT_PC (a store skips on ANY TWC hit) must read bad > 0 (C1's 128
# clears skipped at least).
#
#*****************************************************************************

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s11, 0                   # trap count
  li   a5, 0x5a5a5a5a

  li   t0, 27
  csrw 0x323, t0                # mhpmevent3 = page fill started
  csrr s8, 0xB03                # fills before

  CAPCREATE(s2)                 # the area: NONLIN, all permissions, cursor at its base
  li   t0, CAP_TYPE_NONLIN
  lla  a2, tc_area
  lla  a3, tc_end
  CAPTYPE(s2, t0)
  CAPBOUND(s2, a2, a3)
  li   t0, CAP_PERM_ALL
  CAPPERM(s2, t0)

  # A: a capability in every granule of P0..P2 (768 stc)
  MOVC(a1, s2)
  li   t2, 768
  li   t6, 16
1:
  STC(a1, s2, 0)
  CINCOFFSET(a1, a1, t6)
  addi t2, t2, -1
  bnez t2, 1b
  fence

  # B: flush the L1D, then load every granule of P0..P2 (the TWC installs their tag words)
  csrw 0x7c1, x0
  li   t3, 1
  csrw 0x7c1, t3
  lla  t0, tc_area
  li   t2, 768
2:
  ld   t3, 0(t0)
  addi t0, t0, 16
  addi t2, t2, -1
  bnez t2, 2b

  # C1, C2: P0 even granules, plain, twice
  li   t4, 2
3:
  lla  t0, tc_area
  li   t2, 128
4:
  sd   a5, 0(t0)
  addi t0, t0, 32
  addi t2, t2, -1
  bnez t2, 4b
  addi t4, t4, -1
  bnez t4, 3b

  # C3: P1, stc over the odd granule, plain sd over the even one
  lla  t0, tc_area
  li   t1, 4096
  add  t0, t0, t1               # P1, integer
  li   t1, 4096 + 16
  CINCOFFSET(a1, s2, t1)        # P1 granule 1, capability
  li   t6, 32
  li   t2, 128
5:
  STC(a1, s2, 0)
  sd   a5, 0(t0)
  CINCOFFSET(a1, a1, t6)
  addi t0, t0, 32
  addi t2, t2, -1
  bnez t2, 5b

  # C4: P1, stc over the even granule (0 -> 1), plain sd over the odd one (1 -> 0)
  lla  t0, tc_area
  li   t1, 4096 + 16
  add  t0, t0, t1               # P1 granule 1, integer
  li   t1, 4096
  CINCOFFSET(a1, s2, t1)        # P1 granule 0, capability
  li   t2, 128
6:
  STC(a1, s2, 0)
  sd   a5, 0(t0)
  CINCOFFSET(a1, a1, t6)
  addi t0, t0, 32
  addi t2, t2, -1
  bnez t2, 6b

  # C5: P2, stc then plain sd over the same granule
  lla  t0, tc_area
  li   t1, 8192
  add  t0, t0, t1
  CINCOFFSET(a1, s2, t1)
  li   t6, 16
  li   t2, 256
7:
  STC(a1, s2, 0)
  sd   a5, 0(t0)
  CINCOFFSET(a1, a1, t6)
  addi t0, t0, 16
  addi t2, t2, -1
  bnez t2, 7b

  # C6: P3, the fill next to a cached shared word
  lla  t0, tc_area
  li   t1, 12288 - 16
  add  t0, t0, t1
  ld   t3, 0(t0)                # P2's last granule: its tag word (shared with P3's first bytes) is cached
  li   t1, 12288 + 16
  CINCOFFSET(a1, s2, t1)
  STC(a1, s2, 0)                # P3 granule 1: the page's first tagged store (TAG_FILL), tag 1
  lla  t0, tc_area
  li   t1, 12288
  add  t0, t0, t1
  sd   a5, 0(t0)                # P3 granule 0: plain
  ld   t3, 32(t0)               # P3 granule 2: installs a P3 word
  li   t1, 12288 + 48
  CINCOFFSET(a1, s2, t1)
  STC(a1, s2, 0)                # P3 granule 3: tag 1 ...
  sd   a5, 48(t0)               # ... then plain: 0

  # D: drain, delay, flush, then read the 1,024 tag bytes raw
  fence
  li   t2, 20000
8:
  addi t2, t2, -1
  bnez t2, 8b
  csrr s9, 0xB03
  sub  s9, s9, s8               # page fills during the test
  csrw 0x7c1, x0
  li   t3, 1
  csrw 0x7c1, t3
  lla  t0, tc_area
  li   t1, 0x80000000
  sub  t0, t0, t1
  srli t0, t0, 4
  li   t1, 0xBC2D2D2D
  add  t0, t0, t1               # tag byte of granule 0
  li   s3, 0                    # ok
  li   s4, 0                    # bad
  li   s5, 0xffff               # first bad granule
  li   s6, 0                    # its byte
  li   s7, 0                    # ones read correctly
  li   t2, 0                    # granule index g
9:
  lbu  t3, 0(t0)
  srli t4, t2, 8                # page
  andi t5, t2, 255              # index in page
  andi t1, t5, 1                # P0: odd -> 1
  li   t6, 1
  bne  t4, t6, 10f
  xori t1, t1, 1                # P1: even -> 1
10:
  li   t6, 2
  bne  t4, t6, 11f
  li   t1, 0                    # P2: 0
11:
  li   t6, 3
  bne  t4, t6, 12f
  addi t1, t5, -1
  seqz t1, t1                   # P3: granule 1 -> 1
12:
  bne  t3, t1, 13f
  addi s3, s3, 1
  add  s7, s7, t1
  j    14f
13:
  bnez s4, 15f
  mv   s5, t2
  mv   s6, t3
15:
  addi s4, s4, 1
14:
  addi t0, t0, 1
  addi t2, t2, 1
  li   t6, 1024
  bne  t2, t6, 9b

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)              # capmode for the prints: the board recorder writes through a capability
  CAPPRINT(s3)                  # ok: 1024
  CAPPRINT(s4)                  # bad: 0
  CAPPRINT(s5)                  # first bad granule: 0xffff
  CAPPRINT(s6)                  # its byte: 0
  CAPPRINT(s7)                  # ones read correctly: 257
  CAPPRINT(s9)                  # page fills: 4
  CAPPRINT(s11)                 # traps: 0
  CAP_PASS(s10)

.align 4
trap_handler:
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
tc_area:
  .zero 16384
tc_end:

  TEST_DATA

RVTEST_DATA_END
