# See LICENSE for license details.

#*****************************************************************************
# niss-superpage.S
#-----------------------------------------------------------------------------
#
# DOES A LOAD THROUGH A 2 MiB PAGE TAKE THE NON-IDEMPOTENT STALL? (2026-11-06, the runtime lane's question for mcf on
# transparent huge pages.)
#
# THE SUSPECTED DEFECT, by source (cva6_mmu.sv, identical at e7af48457, 1fc2cd5aa and c9c7952bd; carried over from upstream
# CVA6). lsu_dtlb_ppn_o is a PPN vector: its bit i is paddr bit i+12. On a superpage hit the MMU overwrites part of it
# with virtual-address bits, using PADDR index ranges:
#   2 MiB: lsu_dtlb_ppn_o[20:12] = lsu_vaddr_n[20:12]   (should be ppn[8:0])
#   1 GiB: lsu_dtlb_ppn_o[29:12] = lsu_vaddr_n[29:12]   (should be ppn[17:0])
# so paddr[32:24] (or [41:24]) of the address the load unit checks is replaced by vaddr[20:12] (or [29:12]), and
# ppn[8:0] keeps the PTE's zeros. The only consumer is load_unit's non-idempotent check, against the FPGA's one rule
# [0, 0x8000_0000) (ariane_xilinx.sv build_fpga_config). With DRAM below 4 GiB the checked address is below 2^31 iff
#   2 MiB page: vaddr[20:19] == 00   (the first 512 KiB of every 2 MiB page)
#   1 GiB page: vaddr[29:19] == 0    (the first 512 KiB of every 1 GiB page)
# and such a load is held until it is the oldest instruction with no store in flight (stall_ni). Performance only: the
# cache tag uses the correct paddr.
#
# THE SIM CANNOT SEE IT BY DEFAULT: the simulation config has no NI rule, so NonIdemPotenceEn is 0 and stall_ni is
# constant 0. +define+CAPSTONE_FPGA_NI gives the simulation the FPGA's rule; without the define this test is the
# instrument-off control. +define+SUP_TRACE prints one "SUP NI-ABORT vaddr=..." line per entry into
# ABORT_TRANSACTION_NI, which is the primary reading; cycles are the secondary one.
#
# SCAFFOLD (r34-misaligned-xlate.S): execution stays in M-mode, DATA accesses are translated by MPRV=1 with MPP=S, set
# only around each timed loop. PMP one NAPOT entry RWX. Page tables:
#   root[2] = 1 GiB leaf, identity for 0x8000_0000.. (code, data, the buffer)
#   root[3] -> l1;  l1[1] = 2 MiB leaf  VA 0xC020_0000 -> PA 0x8020_0000
#                   l1[2] -> l0;  l0[off>>12] = 4 KiB leaf VA 0xC040_0000+off -> PA 0x8020_0000+off, for the four offsets
# Every arm reads one 64-byte line whose doubleword 0 holds its own PADDR (written untranslated in setup); the arm prints
# its cycles over NITER iterations and the last a0, which must equal that paddr, or the arm read through a wrong
# mapping and is VOID.
#
# ARMS (shape L = four independent loads per iteration; LS = the same with one store between them). Offsets in the
# 2 MiB page differ only in bits 20:19, and all share the low 18 bits (same D-cache set).
#   #  name     VA                 PA           shape  NI predicted (rule on, current RTL)
#   1  2M-00    0xC024_0000        0x8024_0000  L      YES  (vaddr[20:19] = 00)
#   2  2M-01    0xC02C_0000        0x802C_0000  L      no
#   3  2M-10    0xC034_0000        0x8034_0000  L      no   (the "vaddr[19] = 0 alone" reading predicts YES)
#   4  2M-11    0xC03C_0000        0x803C_0000  L      no
#   5  4K-00    0xC044_0000        0x8024_0000  L      no   (same PA as arm 1: 4 KiB pages are not spliced)
#   6  4K-01    0xC04C_0000        0x802C_0000  L      no
#   7  4K-10    0xC054_0000        0x8034_0000  L      no
#   8  4K-11    0xC05C_0000        0x803C_0000  L      no
#   9  1G       0x8024_0100        0x8024_0100  L      no   (vaddr[29:19] = 4)
#  10  1G-low   buf (0x8000_xxxx)  = VA         L      YES  (vaddr[29:19] = 0)
#  11  bare     0x8024_0180 MPRV=0 = VA         L      no   (untranslated: ppn = vaddr, correct)
#  12  ROM      0x0001_0000 MPRV=0 = VA         L      YES, genuinely (bootrom is inside [0, 0x8000_0000)):
#                                                            the positive control for the rule, independent of the splice
#  13  S2M-00   0xC024_0200        0x8024_0200  LS     YES
#  14  S2M-01   0xC02C_0200        0x802C_0200  LS     no
#  15  S4K-00   0xC044_0200        0x8024_0200  LS     no
#
# THE REVERSE DIRECTION (arms 16-17, added 2026-11-06 after the first three models ran and before their own run, on the
# claim-auditor's point): the same splice can make a GENUINELY non-idempotent address read as idempotent. The vaddr's
# page offset equals the paddr's, so it needs a device address whose own bits [29:19] (1 GiB page) are not all zero:
# CLINT at 0x0200_0000 (bit 25). root[1] = 1 GiB leaf VA 0x4000_0000 -> PA 0; the spliced probe reads
# ppn[29:12] = 0x02004, i.e. an address at or above 2^31, so the load is NOT held and runs speculatively. Arms 1-15 keep
# their code positions (16-17 are appended).
#  16  CLINT    0x0200_4000 MPRV=0 = VA         L      YES, genuinely (the control for arm 17)
#  17  CLINT-1G 0x4200_4000        0x0200_4000  L      current RTL: NO (the hazard); fixed RTL: YES, as arm 16
#
# PREDICTIONS, written before the first run:
#   rule ON, current RTL : NI-ABORT lines >= NITER (32) on arms 1, 10, 12, 13 and 0 on every other arm; arm 1's cycles
#                          >= 1.5 x arm 5's, arm 13's >= 1.5 x arm 15's, arm 10's >= 1.5 x arm 9's.
#   rule OFF (no define) : 0 NI-ABORT lines on every arm, the ROM arm included; arms 1-8 within 10 % of each other.
#   rule ON, fixed RTL   : NI-ABORT lines only on arm 12 (>= NITER); arm 1's cycles within 10 % of arm 5's, arm 10's
#                          within 10 % of arm 9's.
#   arms 16-17 (added later, predicted before their run): rule ON, current RTL: arm 16 >= NITER NI-ABORT lines, arm 17
#                          0; rule ON, fixed RTL: both >= NITER; rule OFF: both 0.
# Every arm's a0 = its paddr (ROM and CLINT arms: printed, not graded); traps 0; the test ends PASS.
#
# This file goes through CPP: no comment line may begin with a hash followed by a directive word, and a MACRO form with
# parentheses inside a comment would expand.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define NITER 32

.option norvc

.macro BODY st
  ld   a0, 0(t1)
  ld   a1, 8(t1)
  .if \st
  sd   t2, 64(t1)
  .endif
  ld   a2, 16(t1)
  ld   a3, 24(t1)
.endm

# t1 = the arm's address on entry. mprv = translate the loop's data accesses. st = shape LS.
.macro ARM mprv, st
  .if \mprv
  li   t5, 0x20000
  csrs mstatus, t5              # MPRV = 1 (MPP = S): data accesses translate
  .endif
  li   t2, 1
  BODY \st                      # warm pass: DTLB and D-cache
  fence
  li   t2, NITER
  csrr s2, mcycle
1:
  BODY \st
  addi t2, t2, -1
  bnez t2, 1b
  csrr s3, mcycle
  li   t5, 0x20000
  csrc mstatus, t5              # MPRV = 0
  sub  s2, s3, s2
  CAPPRINT(s2)                  # cycles over NITER iterations
  CAPPRINT(a0)                  # must be the arm's paddr
.endm

.section .text.init
.globl _start
_start:

  li gp, 1                      # TESTNUM non-zero or RVTEST_FAIL spins instead of reporting
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                   # last cause seen
  li   s11, 0                   # trap count

# ---------------------------------------------------------------- PMP: allow S-mode data accesses (MPRV with MPP=S)
  li   t0, -1
  csrw pmpaddr0, t0
  li   t0, 0x1F                 # NAPOT | X | W | R
  csrw pmpcfg0, t0

# ---------------------------------------------------------------- the lines each arm reads hold their own paddr
  li   t3, 0x80200000
  li   t0, 0x40000
  add  t1, t3, t0
  sd   t1, 0(t1)                # 0x80240000
  addi t4, t1, 0x100
  sd   t4, 0(t4)                # 0x80240100 (1G arm)
  addi t4, t1, 0x180
  sd   t4, 0(t4)                # 0x80240180 (bare arm)
  addi t4, t1, 0x200
  sd   t4, 0(t4)                # 0x80240200 (S arms 13, 15)
  li   t0, 0xC0000
  add  t1, t3, t0
  sd   t1, 0(t1)                # 0x802C0000
  addi t4, t1, 0x200
  sd   t4, 0(t4)                # 0x802C0200 (S arm 14)
  li   t0, 0x140000
  add  t1, t3, t0
  sd   t1, 0(t1)                # 0x80340000
  li   t0, 0x1C0000
  add  t1, t3, t0
  sd   t1, 0(t1)                # 0x803C0000
  la   t1, buf
  sd   t1, 0(t1)                # buf (1G-low arm)
  fence

# ---------------------------------------------------------------- page tables
  la   t0, root_pt
  li   t2, 0x80000              # root[2]: 1 GiB identity leaf, DAGU_XWRV = 0xCF (U = 0)
  slli t2, t2, 10
  ori  t2, t2, 0xCF
  sd   t2, 16(t0)
  la   t1, l1_pt                # root[3] -> l1
  srli t2, t1, 12
  slli t2, t2, 10
  ori  t2, t2, 1
  sd   t2, 24(t0)
  li   t2, 0x80200              # l1[1]: 2 MiB leaf VA 0xC0200000 -> PA 0x80200000 (PPN[0] = 0: aligned)
  slli t2, t2, 10
  ori  t2, t2, 0xCF
  sd   t2, 8(t1)
  la   t3, l0_pt                # l1[2] -> l0
  srli t2, t3, 12
  slli t2, t2, 10
  ori  t2, t2, 1
  sd   t2, 16(t1)
  li   t4, 0x40                 # l0[off >> 12] = 4 KiB leaf -> PA 0x80200000 + off, off in {0x40000, 0xC0000, 0x140000, 0x1C0000}
  li   t6, 4
2:
  li   t2, 0x80200
  add  t2, t2, t4
  slli t2, t2, 10
  ori  t2, t2, 0xCF
  slli t5, t4, 3
  add  t5, t5, t3
  sd   t2, 0(t5)
  addi t4, t4, 0x80
  addi t6, t6, -1
  bnez t6, 2b

  srli t3, t0, 12               # satp = sv39 | root_pt >> 12
  li   t4, 8
  slli t4, t4, 60
  or   t3, t3, t4
  csrw satp, t3
  sfence.vma
  li   t5, 0x1800
  csrc mstatus, t5
  li   t5, 0x0800
  csrs mstatus, t5              # MPP = S (MPRV is set per arm)
  csrr s4, mstatus
  CAPPRINT(s4)                  # witness: MPP = S, MPRV = 0 here

# ---------------------------------------------------------------- the arms
  li   t1, 0xC0240000           # 1  2M-00
  ARM 1, 0
  li   t1, 0xC02C0000           # 2  2M-01
  ARM 1, 0
  li   t1, 0xC0340000           # 3  2M-10
  ARM 1, 0
  li   t1, 0xC03C0000           # 4  2M-11
  ARM 1, 0
  li   t1, 0xC0440000           # 5  4K-00
  ARM 1, 0
  li   t1, 0xC04C0000           # 6  4K-01
  ARM 1, 0
  li   t1, 0xC0540000           # 7  4K-10
  ARM 1, 0
  li   t1, 0xC05C0000           # 8  4K-11
  ARM 1, 0
  li   t1, 0x80240100           # 9  1G
  ARM 1, 0
  la   t1, buf                  # 10 1G-low
  ARM 1, 0
  li   t1, 0x80240180           # 11 bare
  ARM 0, 0
  li   t1, 0x10000              # 12 ROM
  ARM 0, 0
  li   t1, 0xC0240200           # 13 S2M-00
  ARM 1, 1
  li   t1, 0xC02C0200           # 14 S2M-01
  ARM 1, 1
  li   t1, 0xC0440200           # 15 S4K-00
  ARM 1, 1
  li   t1, 0x02004000           # 16 CLINT, untranslated
  ARM 0, 0
  la   t0, root_pt              # root[1]: 1 GiB leaf VA 0x40000000 -> PA 0 (written here so arms 1-15 keep their addresses)
  li   t2, 0xCF
  sd   t2, 8(t0)
  sfence.vma
  li   t1, 0x42004000           # 17 CLINT through a 1 GiB page
  ARM 1, 0

  csrw satp, x0
  CAPPRINT(s11)                 # traps over the whole run: must be 0
  j    pass

# A trap is DATA: record it, step over the faulting instruction, continue.
.align 4
trap_handler:
  csrr t0, mcause
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

# KIT EDIT (silchk-b1-2026-11-21, the one change from capstone-ariane d83d3a633's copy): both exits end through the
# board wrapper's CAP_PASS, which prints board_rec over the UART. RVTEST_PASS/FAIL write tohost, which silicon never
# reads. A failing run therefore prints a SHORTER vector than the pre-registered SR, and that is how it is graded.
fail:
  CAP_PASS(x25)
pass:
  CAP_PASS(x25)

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

.align 12
root_pt:
  .zero 4096
l1_pt:
  .zero 4096
l0_pt:
  .zero 4096

.align 6
buf:
  .zero 64

RVTEST_DATA_END
