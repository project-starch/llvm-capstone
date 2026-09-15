# See LICENSE for license details.

#*****************************************************************************
# r34-misaligned-xlate.S
#-----------------------------------------------------------------------------
#
# ARE THE MISALIGNED CAUSES DELIVERED WHEN ADDRESS TRANSLATION IS ON? R-34 shows that the
# load/store unit's own exceptions are RAISED AND DROPPED when the access is granted in its
# request cycle. That was measured with translation OFF. R-34's residual (a) asks for the
# translation-on path, which is argued from source only.
#
# WHAT THIS CAN AND CANNOT ANSWER, because the residual's wording is wider than any test can be.
# It covers the MISALIGNED causes (4 and 6) ONLY. It says NOTHING about the capability clauses,
# and no test can, because translation-on and that block's gate are mutually exclusive:
# data-access translation via MPRV requires mstatus.MPP below M (csr_regfile.sv:2292), the
# load/store privilege is then that same MPP (:2298), and cap_violation_detection requires it to
# be M. Turning translation on therefore takes the gate out of its satisfied state, and S-mode
# fails it on privilege anyway. The misaligned causes are not gated on capmode or privilege at
# all, which is exactly why they remain reachable here.
#
# THE CONFIGURATION. Execution stays in M-mode so the harness, the trap vector and tohost behave
# normally; only DATA accesses are translated, via MPRV=1 with MPP=S and an sv39 identity map.
# One root page table entry is enough: VA 0x80000000 has VPN[2] = 2, so root[2] as a 1 GiB
# megapage covers 0x80000000..0xC0000000, which contains the code, the stack, the page table
# itself and the buffer.
#
#   ARM X0  control: misaligned ld with translation OFF (MPRV=0)
#           the known R-34 behaviour -- expect NO trap, a shifted value
#   ARM X1  unknown: the SAME misaligned ld with translation ON (MPRV=1, MPP=S, satp sv39)
#           expect NO trap if the drop is independent of translation
#
# THE PRECONDITION IS WITNESSED, NOT ASSUMED. Before ARM X1 reads anything, an ALIGNED load
# through the translated path must return the sentinel. If it does not, translation is not
# working -- a page fault, a bad PTE, an unsupported satp mode -- and ARM X1's silence would mean
# "the mapping is broken", not "the exception was dropped". That distinction is the whole test:
# without it a null result here is unreadable, which is the failure shape this project keeps
# paying for.
#
# FIRST RUN (2026-09-15) WAS REFUSED BY ITS OWN PRECONDITION, and that is why the PMP block below
# exists: the aligned load through the translated path returned 0 with cause 5 (access fault), so
# translation was not working and ARM X1's cause 4 was read under a broken mapping. Recorded rather
# than quietly fixed -- without the precondition that run would have been reported as "misaligned
# IS delivered under translation", which it does not show.
#
# READ THE PAIR. X0 and X1 both silent -> the delivery defect is INDEPENDENT of translation, and
# the MMU forwarding unregistered either way is the explanation. X0 silent and X1 trapping ->
# translation changes delivery, and R-34's mechanism is narrower than stated.
#
# It reports rather than judges: every arm prints its value and the cause its handler saw, and
# the test always reaches PASS. A trap vector is installed so a fault reports instead of vanishing.
#
# This file goes through CPP: no comment line may begin with a hash followed by a directive word,
# and a MACRO form with parentheses inside a comment would expand.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li gp, 1                      # TESTNUM non-zero or RVTEST_FAIL spins instead of reporting
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                   # last cause seen
  li   s11, 0                   # trap count

# ---------------------------------------------------------------- ARM X0: translation OFF
  la   t1, buf
  addi t1, t1, 1                # deliberately misaligned
  li   s2, 0
  ld   s2, 0(t1)                # misaligned ld, MPRV=0: the known R-34 case
  mv   s3, s10                  # cause seen (expect 0 = dropped)
  li   s10, 0

# ---------------------------------------------------------------- PMP: allow S-mode accesses
# With MPRV and MPP=S the effective privilege for DATA accesses is S, so PMP applies to them where
# it did not before. The `p` environment never leaves M-mode and so configures none, and with no
# matching entry an S-mode access is DENIED -- which is exactly what the first run of this test
# read: the precondition load took cause 5 (access fault), not a page fault, and ARM X1's cause 4
# was therefore taken with the mapping not working. One NAPOT entry covering everything, RWX.
  li   t0, -1
  csrw pmpaddr0, t0
  li   t0, 0x1F                 # A = NAPOT (0x18) | X | W | R
  csrw pmpcfg0, t0

# ---------------------------------------------------------------- build the sv39 identity map
# root[2] = megapage for 0x80000000..0xC0000000, PPN = 0x80000, flags DAGU_XWRV = 0xCF
  la   t0, root_pt
  li   t2, 0x80000
  slli t2, t2, 10
  ori  t2, t2, 0xCF
  sd   t2, 16(t0)               # entry index 2 -> byte offset 16

  srli t3, t0, 12               # satp PPN = root_pt >> 12
  li   t4, 8
  slli t4, t4, 60               # MODE = 8 (sv39)
  or   t3, t3, t4
  csrw satp, t3
  sfence.vma

# MPRV = 1 with MPP = S, so DATA accesses translate while we keep executing in M-mode
  li   t5, 0x1800
  csrc mstatus, t5              # clear MPP
  li   t5, 0x0800
  csrs mstatus, t5              # MPP = 01 (S)
  li   t5, 0x20000
  csrs mstatus, t5              # MPRV = 1
  csrr s4, mstatus              # witness: MPRV and MPP as set

# ---------------------------------------------------------------- precondition: translation WORKS
  la   t1, buf
  li   s5, 0
  ld   s5, 0(t1)                # ALIGNED load through the translated path
  mv   s6, s10                  # must be 0, and s5 must be the sentinel
  li   s10, 0

# ---------------------------------------------------------------- ARM X1: translation ON
  la   t1, buf
  addi t1, t1, 1
  li   s7, 0
  ld   s7, 0(t1)                # the SAME misaligned ld, now translated
  mv   s8, s10                  # cause seen
  li   s10, 0

  li   t5, 0x20000
  csrc mstatus, t5              # MPRV = 0 again before reporting
  csrw satp, x0

  CAPPRINT(s2)                  # X0 value   (translation off)
  CAPPRINT(s3)                  # X0 cause   (expect 0)
  CAPPRINT(s4)                  # mstatus witness: MPRV set, MPP = S
  CAPPRINT(s5)                  # precondition value: MUST be the sentinel or X1 is VOID
  CAPPRINT(s6)                  # precondition cause: must be 0
  CAPPRINT(s7)                  # X1 value   (translation on)
  CAPPRINT(s8)                  # X1 cause   -- the result
  CAPPRINT(s11)                 # total traps over the whole run
  j    pass

# -----------------------------------------------------------------------
# A trap is DATA: record it, step over the faulting instruction, continue.
# -----------------------------------------------------------------------
.align 4
trap_handler:
  csrr t0, mcause
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4                # norvc, so every instruction is 4 bytes
  csrw mepc, t0
  mret

fail:
  RVTEST_FAIL
pass:
  RVTEST_PASS

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

.align 12
root_pt:
  .zero 4096

.align 4
buf:
  .dword 0x4C535550             # "LSUP"
  .dword 0x0BADB0B0

RVTEST_DATA_END
